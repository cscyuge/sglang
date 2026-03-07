# SPDX-License-Identifier: Apache-2.0
"""
FlashTalk WanVideo DiT model.

Extends WanTransformerBlock and WanTransformer3DModel with audio cross-attention
layers for audio-driven talking face video generation.
"""

import math

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.dits import WanVideoConfig
from sglang.multimodal_gen.configs.models.dits.wanvideo import (
    FlashTalkWanVideoArchConfig,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    get_sp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.attention.layer import LocalAttention
from sglang.multimodal_gen.runtime.layers.layernorm import (
    FP32LayerNorm,
    LayerNormScaleShift,
    tensor_parallel_rms_norm,
)
from sglang.multimodal_gen.runtime.layers.layernorm import _ensure_contiguous
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    NDRotaryEmbedding,
    _apply_rotary_emb,
    apply_flashinfer_rope_qk_inplace,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import (
    PatchEmbed,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.models.dits.flashtalk_attention import (
    FlashTalkAudioCrossAttention,
)
from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    WanTimeTextImageEmbedding,
    WanTransformer3DModel,
    WanTransformerBlock,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
_is_cuda = current_platform.is_cuda()


class FlashTalkWanTransformerBlock(WanTransformerBlock):
    """WanTransformerBlock extended with audio cross-attention.

    Forward flow: self_attn -> text_cross_attn -> audio_cross_attn -> ffn
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        attention_type: str = "original",
        sla_topk: float = 0.1,
        # FlashTalk audio params
        audio_dim: int = 768,
        audio_qk_norm: bool = False,
        class_range: int = 24,
        class_interval: int = 4,
    ):
        super().__init__(
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            supported_attention_backends=supported_attention_backends,
            prefix=prefix,
            attention_type=attention_type,
            sla_topk=sla_topk,
        )

        # Audio cross-attention layer (inserted between text cross-attn and FFN)
        self.audio_cross_attn = FlashTalkAudioCrossAttention(
            dim=dim,
            audio_dim=audio_dim,
            num_heads=num_heads,
            qk_norm=audio_qk_norm,
            eps=eps,
            class_range=class_range,
            class_interval=class_interval,
        )
        # LayerNorm before audio cross-attention
        self.norm_audio = FP32LayerNorm(dim, elementwise_affine=True)

        # Text/image cross-attention context is replicated on all SP ranks.
        # USP all-to-all is wasteful here (creates sp_size-x KV expansion).
        # Replace with local attention that skips all-to-all entirely.
        if get_sp_world_size() > 1:
            self.attn2.attn = LocalAttention(
                num_heads=self.attn2.local_num_heads,
                head_size=self.attn2.head_dim,
                causal=False,
                supported_attention_backends=supported_attention_backends,
            )

        # Pre-allocate null tensor for residual norm (avoids per-forward allocation)
        self.register_buffer("_null_param", torch.zeros(1), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, ...],
        audio_context: torch.Tensor | None = None,
        grid_shape: tuple[int, int, int] | None = None,
        human_num: int = 1,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        orig_dtype = hidden_states.dtype

        if temb.dim() == 4:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            e = self.scale_shift_table + temb.float()
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                e.chunk(6, dim=1)
            )

        assert shift_msa.dtype == torch.float32

        # 1. Self-attention
        norm_hidden_states = self.norm1(hidden_states, shift_msa, scale_msa)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        if self.norm_q is not None:
            if self.tp_rmsnorm:
                query = tensor_parallel_rms_norm(query, self.norm_q)
            else:
                query = self.norm_q(query)
        if self.norm_k is not None:
            if self.tp_rmsnorm:
                key = tensor_parallel_rms_norm(key, self.norm_k)
            else:
                key = self.norm_k(key)
        query = query.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))
        key = key.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))
        value = value.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))

        # Apply rotary embeddings (cos_sin_cache pre-computed at model level)
        if len(freqs_cis) == 3:
            cos, sin, cos_sin_cache = freqs_cis
        else:
            cos, sin = freqs_cis
            cos_sin_cache = None

        if cos_sin_cache is not None and _is_cuda and query.shape == key.shape:
            query, key = apply_flashinfer_rope_qk_inplace(
                query, key, cos_sin_cache, is_neox=False
            )
        elif _is_cuda and query.shape == key.shape:
            cos_sin_cache = torch.cat(
                [
                    cos.to(dtype=torch.float32).contiguous(),
                    sin.to(dtype=torch.float32).contiguous(),
                ],
                dim=-1,
            )
            query, key = apply_flashinfer_rope_qk_inplace(
                query, key, cos_sin_cache, is_neox=False
            )
        else:
            query, key = _apply_rotary_emb(
                query, cos, sin, is_neox_style=False
            ), _apply_rotary_emb(key, cos, sin, is_neox_style=False)

        attn_output = self.attn1(query, key, value)
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = self._null_param.to(dtype=hidden_states.dtype)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale
        )
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype
        ), hidden_states.to(orig_dtype)

        # 2. Text cross-attention
        attn_output = self.attn2(
            norm_hidden_states, context=encoder_hidden_states, context_lens=None
        )
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa
        )
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype
        ), hidden_states.to(orig_dtype)

        # 3. Audio cross-attention (FlashTalk addition)
        if audio_context is not None and grid_shape is not None:
            audio_input = self.norm_audio(hidden_states)
            audio_output = self.audio_cross_attn(
                audio_input,
                encoder_hidden_states=audio_context,
                shape=grid_shape,
                human_num=human_num,
            )
            hidden_states = hidden_states + audio_output

            # Recompute FFN input norm to include audio contribution.
            # Original FlashTalk: ffn(norm2(x) * (1 + scale) + shift) where x
            # includes audio output. Without this, the FFN misses the audio
            # signal, causing progressive quality degradation in multi-chunk
            # generation.
            # Use the fused CuTe DSL kernel (norm + scale_shift in one pass)
            # instead of separate FP32LayerNorm + manual scale_shift, which
            # would generate ~8 small kernels per block (bf16→fp32, layer_norm,
            # fp32→bf16, bf16→fp32 again, add, mul, add, fp32→bf16).
            from sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift import (
                fused_norm_scale_shift,
            )

            norm_hidden_states = fused_norm_scale_shift(
                hidden_states.contiguous(),
                _ensure_contiguous(
                    getattr(self.cross_attn_residual_norm.norm, "weight", None)
                ),
                _ensure_contiguous(
                    getattr(self.cross_attn_residual_norm.norm, "bias", None)
                ),
                c_scale_msa.contiguous(),
                c_shift_msa.contiguous(),
                self.cross_attn_residual_norm.norm_type,
                self.cross_attn_residual_norm.eps,
            )

        # 4. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(ff_output, c_gate_msa, hidden_states)
        hidden_states = hidden_states.to(orig_dtype)

        return hidden_states


class FlashTalkWanTransformer3DModel(WanTransformer3DModel):
    """WanTransformer3DModel extended for FlashTalk with audio cross-attention.

    Uses FlashTalkWanTransformerBlock instead of WanTransformerBlock,
    passes audio_context through to each block.
    """

    # Override with FlashTalk original->sglang mapping (instead of diffusers->sglang)
    param_names_mapping = FlashTalkWanVideoArchConfig().param_names_mapping

    def __init__(self, config: WanVideoConfig, hf_config: dict | None = None) -> None:
        # We override __init__ to use FlashTalkWanTransformerBlock
        # Call grandparent init to skip WanTransformer3DModel's block creation
        CachableDiT.__init__(self, config, hf_config=hf_config or {})

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.inner_dim = inner_dim
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = inner_dim
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(
            in_chans=config.in_channels,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
            flatten=False,
        )

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. FlashTalk Transformer blocks (with audio cross-attention)
        audio_dim = getattr(config, "audio_dim", 768)

        self.blocks = nn.ModuleList(
            [
                FlashTalkWanTransformerBlock(
                    inner_dim,
                    config.ffn_dim,
                    config.num_attention_heads,
                    config.qk_norm,
                    config.cross_attn_norm,
                    config.eps,
                    config.added_kv_proj_dim,
                    self._supported_attention_backends,
                    prefix=f"{config.prefix}.blocks.{i}",
                    attention_type=config.attention_type,
                    sla_topk=config.sla_topk,
                    audio_dim=audio_dim,
                )
                for i in range(config.num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = LayerNormScaleShift(
            inner_dim,
            eps=config.eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )
        self.proj_out = ColumnParallelLinear(
            inner_dim,
            config.out_channels * math.prod(config.patch_size),
            bias=True,
            gather_output=True,
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        self.cnt = 0
        self.__post_init__()

        self.sp_size = get_sp_world_size()

        d = self.hidden_size // self.num_attention_heads
        self.rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        self.rotary_emb = NDRotaryEmbedding(
            rope_dim_list=self.rope_dim_list,
            rope_theta=10000,
            dtype=(
                torch.float32
                if current_platform.is_mps() or current_platform.is_musa()
                else torch.float64
            ),
        )

        self.layer_names = ["blocks"]

    def _init_teacache_state(self) -> None:
        """FlashTalk disables model-level TeaCache.

        Cross-chunk caching at the transformer residual level doesn't work
        because hidden_states differ completely across chunks (different noise
        and motion latent). Instead, FlashTalk uses adaptive step reduction
        at the denoising stage level — see FlashTalkDenoisingStage.
        """
        super()._init_teacache_state()

    def should_skip_forward_for_cached_states(self, **kwargs) -> bool:
        """Always compute — FlashTalk handles caching at the pipeline level."""
        return False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        audio_context: torch.Tensor | None = None,
        human_num: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        forward_batch = get_forward_context().forward_batch
        if forward_batch is not None:
            sequence_shard_enabled = (
                forward_batch.enable_sequence_shard and self.sp_size > 1
            )
        else:
            sequence_shard_enabled = False
        self.enable_teacache = (
            forward_batch is not None and forward_batch.enable_teacache
        )

        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image, list):
            encoder_hidden_states_image = (
                encoder_hidden_states_image[0]
                if len(encoder_hidden_states_image) > 0
                else None
            )
        elif not isinstance(encoder_hidden_states_image, torch.Tensor):
            encoder_hidden_states_image = None

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        if not sequence_shard_enabled:
            freqs_cos, freqs_sin = self.rotary_emb.forward_from_grid(
                (
                    post_patch_num_frames * self.sp_size,
                    post_patch_height,
                    post_patch_width,
                ),
                shard_dim=0,
                start_frame=0,
                device=hidden_states.device,
            )
            assert freqs_cos.dtype == torch.float32
            assert freqs_cos.device == hidden_states.device
            cos_sin_cache = torch.cat(
                [freqs_cos.contiguous(), freqs_sin.contiguous()], dim=-1
            )
            freqs_cis = (freqs_cos.float(), freqs_sin.float(), cos_sin_cache)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        seq_len_orig = hidden_states.shape[1]
        seq_shard_pad = 0
        if sequence_shard_enabled:
            if seq_len_orig % self.sp_size != 0:
                seq_shard_pad = self.sp_size - (seq_len_orig % self.sp_size)
                pad = torch.zeros(
                    (batch_size, seq_shard_pad, hidden_states.shape[2]),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                hidden_states = torch.cat([hidden_states, pad], dim=1)
            sp_rank = get_sp_group().rank_in_group
            local_seq_len = hidden_states.shape[1] // self.sp_size
            hidden_states = hidden_states.view(
                batch_size, self.sp_size, local_seq_len, hidden_states.shape[2]
            )
            hidden_states = hidden_states[:, sp_rank, :, :]

            frame_stride = post_patch_height * post_patch_width
            freqs_cos, freqs_sin = self._compute_rope_for_sequence_shard(
                local_seq_len,
                sp_rank,
                frame_stride,
                post_patch_width,
                hidden_states.device,
            )
            cos_sin_cache = torch.cat(
                [freqs_cos.float().contiguous(), freqs_sin.float().contiguous()],
                dim=-1,
            )
            freqs_cis = (freqs_cos.float(), freqs_sin.float(), cos_sin_cache)

        # Timestep embedding
        if timestep.dim() == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                timestep_seq_len=ts_seq_len,
            )
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        encoder_hidden_states = (
            encoder_hidden_states.to(orig_dtype)
            if not current_platform.is_amp_supported()
            else encoder_hidden_states
        )

        # Grid shape for audio cross-attention
        grid_shape = (post_patch_num_frames, post_patch_height, post_patch_width)

        # Ensure consistent dtype for GEMM — inputs from different
        # encoders (text fp32, image fp16) may differ, but attention requires
        # q/k/v to match dtype. Use the model's compute dtype (from patch_embedding).
        param_dtype = orig_dtype
        hidden_states = hidden_states.to(param_dtype)
        encoder_hidden_states = encoder_hidden_states.to(param_dtype)
        if audio_context is not None:
            audio_context = audio_context.to(param_dtype)

        # 4. Transformer blocks with audio context
        should_skip_forward = self.should_skip_forward_for_cached_states(
            timestep_proj=timestep_proj, temb=temb
        )

        if should_skip_forward:
            hidden_states = self.retrieve_cached_states(hidden_states)
        else:
            if self.enable_teacache:
                original_hidden_states = hidden_states.clone()

            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    freqs_cis,
                    audio_context=audio_context,
                    grid_shape=grid_shape,
                    human_num=human_num,
                )

            if self.enable_teacache:
                self.maybe_cache_states(hidden_states, original_hidden_states)
        self.cnt += 1

        if sequence_shard_enabled:
            hidden_states = hidden_states.contiguous()
            hidden_states = sequence_model_parallel_all_gather(hidden_states, dim=1)
            if seq_shard_pad > 0:
                hidden_states = hidden_states[:, :seq_len_orig, :]

        # 5. Output norm, projection & unpatchify
        if temb.dim() == 3:
            shift, scale = (
                self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)
            ).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states, _ = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return output
