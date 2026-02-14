# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_blocks(n: str, m) -> bool:
    return "blocks" in n and str.isdigit(n.split(".")[-1])


@dataclass
class WanVideoArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks])

    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^patch_embedding\.(.*)$": r"patch_embedding.proj.\1",
            r"^condition_embedder\.text_embedder\.linear_1\.(.*)$": r"condition_embedder.text_embedder.fc_in.\1",
            r"^condition_embedder\.text_embedder\.linear_2\.(.*)$": r"condition_embedder.text_embedder.fc_out.\1",
            r"^condition_embedder\.time_embedder\.linear_1\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_in.\1",
            r"^condition_embedder\.time_embedder\.linear_2\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_out.\1",
            r"^condition_embedder\.time_proj\.(.*)$": r"condition_embedder.time_modulation.linear.\1",
            r"^condition_embedder\.image_embedder\.ff\.net\.0\.proj\.(.*)$": r"condition_embedder.image_embedder.ff.fc_in.\1",
            r"^condition_embedder\.image_embedder\.ff\.net\.2\.(.*)$": r"condition_embedder.image_embedder.ff.fc_out.\1",
            r"^blocks\.(\d+)\.attn1\.to_q\.(.*)$": r"blocks.\1.to_q.\2",
            r"^blocks\.(\d+)\.attn1\.to_k\.(.*)$": r"blocks.\1.to_k.\2",
            r"^blocks\.(\d+)\.attn1\.to_v\.(.*)$": r"blocks.\1.to_v.\2",
            r"^blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$": r"blocks.\1.to_out.\2",
            r"^blocks\.(\d+)\.attn1\.norm_q\.(.*)$": r"blocks.\1.norm_q.\2",
            r"^blocks\.(\d+)\.attn1\.norm_k\.(.*)$": r"blocks.\1.norm_k.\2",
            r"^blocks\.(\d+)\.attn1\.attn_op\.local_attn\.proj_l\.(.*)$": r"blocks.\1.attn1.local_attn.proj_l.\2",
            r"^blocks\.(\d+)\.attn2\.to_out\.0\.(.*)$": r"blocks.\1.attn2.to_out.\2",
            r"^blocks\.(\d+)\.ffn\.net\.0\.proj\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
            r"^blocks\.(\d+)\.ffn\.net\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
            r"^blocks\.(\d+)\.norm2\.(.*)$": r"blocks.\1.self_attn_residual_norm.norm.\2",
        }
    )

    reverse_param_names_mapping: dict = field(default_factory=lambda: {})

    # Some LoRA adapters use the original official layer names instead of hf layer names,
    # so apply this before the param_names_mapping
    lora_param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.attn1.to_q.\2",
            r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.attn1.to_k.\2",
            r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.attn1.to_v.\2",
            r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.attn1.to_out.0.\2",
            r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
            r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
            r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": r"blocks.\1.attn2.to_out.0.\2",
            r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
            r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
        }
    )

    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len = 512
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 40
    cross_attn_norm: bool = True
    qk_norm: str = "rms_norm_across_heads"
    eps: float = 1e-6
    image_dim: int | None = None
    added_kv_proj_dim: int | None = None
    rope_max_seq_len: int = 1024
    pos_embed_seq_len: int | None = None
    exclude_lora_layers: list[str] = field(default_factory=lambda: ["embedder"])

    # Wan MoE
    boundary_ratio: float | None = None

    # Causal Wan
    local_attn_size: int = (
        -1
    )  # Window size for temporal local attention (-1 indicates global attention)
    sink_size: int = (
        0  # Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
    )
    num_frames_per_block: int = 3
    sliding_window_num_frames: int = 21
    attention_type: str = "original"
    sla_topk: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class WanVideoConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=WanVideoArchConfig)

    prefix: str = "Wan"


@dataclass
class FlashTalkWanVideoArchConfig(WanVideoArchConfig):
    """Extended WanVideo arch config for FlashTalk with audio cross-attention.

    Overrides in_channels/image_dim for FlashTalk's I2V architecture and
    provides a complete param_names_mapping from FlashTalk's original checkpoint
    keys (non-diffusers naming) to sglang internal names.
    """

    has_audio_cross_attn: bool = True
    audio_dim: int = 768
    audio_context_tokens: int = 32

    # FlashTalk I2V: 16 latent + 16 condition + 4 mask
    in_channels: int = 36
    # CLIP ViT-Huge output dimension for WanImageEmbedding
    image_dim: int = 1280
    # Enable I2V cross-attention (k_img/v_img projections for CLIP features)
    added_kv_proj_dim: int = 5120

    # Complete mapping from FlashTalk original checkpoint keys to sglang model keys
    param_names_mapping: dict = field(
        default_factory=lambda: {
            # Embeddings
            r"^patch_embedding\.(.*)$": r"patch_embedding.proj.\1",
            r"^time_embedding\.0\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_in.\1",
            r"^time_embedding\.2\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_out.\1",
            r"^time_projection\.1\.(.*)$": r"condition_embedder.time_modulation.linear.\1",
            r"^text_embedding\.0\.(.*)$": r"condition_embedder.text_embedder.fc_in.\1",
            r"^text_embedding\.2\.(.*)$": r"condition_embedder.text_embedder.fc_out.\1",
            r"^img_emb\.proj\.0\.(.*)$": r"condition_embedder.image_embedder.norm1.\1",
            r"^img_emb\.proj\.1\.(.*)$": r"condition_embedder.image_embedder.ff.fc_in.\1",
            r"^img_emb\.proj\.3\.(.*)$": r"condition_embedder.image_embedder.ff.fc_out.\1",
            r"^img_emb\.proj\.4\.(.*)$": r"condition_embedder.image_embedder.norm2.\1",
            # Output head
            r"^head\.head\.(.*)$": r"proj_out.\1",
            r"^head\.modulation$": r"scale_shift_table",
            # Self-attention: original self_attn.{q,k,v,o} -> sglang to_{q,k,v}, to_out
            r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.to_q.\2",
            r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.to_k.\2",
            r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.to_v.\2",
            r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.to_out.\2",
            r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$": r"blocks.\1.norm_q.\2",
            r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$": r"blocks.\1.norm_k.\2",
            # Cross-attention: original cross_attn.* -> sglang attn2.*
            r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
            r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
            r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": r"blocks.\1.attn2.to_out.\2",
            r"^blocks\.(\d+)\.cross_attn\.k_img\.(.*)$": r"blocks.\1.attn2.add_k_proj.\2",
            r"^blocks\.(\d+)\.cross_attn\.v_img\.(.*)$": r"blocks.\1.attn2.add_v_proj.\2",
            r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$": r"blocks.\1.attn2.norm_q.\2",
            r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$": r"blocks.\1.attn2.norm_k.\2",
            r"^blocks\.(\d+)\.cross_attn\.norm_k_img\.(.*)$": r"blocks.\1.attn2.norm_added_k.\2",
            # FFN: original ffn.{0,2} -> sglang ffn.{fc_in,fc_out}
            r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
            r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
            # Modulation
            r"^blocks\.(\d+)\.modulation$": r"blocks.\1.scale_shift_table",
            # norm3 (cross-attn norm) -> self_attn_residual_norm.norm
            r"^blocks\.(\d+)\.norm3\.(.*)$": r"blocks.\1.self_attn_residual_norm.norm.\2",
            # FlashTalk audio: norm_x -> norm_audio
            r"^blocks\.(\d+)\.norm_x\.(.*)$": r"blocks.\1.norm_audio.\2",
            # audio_cross_attn.* passes through (same naming in checkpoint and sglang)
        }
    )


@dataclass
class FlashTalkWanVideoConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=FlashTalkWanVideoArchConfig)

    prefix: str = "Wan"
