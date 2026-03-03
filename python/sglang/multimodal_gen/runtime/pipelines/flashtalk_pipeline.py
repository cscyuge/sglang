# SPDX-License-Identifier: Apache-2.0
"""
FlashTalk pipeline implementation.

Audio-driven talking face video generation pipeline based on Wan 14B I2V,
extended with Wav2Vec2 audio encoder, AudioProjModel, and audio cross-attention.

Overrides load_modules to support FlashTalk's flat directory structure with
original (non-diffusers) naming conventions.
"""

import gc
import json
import os
import re
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from safetensors.torch import safe_open
from transformers import AutoTokenizer

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
    get_param_names_mapping,
    set_default_torch_dtype,
    skip_init_modules,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.models.adapter.audio_proj import AudioProjModel
from sglang.multimodal_gen.runtime.models.dits.flashtalk_wanvideo import (
    FlashTalkWanTransformer3DModel,
)
from sglang.multimodal_gen.runtime.models.encoders.wav2vec2 import (
    Wav2Vec2AudioEncoder,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.models.vaes.common import (
    DiagonalGaussianDistribution,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DecodingStage,
    ImageEncodingStage,
    ImageVAEEncodingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.audio_encoding import (
    AudioEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flashtalk import (
    FlashTalkColorCorrectionStage,
    FlashTalkDenoisingStage,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


def _chunk_frames_to_numpy(chunk_frames: torch.Tensor) -> np.ndarray:
    """Convert a GPU chunk tensor to a CPU numpy array for JPEG saving.

    This performs the GPU→CPU transfer on the calling thread so that the
    expensive JPEG encoding can be handed off to a background thread without
    any CUDA dependency.

    Args:
        chunk_frames: Tensor of shape (1, 3, T, H, W) in [0, 1] float range.

    Returns:
        numpy array of shape (T, H, W, 3) with dtype uint8.
    """
    return (
        (chunk_frames[0] * 255)
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()
    )


class VAECudaGraphRunner:
    """Captures and replays a CUDA graph for a fixed-shape VAE forward pass.

    Used to eliminate kernel launch overhead for per-chunk VAE decode/encode
    when input shapes are constant across all chunks.
    """

    def __init__(self, num_warmups: int = 2):
        self.graph = None
        self.num_warmups = num_warmups
        self.static_input = None
        self.static_output = None
        self._captured = False

    def capture(self, forward_fn, sample_input):
        """Warm up and capture a CUDA graph for ``forward_fn``."""
        self.static_input = sample_input.clone()

        def run_once():
            return forward_fn(self.static_input)

        # Warmup on a side stream (isolates warmup allocations)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.num_warmups):
                run_once()
        torch.cuda.current_stream().wait_stream(s)

        # Capture on the *current* (default) stream so that replay() —
        # which also runs on the default stream — has no cross-stream
        # synchronisation gap with the copy_() that precedes it.
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = run_once()
        self._captured = True

    def replay(self, input_tensor):
        """Copy *input_tensor* into the static buffer and replay the graph."""
        self.static_input.copy_(input_tensor)
        self.graph.replay()
        return self.static_output


def _save_chunk_frames_for_streaming(
    frames_np: np.ndarray,
    frame_dir: str,
    chunk_idx: int,
    frames_per_chunk: int,
) -> None:
    """Save per-chunk frames as individual JPEGs for real-time MJPEG streaming.

    This function is pure CPU (numpy + imageio) and safe to run in a
    background thread.

    Args:
        frames_np: numpy array of shape (T, H, W, 3) uint8.
        frame_dir: Directory to write frame_NNNNN.jpg files into.
        chunk_idx: Zero-based chunk index (used to compute global frame offset).
        frames_per_chunk: Number of frames per chunk (typically 28).
    """
    import imageio

    base_idx = chunk_idx * frames_per_chunk
    for i in range(frames_np.shape[0]):
        path = os.path.join(frame_dir, f"frame_{base_idx + i:05d}.jpg")
        imageio.imwrite(path, frames_np[i])


def _wait_for_session_audio_chunk(
    session_dir: str,
    chunk_idx: int,
    cancel_file: str | None = None,
    timeout: float = 300.0,
    poll_interval: float = 0.05,
) -> np.ndarray | None:
    """Wait for a session audio chunk file to appear.

    Returns the audio samples as a float32 numpy array, or None if the
    session ended (``end`` sentinel), was cancelled, or timed out.
    """
    chunk_path = os.path.join(session_dir, "audio_chunks", f"chunk_{chunk_idx:04d}.npy")
    end_path = os.path.join(session_dir, "end")

    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(end_path):
            return None
        if cancel_file and os.path.exists(cancel_file):
            return None
        if os.path.exists(chunk_path):
            try:
                return np.load(chunk_path)
            except Exception:
                # File may be mid-write; retry after a short wait
                time.sleep(0.01)
                try:
                    return np.load(chunk_path)
                except Exception:
                    return None
        time.sleep(poll_interval)
    return None


def _apply_fp8_quant_to_model(model: torch.nn.Module, fp8_config) -> int:
    """Patch block-level linear layers for FP8 quantization on meta device.

    Must be called after model creation on meta device, before weight loading.
    For each ColumnParallelLinear/RowParallelLinear inside transformer blocks:
      - Replaces weight parameter with float8_e4m3fn dtype
      - Adds weight_scale_inv parameter (float32)
      - Sets quant_method to Fp8LinearMethod

    Returns:
        Number of layers patched.
    """
    from sglang.multimodal_gen.runtime.layers.linear import LinearBase
    from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8LinearMethod

    block_size = fp8_config.weight_block_size
    fp8_method = Fp8LinearMethod(fp8_config)
    patched = 0

    for name, module in model.named_modules():
        if not isinstance(module, LinearBase):
            continue
        if not name.startswith("blocks."):
            continue

        old_weight = module.weight
        out_features, in_features = old_weight.shape

        if out_features % block_size[0] != 0 or in_features % block_size[1] != 0:
            logger.warning(
                "Skipping FP8 for %s: shape (%d, %d) not divisible by %s",
                name,
                out_features,
                in_features,
                block_size,
            )
            continue

        # Replace weight with fp8 dtype
        new_weight = torch.nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                dtype=torch.float8_e4m3fn,
                device=old_weight.device,
            ),
            requires_grad=False,
        )
        for attr in ("output_dim", "input_dim"):
            val = getattr(old_weight, attr, None)
            if val is not None:
                setattr(new_weight, attr, val)
        module.weight = new_weight

        # Add weight_scale_inv parameter
        scale_shape = (out_features // block_size[0], in_features // block_size[1])
        scale = torch.nn.Parameter(
            torch.empty(*scale_shape, dtype=torch.float32, device=old_weight.device),
            requires_grad=False,
        )
        for attr in ("output_dim", "input_dim"):
            val = getattr(old_weight, attr, None)
            if val is not None:
                setattr(scale, attr, val)
        module.register_parameter("weight_scale_inv", scale)

        module.quant_method = fp8_method
        patched += 1

    logger.info("Applied FP8 block quantization to %d linear layers", patched)
    return patched


class FlashTalkPipeline(LoRAPipeline, ComposedPipelineBase):
    """FlashTalk audio-driven talking face video generation pipeline.

    Pipeline stages:
    1. InputValidation - validate input parameters
    2. TextEncoding - T5 encode text prompt
    3. ImageEncoding - CLIP encode condition face image
    4. AudioEncoding - Wav2Vec2 + AudioProj encode audio
    5. TimestepPreparation - prepare timesteps
    6. LatentPreparation - initialize noise latents
    7. ImageVAEEncoding - VAE encode condition image to latent
    8. FlashTalkDenoising - no-CFG denoising with audio context
    9. Decoding - VAE decode to pixel space
    10. FlashTalkColorCorrection - Lab color correction
    """

    pipeline_name = "FlashTalkPipeline"
    is_video_pipeline = True

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
        "image_encoder",
        "image_processor",
        "audio_encoder",
        "audio_proj",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=server_args.pipeline_config.flow_shift
        )

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """Load modules from FlashTalk's flat directory structure.

        FlashTalk uses original (non-diffusers) naming:
        - config.json + diffusion_pytorch_model-*.safetensors -> Transformer + AudioProj
        - Wan2.1_VAE.pth -> VAE
        - models_t5_umt5-xxl-enc-bf16.pth -> T5 encoder
        - models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth -> CLIP
        - google/umt5-xxl/ -> T5 tokenizer
        - xlm-roberta-large/ -> CLIP tokenizer (not used as image_processor)
        """
        # If pre-loaded modules are provided, use them directly
        if loaded_modules is not None:
            return loaded_modules

        # FlashTalk uses a flat directory structure (no model_index.json),
        # so we skip maybe_download_model which requires diffusers layout.
        # Just validate the local path exists.
        model_path = self.model_path
        if not os.path.isdir(model_path):
            raise ValueError(
                f"FlashTalk model path does not exist: {model_path}. "
                "FlashTalk requires a local directory with flat checkpoint files."
            )
        # Verify expected files exist
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"config.json not found in {model_path}. "
                "Expected FlashTalk flat directory structure."
            )
        logger.info("FlashTalk model path: %s", model_path)

        pipeline_config = server_args.pipeline_config
        device = get_local_torch_device()

        loaded_components: dict[str, Any] = {}

        # ---- 4a. Transformer (sharded safetensors via FSDP loader) ----
        logger.info("Loading FlashTalk transformer...")
        loaded_components["transformer"] = self._load_transformer(
            model_path, server_args, device
        )

        # ---- 4b. AudioProj (from same safetensors, audio_proj.* keys) ----
        logger.info("Loading FlashTalk AudioProjModel...")
        loaded_components["audio_proj"] = self._load_audio_proj(
            model_path, server_args, device
        )

        # ---- 4c. VAE (Wan2.1_VAE.pth) ----
        logger.info("Loading FlashTalk VAE...")
        loaded_components["vae"] = self._load_vae(model_path, server_args, device)

        # ---- 4d. T5 Text Encoder ----
        logger.info("Loading FlashTalk T5 text encoder...")
        loaded_components["text_encoder"] = self._load_text_encoder(
            model_path, server_args, device
        )

        # ---- 4e. CLIP Image Encoder ----
        logger.info("Loading FlashTalk CLIP image encoder...")
        loaded_components["image_encoder"] = self._load_image_encoder(
            model_path, server_args, device
        )

        # ---- 4f. Tokenizer ----
        logger.info("Loading FlashTalk tokenizer...")
        tokenizer_path = os.path.join(model_path, "google", "umt5-xxl")
        if os.path.isdir(tokenizer_path):
            loaded_components["tokenizer"] = AutoTokenizer.from_pretrained(
                tokenizer_path
            )
        else:
            logger.warning(
                "T5 tokenizer directory not found at %s, trying model_path root",
                tokenizer_path,
            )
            loaded_components["tokenizer"] = AutoTokenizer.from_pretrained(model_path)

        # ---- 4f. Image processor ----
        # FlashTalk ships CLIP tokenizer in xlm-roberta-large/ but no image_processor.
        # Create a default processor for ViT-Huge-14 (224x224, standard normalization).
        loaded_components["image_processor"] = self._create_image_processor()

        # ---- 4g. Wav2Vec2 audio encoder (optional) ----
        # Check pipeline_config first, then fall back to component_paths
        # (--audio-encoder-path is parsed into component_paths["audio_encoder"])
        audio_encoder_path = getattr(pipeline_config, "audio_encoder_path", None)
        if not audio_encoder_path:
            audio_encoder_path = server_args.component_paths.get("audio_encoder")
        if audio_encoder_path:
            logger.info("Loading Wav2Vec2 audio encoder from %s", audio_encoder_path)
            audio_encoder = Wav2Vec2AudioEncoder(model_path=audio_encoder_path)
            audio_encoder = audio_encoder.to(device)
            audio_encoder.eval()
            loaded_components["audio_encoder"] = audio_encoder

            try:
                from transformers import Wav2Vec2FeatureExtractor

                loaded_components["wav2vec_feature_extractor"] = (
                    Wav2Vec2FeatureExtractor.from_pretrained(audio_encoder_path)
                )
            except Exception as e:
                logger.warning("Could not load Wav2Vec2FeatureExtractor: %s", e)
                loaded_components["wav2vec_feature_extractor"] = None
        else:
            logger.info(
                "No audio_encoder_path specified; audio encoder will not be loaded. "
                "Pass --audio-encoder-path to enable audio-driven generation."
            )
            loaded_components["audio_encoder"] = None
            loaded_components["wav2vec_feature_extractor"] = None

        # Remove optional None modules from required list so the check passes.
        # Work on a copy to avoid mutating the class-level list.
        required = list(self._required_config_modules)
        for name in ("audio_encoder", "image_processor"):
            if loaded_components.get(name) is None and name in required:
                required.remove(name)
        self._required_config_modules = required

        logger.info(
            "FlashTalk modules loaded: %s",
            [k for k, v in loaded_components.items() if v is not None],
        )
        return loaded_components

    def _load_transformer(
        self, model_path: str, server_args: ServerArgs, device: torch.device
    ) -> torch.nn.Module:
        """Load the DiT transformer via FSDP, filtering out audio_proj keys.

        Uses inline FSDP loading (instead of maybe_load_fsdp_model) to handle
        computed buffers (like RoPE inv_freq) that stay on meta device after
        checkpoint loading.
        """
        from torch.distributed.fsdp import MixedPrecisionPolicy

        from sglang.multimodal_gen.runtime.loader.fsdp_load import (
            load_model_from_full_model_state_dict,
        )
        from sglang.multimodal_gen.utils import set_mixed_precision_policy

        # Read config.json to update dit_config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            raw_config = json.load(f)

        dit_config = server_args.pipeline_config.dit_config
        # Update arch config from checkpoint's config.json
        if "in_dim" in raw_config:
            dit_config.arch_config.in_channels = raw_config["in_dim"]
        if "out_dim" in raw_config:
            dit_config.arch_config.out_channels = raw_config["out_dim"]
        if "dim" in raw_config:
            head_dim = dit_config.arch_config.attention_head_dim
            dit_config.arch_config.num_attention_heads = raw_config["dim"] // head_dim
        if "num_layers" in raw_config:
            dit_config.arch_config.num_layers = raw_config["num_layers"]
        if "ffn_dim" in raw_config:
            dit_config.arch_config.ffn_dim = raw_config["ffn_dim"]
        # Re-derive hidden_size and num_channels_latents
        dit_config.arch_config.__post_init__()

        safetensors_list = _list_safetensors_files(model_path)
        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {model_path}")

        default_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]
        param_dtype = default_dtype
        reduce_dtype = torch.float32

        logger.info(
            "Loading FlashTalkWanTransformer3DModel from %d safetensors files, dtype=%s",
            len(safetensors_list),
            default_dtype,
        )

        # Setup mixed precision policy
        mp_policy = MixedPrecisionPolicy(
            param_dtype, reduce_dtype, None, cast_forward_inputs=False
        )
        set_mixed_precision_policy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            output_dtype=None,
            mp_policy=mp_policy,
        )

        # Create model on meta device
        with set_default_torch_dtype(default_dtype), torch.device("meta"):
            model = FlashTalkWanTransformer3DModel(config=dit_config)

        # Apply FP8 block quantization if checkpoint has quantization_config
        quant_config_dict = raw_config.get("quantization_config")
        if quant_config_dict and quant_config_dict.get("quant_method") == "fp8":
            from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
                Fp8Config,
            )

            weight_block_size = quant_config_dict.get("weight_block_size", [128, 128])
            fp8_config = Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                weight_block_size=weight_block_size,
            )
            _apply_fp8_quant_to_model(model, fp8_config)

        # Load weights (also auto-calls process_weights_after_loading for FP8)
        weight_iterator = safetensors_weights_iterator(safetensors_list)
        param_names_mapping_fn = get_param_names_mapping(model.param_names_mapping)
        load_model_from_full_model_state_dict(
            model,
            weight_iterator,
            device,
            default_dtype,
            strict=False,
            cpu_offload=server_args.dit_cpu_offload,
            param_names_mapping=param_names_mapping_fn,
        )

        # Materialize any remaining meta-device buffers (e.g., RoPE inv_freq)
        # that are computed at init time and not stored in checkpoints.
        target = torch.device("cpu") if server_args.dit_cpu_offload else device
        for name, buf in list(model.named_buffers()):
            if buf.is_meta:
                logger.info("Materializing meta buffer: %s (shape=%s)", name, buf.shape)
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                attr_name = parts[-1]
                if attr_name == "inv_freq":
                    # RoPE inv_freq: 1 / (theta ** (arange(0, dim, 2) / dim))
                    dim = buf.shape[0] * 2
                    theta = 10000.0
                    inv_freq = 1.0 / (
                        theta ** (torch.arange(0, dim, 2, device=target).float() / dim)
                    )
                    parent.register_buffer(attr_name, inv_freq, persistent=False)
                else:
                    real_buf = torch.zeros(buf.shape, dtype=buf.dtype, device=target)
                    parent.register_buffer(attr_name, real_buf, persistent=False)

        # Verify no meta tensors remain
        from itertools import chain as iterchain

        for n, p in iterchain(model.named_parameters(), model.named_buffers()):
            if p.is_meta:
                raise RuntimeError(f"Unexpected param or buffer {n} on meta device.")
            if isinstance(p, torch.nn.Parameter):
                p.requires_grad = False

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded transformer with %.2fB parameters", total_params / 1e9)
        return model

    def _load_audio_proj(
        self, model_path: str, server_args: ServerArgs, device: torch.device
    ) -> torch.nn.Module:
        """Extract audio_proj weights from safetensors and load into AudioProjModel."""
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            raw_config = json.load(f)

        # Extract audio_proj.* keys from safetensors first to derive dimensions
        safetensors_list = _list_safetensors_files(model_path)
        audio_proj_state_dict = {}
        prefix = "audio_proj."
        for st_file in safetensors_list:
            with safe_open(st_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith(prefix):
                        stripped_key = key[len(prefix) :]
                        audio_proj_state_dict[stripped_key] = f.get_tensor(key)

        # Derive dimensions from config and checkpoint shapes
        num_audio_layers = raw_config.get("num_audio_layers", 12)
        audio_feat_dim = raw_config.get("audio_feat_dim", 768)
        audio_window_first = raw_config.get("audio_window", 5)

        # Infer audio_window_vf from proj1_vf weight shape if available
        audio_window_vf = raw_config.get("audio_window_vf", 12)
        if "proj1_vf.weight" in audio_proj_state_dict:
            vf_input_dim = audio_proj_state_dict["proj1_vf.weight"].shape[1]
            audio_window_vf = vf_input_dim // (num_audio_layers * audio_feat_dim)
            logger.info(
                "Inferred audio_window_vf=%d from proj1_vf weight shape %s",
                audio_window_vf,
                audio_proj_state_dict["proj1_vf.weight"].shape,
            )

        # Build AudioProjModel from config
        audio_proj = AudioProjModel(
            audio_window_first=audio_window_first,
            audio_window_vf=audio_window_vf,
            context_tokens=raw_config.get("context_tokens", 32),
            output_dim=raw_config.get("output_dim", 768),
            hidden_dim=raw_config.get("intermediate_dim", 512),
            num_audio_layers=num_audio_layers,
            audio_feat_dim=audio_feat_dim,
        )

        if audio_proj_state_dict:
            missing, unexpected = audio_proj.load_state_dict(
                audio_proj_state_dict, strict=False
            )
            if missing:
                logger.warning("AudioProj missing keys: %s", missing)
            if unexpected:
                logger.warning("AudioProj unexpected keys: %s", unexpected)
            logger.info(
                "Loaded AudioProjModel with %d parameters",
                sum(p.numel() for p in audio_proj.parameters()),
            )
        else:
            logger.warning(
                "No audio_proj.* keys found in safetensors; "
                "AudioProjModel will use random initialization."
            )

        audio_proj = audio_proj.to(device)
        audio_proj.eval()
        return audio_proj

    def _load_vae(
        self, model_path: str, server_args: ServerArgs, device: torch.device
    ) -> torch.nn.Module:
        """Load VAE from Wan2.1_VAE.pth."""
        vae_path = os.path.join(model_path, "Wan2.1_VAE.pth")
        if not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE checkpoint not found: {vae_path}")

        vae_config = server_args.pipeline_config.vae_config
        vae_precision = server_args.pipeline_config.vae_precision
        vae_dtype = PRECISION_TO_TYPE[vae_precision]

        # Resolve VAE class from registry
        vae_cls, _ = ModelRegistry.resolve_model_cls("AutoencoderKLWan")

        should_offload = server_args.vae_cpu_offload
        target_device = torch.device("cpu") if should_offload else device

        with set_default_torch_dtype(vae_dtype), skip_init_modules():
            vae = vae_cls(vae_config).to(target_device)

        state_dict = torch.load(vae_path, map_location="cpu", weights_only=True)
        mapped_sd = self._remap_wan_vae_keys(state_dict, vae_config)
        del state_dict
        missing, unexpected = vae.load_state_dict(mapped_sd, strict=False)
        if missing:
            logger.warning("VAE missing keys (%d): %s", len(missing), missing[:10])
        if unexpected:
            logger.warning(
                "VAE unexpected keys (%d): %s", len(unexpected), unexpected[:10]
            )
        del mapped_sd

        if not should_offload:
            vae = vae.to(device)
        vae.eval()
        logger.info(
            "Loaded VAE with %d parameters", sum(p.numel() for p in vae.parameters())
        )
        return vae

    def _load_text_encoder(
        self, model_path: str, server_args: ServerArgs, device: torch.device
    ) -> torch.nn.Module:
        """Load T5 encoder from models_t5_umt5-xxl-enc-bf16.pth with key remapping."""
        t5_path = os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth")
        if not os.path.exists(t5_path):
            raise FileNotFoundError(f"T5 checkpoint not found: {t5_path}")

        encoder_config = server_args.pipeline_config.text_encoder_configs[0]
        encoder_precision = server_args.pipeline_config.text_encoder_precisions[0]
        encoder_dtype = PRECISION_TO_TYPE[encoder_precision]

        # Check if we should use FSDP-based CPU offloading (matching standard loader)
        should_offload = server_args.text_encoder_cpu_offload
        arch_config = getattr(encoder_config, "arch_config", encoder_config)
        fsdp_conditions = getattr(arch_config, "_fsdp_shard_conditions", [])
        use_fsdp_offload = should_offload and len(fsdp_conditions) > 0

        # Resolve T5 encoder from registry (UMT5EncoderModel for umt5-xxl)
        model_cls, _ = ModelRegistry.resolve_model_cls("UMT5EncoderModel")

        with set_default_torch_dtype(encoder_dtype), skip_init_modules():
            model = model_cls(encoder_config)

        state_dict = torch.load(t5_path, map_location="cpu", weights_only=True)

        # Remap Wan's original T5 keys to sglang UMT5EncoderModel keys
        mapped_sd = self._remap_wan_t5_keys(state_dict)
        # Use load_weights which handles qkv fusing
        model.load_weights(mapped_sd.items())
        del state_dict, mapped_sd

        if use_fsdp_offload:
            # Use FSDP sharding with CPU offload (auto-moves params to GPU during forward)
            import torch.distributed as dist

            from sglang.multimodal_gen.runtime.loader.fsdp_load import shard_model

            mesh = dist.init_device_mesh(
                current_platform.device_type,
                mesh_shape=(1, dist.get_world_size()),
                mesh_dim_names=("offload", "replicate"),
            )
            shard_model(
                model,
                cpu_offload=True,
                reshard_after_forward=True,
                mesh=mesh["offload"],
                fsdp_shard_conditions=fsdp_conditions,
                pin_cpu_memory=server_args.pin_cpu_memory,
            )
        else:
            model = model.to(device)

        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded T5 encoder with %.2fB parameters", total_params / 1e9)
        return model

    @staticmethod
    def _remap_wan_t5_keys(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Remap Wan's original T5 naming to HuggingFace-style T5 keys.

        Wan naming -> HuggingFace T5 naming:
            token_embedding.weight -> shared.weight
            norm.weight -> encoder.final_layer_norm.weight
            blocks.N.norm1.weight -> encoder.block.N.layer.0.layer_norm.weight
            blocks.N.norm2.weight -> encoder.block.N.layer.1.layer_norm.weight
            blocks.N.attn.q.weight -> encoder.block.N.layer.0.SelfAttention.q.weight
            blocks.N.attn.k.weight -> encoder.block.N.layer.0.SelfAttention.k.weight
            blocks.N.attn.v.weight -> encoder.block.N.layer.0.SelfAttention.v.weight
            blocks.N.attn.o.weight -> encoder.block.N.layer.0.SelfAttention.o.weight
            blocks.N.pos_embedding.embedding.weight -> encoder.block.N.layer.0.SelfAttention.relative_attention_bias.weight
            blocks.N.ffn.gate.0.weight -> encoder.block.N.layer.1.DenseReluDense.wi_0.weight
            blocks.N.ffn.fc1.weight -> encoder.block.N.layer.1.DenseReluDense.wi_1.weight
            blocks.N.ffn.fc2.weight -> encoder.block.N.layer.1.DenseReluDense.wo.weight
        """

        mapped = {}
        for key, tensor in state_dict.items():
            new_key = None

            if key == "token_embedding.weight":
                new_key = "shared.weight"
            elif key == "norm.weight":
                new_key = "encoder.final_layer_norm.weight"
            else:
                m = re.match(r"blocks\.(\d+)\.(.+)", key)
                if m:
                    idx = m.group(1)
                    rest = m.group(2)
                    block_prefix = f"encoder.block.{idx}"

                    if rest == "norm1.weight":
                        new_key = f"{block_prefix}.layer.0.layer_norm.weight"
                    elif rest == "norm2.weight":
                        new_key = f"{block_prefix}.layer.1.layer_norm.weight"
                    elif rest == "attn.q.weight":
                        new_key = f"{block_prefix}.layer.0.SelfAttention.q.weight"
                    elif rest == "attn.k.weight":
                        new_key = f"{block_prefix}.layer.0.SelfAttention.k.weight"
                    elif rest == "attn.v.weight":
                        new_key = f"{block_prefix}.layer.0.SelfAttention.v.weight"
                    elif rest == "attn.o.weight":
                        new_key = f"{block_prefix}.layer.0.SelfAttention.o.weight"
                    elif rest == "pos_embedding.embedding.weight":
                        new_key = f"{block_prefix}.layer.0.SelfAttention.relative_attention_bias.weight"
                    elif rest == "ffn.gate.0.weight":
                        new_key = f"{block_prefix}.layer.1.DenseReluDense.wi_0.weight"
                    elif rest == "ffn.fc1.weight":
                        new_key = f"{block_prefix}.layer.1.DenseReluDense.wi_1.weight"
                    elif rest == "ffn.fc2.weight":
                        new_key = f"{block_prefix}.layer.1.DenseReluDense.wo.weight"

            if new_key is not None:
                mapped[new_key] = tensor

        return mapped

    @staticmethod
    def _remap_wan_vae_keys(
        state_dict: dict[str, torch.Tensor],
        vae_config,
    ) -> dict[str, torch.Tensor]:
        """Remap Wan original VAE keys to sglang AutoencoderKLWan keys.

        Original Wan VAE uses flat lists (downsamples/upsamples) with
        Sequential-style sub-keys (residual.0, residual.2, etc.), while sglang
        uses hierarchical blocks with named sub-modules (norm1, conv1, etc.).

        Encoder down_blocks are a flat ModuleList matching the checkpoint, so
        the index mapping is 1:1. Decoder up_blocks are hierarchical
        (WanUpBlock with resnets + upsamplers), requiring flat→hierarchical
        index conversion.
        """

        arch_config = getattr(vae_config, "arch_config", vae_config)
        dim_mult = list(arch_config.dim_mult)
        num_res_blocks = arch_config.num_res_blocks

        # Sub-key mapping within residual blocks:
        #   original Sequential indices → sglang named sub-modules
        _res_sub = {
            "residual.0.gamma": "norm1.gamma",
            "residual.2.weight": "conv1.weight",
            "residual.2.bias": "conv1.bias",
            "residual.3.gamma": "norm2.gamma",
            "residual.6.weight": "conv2.weight",
            "residual.6.bias": "conv2.bias",
            "shortcut.weight": "conv_shortcut.weight",
            "shortcut.bias": "conv_shortcut.bias",
        }

        # Build decoder flat index → (block_idx, type, local_idx)
        num_blocks = len(dim_mult)
        resnets_per_block = num_res_blocks + 1  # WanUpBlock adds 1 extra
        dec_flat_map: dict[int, tuple[int, str, int]] = {}
        flat_idx = 0
        for block_idx in range(num_blocks):
            for resnet_idx in range(resnets_per_block):
                dec_flat_map[flat_idx] = (block_idx, "resnet", resnet_idx)
                flat_idx += 1
            # All blocks except the last have an upsampler
            if block_idx != num_blocks - 1:
                dec_flat_map[flat_idx] = (block_idx, "upsampler", 0)
                flat_idx += 1

        mapped: dict[str, torch.Tensor] = {}

        for key, tensor in state_dict.items():
            new_key = None

            # --- Top-level quant convs ---
            if key.startswith("conv1."):
                new_key = "quant_conv." + key[len("conv1.") :]
            elif key.startswith("conv2."):
                new_key = "post_quant_conv." + key[len("conv2.") :]

            # --- Encoder ---
            elif key.startswith("encoder.conv1."):
                new_key = "encoder.conv_in." + key[len("encoder.conv1.") :]

            elif key.startswith("encoder.head."):
                rest = key[len("encoder.head.") :]
                if rest.startswith("0."):
                    # head.0 = RMS_norm → norm_out
                    new_key = "encoder.norm_out." + rest[2:]
                elif rest.startswith("2."):
                    # head.2 = CausalConv3d → conv_out
                    new_key = "encoder.conv_out." + rest[2:]

            elif key.startswith("encoder.middle."):
                rest = key[len("encoder.middle.") :]
                m = re.match(r"^(\d+)\.(.+)$", rest)
                if m:
                    mid_idx = int(m.group(1))
                    sub = m.group(2)
                    if mid_idx == 0:
                        # First resnet
                        mapped_sub = _res_sub.get(sub, sub)
                        new_key = f"encoder.mid_block.resnets.0.{mapped_sub}"
                    elif mid_idx == 1:
                        # Attention block (sub-keys match directly)
                        new_key = f"encoder.mid_block.attentions.0.{sub}"
                    elif mid_idx == 2:
                        # Second resnet
                        mapped_sub = _res_sub.get(sub, sub)
                        new_key = f"encoder.mid_block.resnets.1.{mapped_sub}"

            elif key.startswith("encoder.downsamples."):
                rest = key[len("encoder.downsamples.") :]
                m = re.match(r"^(\d+)\.(.+)$", rest)
                if m:
                    idx = int(m.group(1))
                    sub = m.group(2)
                    # Encoder down_blocks is a flat list with same indexing
                    mapped_sub = _res_sub.get(sub, sub)
                    new_key = f"encoder.down_blocks.{idx}.{mapped_sub}"

            # --- Decoder ---
            elif key.startswith("decoder.conv1."):
                new_key = "decoder.conv_in." + key[len("decoder.conv1.") :]

            elif key.startswith("decoder.head."):
                rest = key[len("decoder.head.") :]
                if rest.startswith("0."):
                    new_key = "decoder.norm_out." + rest[2:]
                elif rest.startswith("2."):
                    new_key = "decoder.conv_out." + rest[2:]

            elif key.startswith("decoder.middle."):
                rest = key[len("decoder.middle.") :]
                m = re.match(r"^(\d+)\.(.+)$", rest)
                if m:
                    mid_idx = int(m.group(1))
                    sub = m.group(2)
                    if mid_idx == 0:
                        mapped_sub = _res_sub.get(sub, sub)
                        new_key = f"decoder.mid_block.resnets.0.{mapped_sub}"
                    elif mid_idx == 1:
                        new_key = f"decoder.mid_block.attentions.0.{sub}"
                    elif mid_idx == 2:
                        mapped_sub = _res_sub.get(sub, sub)
                        new_key = f"decoder.mid_block.resnets.1.{mapped_sub}"

            elif key.startswith("decoder.upsamples."):
                rest = key[len("decoder.upsamples.") :]
                m = re.match(r"^(\d+)\.(.+)$", rest)
                if m:
                    idx = int(m.group(1))
                    sub = m.group(2)
                    if idx in dec_flat_map:
                        block_idx, typ, local_idx = dec_flat_map[idx]
                        if typ == "resnet":
                            mapped_sub = _res_sub.get(sub, sub)
                            new_key = (
                                f"decoder.up_blocks.{block_idx}"
                                f".resnets.{local_idx}.{mapped_sub}"
                            )
                        elif typ == "upsampler":
                            # Resample sub-keys (resample.1.*, time_conv.*)
                            # match directly between original and sglang
                            new_key = (
                                f"decoder.up_blocks.{block_idx}" f".upsamplers.0.{sub}"
                            )

            if new_key is not None:
                mapped[new_key] = tensor
            else:
                logger.warning("VAE key not mapped: %s", key)
                mapped[key] = tensor

        return mapped

    def _load_image_encoder(
        self, model_path: str, server_args: ServerArgs, device: torch.device
    ) -> torch.nn.Module:
        """Load CLIP image encoder from .pth file with OpenCLIP->sglang key mapping."""
        clip_pattern = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        clip_path = os.path.join(model_path, clip_pattern)
        if not os.path.exists(clip_path):
            logger.warning(
                "CLIP checkpoint not found at %s; image encoder will not be loaded.",
                clip_path,
            )
            return None

        encoder_config = server_args.pipeline_config.image_encoder_config
        encoder_precision = server_args.pipeline_config.image_encoder_precision
        encoder_dtype = PRECISION_TO_TYPE[encoder_precision]

        should_offload = server_args.image_encoder_cpu_offload
        target_device = torch.device("cpu") if should_offload else device

        model_cls, _ = ModelRegistry.resolve_model_cls("CLIPVisionModel")

        with set_default_torch_dtype(encoder_dtype), skip_init_modules():
            model = model_cls(encoder_config)

        state_dict = torch.load(clip_path, map_location="cpu", weights_only=True)

        # Remap OpenCLIP visual keys to sglang CLIPVisionModel keys
        mapped_sd = self._remap_openclip_visual_keys(state_dict, model)
        missing, unexpected = model.load_state_dict(mapped_sd, strict=False)
        if missing:
            logger.warning("CLIP missing keys (%d): %s", len(missing), missing[:10])
        if unexpected:
            logger.warning(
                "CLIP unexpected keys (%d): %s", len(unexpected), unexpected[:10]
            )
        del state_dict, mapped_sd

        model = model.to(target_device)
        model.eval()
        logger.info(
            "Loaded CLIP image encoder with %d parameters",
            sum(p.numel() for p in model.parameters()),
        )
        return model

    @staticmethod
    def _remap_openclip_visual_keys(
        state_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Remap OpenCLIP visual.* keys to sglang CLIPVisionModel keys.

        OpenCLIP naming:
            visual.cls_embedding -> vision_model.embeddings.class_embedding
            visual.patch_embedding.weight -> vision_model.embeddings.patch_embedding.weight
            visual.pos_embedding -> vision_model.embeddings.position_embedding.weight
            visual.pre_norm.* -> vision_model.pre_layrnorm.*
            visual.post_norm.* -> vision_model.post_layernorm.*
            visual.transformer.N.norm1.* -> vision_model.encoder.layers.N.layer_norm1.*
            visual.transformer.N.norm2.* -> vision_model.encoder.layers.N.layer_norm2.*
            visual.transformer.N.attn.to_qkv.* -> vision_model.encoder.layers.N.self_attn.qkv_proj.*
            visual.transformer.N.attn.proj.* -> vision_model.encoder.layers.N.self_attn.out_proj.*
            visual.transformer.N.mlp.0.* -> vision_model.encoder.layers.N.mlp.fc1.*
            visual.transformer.N.mlp.2.* -> vision_model.encoder.layers.N.mlp.fc2.*
            visual.head -> visual_projection.weight (if present)
        """

        mapped = {}
        prefix = "visual."

        for key, tensor in state_dict.items():
            if not key.startswith(prefix):
                continue
            vkey = key[len(prefix) :]

            new_key = None
            if vkey == "cls_embedding":
                # Shape (1,1,D) -> (D,) for class_embedding
                new_key = "vision_model.embeddings.class_embedding"
                tensor = tensor.squeeze()
            elif vkey == "patch_embedding.weight":
                new_key = "vision_model.embeddings.patch_embedding.weight"
            elif vkey == "pos_embedding":
                new_key = "vision_model.embeddings.position_embedding.weight"
                # Shape (1, S, D) -> (S, D)
                tensor = tensor.squeeze(0)
            elif vkey.startswith("pre_norm."):
                suffix = vkey[len("pre_norm.") :]
                new_key = f"vision_model.pre_layrnorm.{suffix}"
            elif vkey.startswith("post_norm."):
                suffix = vkey[len("post_norm.") :]
                new_key = f"vision_model.post_layernorm.{suffix}"
            else:
                # Transformer layers
                m = re.match(r"transformer\.(\d+)\.(.+)", vkey)
                if m:
                    layer_idx = m.group(1)
                    rest = m.group(2)
                    layer_prefix = f"vision_model.encoder.layers.{layer_idx}"

                    if rest.startswith("attn.to_qkv."):
                        suffix = rest[len("attn.to_qkv.") :]
                        new_key = f"{layer_prefix}.self_attn.qkv_proj.{suffix}"
                    elif rest.startswith("attn.proj."):
                        suffix = rest[len("attn.proj.") :]
                        new_key = f"{layer_prefix}.self_attn.out_proj.{suffix}"
                    elif rest.startswith("norm1."):
                        suffix = rest[len("norm1.") :]
                        new_key = f"{layer_prefix}.layer_norm1.{suffix}"
                    elif rest.startswith("norm2."):
                        suffix = rest[len("norm2.") :]
                        new_key = f"{layer_prefix}.layer_norm2.{suffix}"
                    elif rest.startswith("mlp.0."):
                        suffix = rest[len("mlp.0.") :]
                        new_key = f"{layer_prefix}.mlp.fc1.{suffix}"
                    elif rest.startswith("mlp.2."):
                        suffix = rest[len("mlp.2.") :]
                        new_key = f"{layer_prefix}.mlp.fc2.{suffix}"

            if new_key is not None:
                mapped[new_key] = tensor

        # Also check if visual.head should map to visual_projection
        if "visual.head" in state_dict:
            mapped["visual_projection.weight"] = state_dict["visual.head"]

        return mapped

    @staticmethod
    def _create_image_processor():
        """Create a default CLIPImageProcessor for ViT-Huge-14 (224x224)."""
        try:
            from transformers import CLIPImageProcessor

            return CLIPImageProcessor(
                size={"shortest_edge": 224},
                crop_size={"height": 224, "width": 224},
                do_resize=True,
                do_center_crop=True,
                do_normalize=True,
                image_mean=[0.48145466, 0.4578275, 0.40821073],
                image_std=[0.26862954, 0.26130258, 0.27577711],
            )
        except Exception as e:
            logger.warning("Could not create CLIPImageProcessor: %s", e)
            return None

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Set up pipeline stages with proper dependency injection."""

        # 1. Input validation
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage(),
        )

        # 2. Text encoding (T5)
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )

        # 3. Image encoding (CLIP)
        if (
            self.get_module("image_encoder") is not None
            and self.get_module("image_processor") is not None
        ):
            self.add_stage(
                stage_name="image_encoding_stage",
                stage=ImageEncodingStage(
                    image_encoder=self.get_module("image_encoder"),
                    image_processor=self.get_module("image_processor"),
                ),
            )

        # 4. Audio encoding (Wav2Vec2 + AudioProj)
        audio_encoder = self.get_module("audio_encoder")
        audio_proj = self.get_module("audio_proj")
        wav2vec_feature_extractor = self.get_module("wav2vec_feature_extractor")
        if audio_encoder is not None and audio_proj is not None:
            self.add_stage(
                stage_name="audio_encoding_stage",
                stage=AudioEncodingStage(
                    audio_encoder=audio_encoder,
                    audio_proj=audio_proj,
                    wav2vec_feature_extractor=wav2vec_feature_extractor,
                ),
            )

        # 5. Timestep preparation
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler"),
            ),
        )

        # 6. Latent preparation
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        # 7. Image VAE encoding
        self.add_stage(
            stage_name="image_latent_preparation_stage",
            stage=ImageVAEEncodingStage(vae=self.get_module("vae")),
        )

        # 8. FlashTalk denoising (no CFG, with audio context)
        self.add_stage(
            stage_name="denoising_stage",
            stage=FlashTalkDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # 9. Decoding
        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(vae=self.get_module("vae")),
        )

        # 10. Color correction
        self.add_stage(
            stage_name="color_correction_stage",
            stage=FlashTalkColorCorrectionStage(),
        )

    # ------------------------------------------------------------------
    # Multi-chunk generation
    # ------------------------------------------------------------------

    def _find_denoising_stage_index(self) -> int:
        """Return the index of the FlashTalkDenoisingStage in self.stages."""
        for i, stage in enumerate(self.stages):
            if isinstance(stage, FlashTalkDenoisingStage):
                return i
        raise RuntimeError("FlashTalkDenoisingStage not found in pipeline stages")

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        """Multi-chunk FlashTalk generation.

        Runs one-time stages (input validation through image VAE encoding),
        then loops over audio chunks: for each chunk, slices audio features,
        re-creates noise latents, denoises, decodes, applies color correction,
        and re-encodes the last ``motion_frames_num`` decoded frames as the
        motion latent for the next chunk.
        """

        if not batch.is_warmup and not batch.suppress_logs:
            logger.info(
                "Running FlashTalk pipeline stages: %s",
                list(self._stage_name_mapping.keys()),
                main_process_only=True,
            )

        denoising_idx = self._find_denoising_stage_index()

        # --- Run one-time stages (0 .. denoising_idx-1) ---
        for stage in self.stages[:denoising_idx]:
            batch = stage.forward(batch, server_args)

        # --- Check if multi-chunk is needed ---
        pipeline_config = server_args.pipeline_config

        vae_temporal_factor = (
            pipeline_config.vae_config.arch_config.scale_factor_temporal
            if hasattr(pipeline_config, "vae_config")
            else 4
        )

        # Use the training-expected chunk size (33 frames), NOT the SP-padded
        # batch.num_frames (which may be 61 for 8-GPU divisibility).
        # FlashTalk was trained on 33-frame chunks and the model architecture
        # (AudioProj, motion latent, denoising) all expect this frame count.
        chunk_frame_num = getattr(pipeline_config, "chunk_frame_num", 33)
        motion_frames_num = pipeline_config.motion_frames_num  # 5

        audio_features_all = batch.extra.get("audio_features_all")
        total_audio_video_frames = batch.extra.get("total_audio_video_frames", 0)

        # Session mode: open-ended live streaming (audio arrives incrementally)
        is_session = batch.extra.get("session_mode", False)
        session_dir = batch.extra.get("session_dir")

        # If no audio features or audio is short enough for a single chunk,
        # fall back to default single-chunk pipeline (unless session mode)
        if not is_session and (
            audio_features_all is None or total_audio_video_frames <= chunk_frame_num
        ):
            for stage in self.stages[denoising_idx:]:
                result = stage.forward(batch, server_args)
                if isinstance(result, OutputBatch):
                    return result
                batch = result
            return OutputBatch(output=batch.output, metrics=batch.metrics)

        # --- Multi-chunk mode: override SP-padded frame count ---
        frame_num = chunk_frame_num  # 33
        slice_len = frame_num - motion_frames_num  # 28

        # Override batch.num_frames so latent_prep_stage creates correctly-sized
        # noise latents (9 temporal steps instead of 16).
        batch.num_frames = frame_num

        # Slice image_latent to match chunk temporal dimensions.
        # SP padding produced extra zero temporal steps that we don't need.
        chunk_latent_num_frames = (frame_num - 1) // vae_temporal_factor + 1  # 9
        if (
            batch.image_latent is not None
            and batch.image_latent.shape[2] > chunk_latent_num_frames
        ):
            logger.info(
                "Slicing image_latent temporal dim from %d to %d for %d-frame chunks",
                batch.image_latent.shape[2],
                chunk_latent_num_frames,
                frame_num,
            )
            batch.image_latent = batch.image_latent[
                :, :, :chunk_latent_num_frames, :, :
            ]

        if is_session:
            num_chunks = 0  # open-ended; logged differently
            logger.info(
                "Session mode: open-ended generation (frame_num=%d, slice_len=%d, "
                "motion_frames=%d)",
                frame_num,
                slice_len,
                motion_frames_num,
            )
        else:
            # Match the original FlashTalk chunk count formula.
            num_chunks = (total_audio_video_frames - frame_num) // slice_len + 1

            # Debug: limit chunks for faster testing
            _debug_max_chunks = int(os.environ.get("FLASHTALK_MAX_CHUNKS", "0"))
            if _debug_max_chunks > 0:
                num_chunks = min(num_chunks, _debug_max_chunks)

            logger.info(
                "Multi-chunk generation: %d chunks (total_frames=%d, frame_num=%d, "
                "slice_len=%d, motion_frames=%d)",
                num_chunks,
                total_audio_video_frames,
                frame_num,
                slice_len,
                motion_frames_num,
            )

        # --- Stage references (look up by type to avoid fragile index assumptions) ---
        denoising_stage = self.stages[denoising_idx]
        color_correction_stage = None
        decoding_stage = None
        for stage in self.stages[denoising_idx + 1 :]:
            if isinstance(stage, FlashTalkColorCorrectionStage):
                color_correction_stage = stage
            elif isinstance(stage, DecodingStage):
                decoding_stage = stage
        if decoding_stage is None:
            raise RuntimeError("DecodingStage not found in pipeline stages")
        if color_correction_stage is None:
            raise RuntimeError(
                "FlashTalkColorCorrectionStage not found in pipeline stages"
            )

        # --- Audio proj + VAE references ---
        audio_proj = self.get_module("audio_proj")
        audio_encoder = self.get_module("audio_encoder")
        wav2vec_feature_extractor = self.get_module("wav2vec_feature_extractor")
        vae = self.get_module("vae")
        device = get_local_torch_device()

        # --- Per-chunk sliding window audio setup ---
        # Match original FlashTalk's streaming audio processing:
        # - 8-second sliding deque (silence-padded)
        # - Per-chunk loudness normalization + wav2vec2 processing
        # - Extract last frame_num frames from the wav2vec2 output
        raw_audio_array = batch.extra.get("raw_audio_array")
        sample_rate = batch.extra.get("audio_sample_rate", 16000)
        fps = batch.extra.get("audio_fps", 25)
        cached_audio_duration = getattr(pipeline_config, "cached_audio_duration", 8)

        # Session mode always uses streaming audio (chunks arrive via files)
        use_streaming_audio = (
            raw_audio_array is not None or is_session
        ) and audio_encoder is not None

        speech_slices = None  # only set for non-session streaming audio

        if use_streaming_audio:
            cached_audio_length = sample_rate * cached_audio_duration  # 128000
            audio_end_idx = cached_audio_duration * fps  # 200
            audio_start_idx = audio_end_idx - frame_num  # 167

            # Initialize sliding audio deque with silence (matching original)
            audio_dq: deque = deque(
                [0.0] * cached_audio_length, maxlen=cached_audio_length
            )

            if raw_audio_array is not None:
                # Normal multi-chunk: split raw audio into per-chunk sample slices
                slice_samples = slice_len * sample_rate // fps  # 17920
                total_samples = len(raw_audio_array)
                n_full_slices = total_samples // slice_samples
                if n_full_slices > 0:
                    speech_slices = raw_audio_array[
                        : n_full_slices * slice_samples
                    ].reshape(n_full_slices, slice_samples)
                else:
                    speech_slices = raw_audio_array[:slice_samples].reshape(1, -1)

                logger.info(
                    "Per-chunk streaming audio: cached_duration=%ds, "
                    "deque_len=%d, slice_samples=%d, num_slices=%d",
                    cached_audio_duration,
                    cached_audio_length,
                    slice_samples,
                    len(speech_slices),
                )
            else:
                # Session mode: audio arrives per-chunk via .npy files
                logger.info(
                    "Session streaming audio: cached_duration=%ds, deque_len=%d",
                    cached_audio_duration,
                    cached_audio_length,
                )

        # Pre-import pyloudnorm for per-chunk loudness normalization
        _pyln = None
        _pyln_meter = None
        if use_streaming_audio:
            try:
                import pyloudnorm as _pyln

                _pyln_meter = _pyln.Meter(sample_rate)
            except ImportError:
                pass

        # VAE normalization factors (same as ImageVAEEncodingStage uses)
        scaling_factor, shift_factor = pipeline_config.get_decode_scale_and_shift(
            device, torch.float32, vae
        )

        # --- Initial motion latent from image_latent ---
        # image_latent shape: (B, mask_ch + 16, N_t, H_lat, W_lat)
        # mask_ch = temporal_compression_ratio (typically 4)
        # Extract first temporal step of the VAE latent channels (skip mask)
        mask_channels = vae_temporal_factor  # temporal_compression_ratio
        initial_motion_latent = batch.image_latent[:, mask_channels:, :1, :, :]
        batch.extra["motion_latent"] = initial_motion_latent

        # Set up color reference from condition image for color correction
        if batch.condition_image is not None:
            import PIL.Image

            if isinstance(batch.condition_image, torch.Tensor):
                cond_img = batch.condition_image
                if cond_img.dim() == 4:
                    cond_img = cond_img.unsqueeze(2)  # (B,C,H,W) -> (B,C,1,H,W)
                # Normalize to [-1, 1] if needed
                if cond_img.min() >= 0 and cond_img.max() <= 1:
                    cond_img = cond_img * 2 - 1
                batch.extra["color_reference"] = cond_img.to(device)
            elif isinstance(batch.condition_image, PIL.Image.Image):
                arr = np.array(batch.condition_image).astype(np.float32) / 255.0
                cond_tensor = (
                    torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
                )  # (1, C, 1, H, W)
                cond_tensor = cond_tensor * 2 - 1  # [0,1] -> [-1,1]
                batch.extra["color_reference"] = cond_tensor.to(device)

        # --- Multi-chunk loop ---
        all_chunk_frames = []

        # When using multi-GPU SP, skip VAE and audio_encoder CPU offload
        # during the chunk loop. The original FlashTalk keeps everything on
        # GPU permanently. With 8x H20 (95 GB each) and VAE ~250 MB + audio
        # encoder ~190 MB, this is negligible memory but avoids ~100-200ms
        # of CPU<->GPU roundtrip per chunk.
        sp_size = get_sp_world_size()
        skip_vae_offload = sp_size > 1
        skip_audio_offload = sp_size > 1

        if skip_vae_offload:
            vae.to(device)

        # Compile VAE encode/decode if torch.compile is enabled AND warmup is
        # requested.  VAE compile saves ~0.2s per chunk but adds 40-50s to the
        # first chunk (compilation overhead), so it's only worth it for long
        # runs (100+ chunks) or when warmup pre-compiles everything.
        if server_args.enable_torch_compile and server_args.warmup:
            vae_compile_mode = "default"
            if not getattr(vae, "_flashtalk_vae_compiled", False):
                logger.info(
                    "Compiling VAE encode/decode with mode: %s", vae_compile_mode
                )
                vae.to(device)
                vae.encode = torch.compile(vae.encode, mode=vae_compile_mode)
                vae.decode = torch.compile(vae.decode, mode=vae_compile_mode)
                vae._flashtalk_vae_compiled = True

        # --- Wav2Vec2 CUDA Graph capture ---
        # The wav2vec2 encoder has fixed input shape (1, 128000) every chunk
        # (8s × 16kHz) and fixed output (1, 200, 12, 768). Its 12 F32
        # transformer layers are heavily CPU-bound in eager mode (~21ms wall
        # for ~5ms GPU work). CUDA graph eliminates all Python dispatch.
        _wav2vec_graph_runner = None
        if (
            use_streaming_audio
            and audio_encoder is not None
            and not getattr(audio_encoder, "_flashtalk_wav2vec_graphed", False)
        ):
            audio_encoder.to(device)
            audio_encoder.eval()
            _audio_end_idx = cached_audio_duration * fps  # 200
            _wav2vec_sample = torch.randn(
                1, sample_rate * cached_audio_duration,
                device=device, dtype=torch.float32,
            )

            def _wav2vec_forward(x):
                with torch.no_grad():
                    return audio_encoder(x, num_video_frames=_audio_end_idx)

            _wav2vec_graph_runner = VAECudaGraphRunner()
            _wav2vec_graph_runner.capture(_wav2vec_forward, _wav2vec_sample)
            del _wav2vec_sample
            audio_encoder._flashtalk_wav2vec_graphed = True
            logger.info("Wav2Vec2 CUDA graph captured")

        # --- VAE CUDA Graph capture (decode + encode) ---
        # Captures vae._decode and vae._encode as CUDA graphs to eliminate
        # kernel launch overhead across 33+ repeated chunks.  Uses the
        # single-pass (non-feature-cache) paths which are pure tensor ops.
        use_vae_cuda_graph = (
            os.environ.get("SGLANG_FLASHTALK_VAE_CUDA_GRAPH", "0") == "1"
        )
        _vae_decode_runner = None
        _vae_encode_runner = None
        _vae_decode_temporal_trim = 0  # leading frames to discard from _decode output

        if use_vae_cuda_graph:
            _vae_dtype = PRECISION_TO_TYPE[pipeline_config.vae_precision]
            _z_dim = pipeline_config.vae_config.arch_config.z_dim
            _vae_spatial_stride = 8
            _H_lat = batch.height // _vae_spatial_stride
            _W_lat = batch.width // _vae_spatial_stride

            vae.to(device=device, dtype=_vae_dtype)

            # Fuse WanRMS_norm modules: bake scale into gamma and
            # torch.compile the forward for Triton kernel fusion
            # (6 eager kernels → 2 compiled kernels per norm call).
            # Norms inside residual blocks are fused with SiLU to also
            # eliminate the separate activation kernel (3 → 2 kernels).
            from sglang.multimodal_gen.runtime.models.vaes.parallel.wan_common_utils import (
                WanRMS_norm,
            )
            from sglang.multimodal_gen.runtime.models.vaes.parallel.wan_dist_utils import (
                WanDistResample,
                WanDistResidualBlock,
            )
            from sglang.multimodal_gen.runtime.models.vaes.wanvae import (
                WanResample,
                WanResidualBlock,
            )

            # First pass: fuse norms in residual blocks with SiLU
            _silu_norms: set[int] = set()
            for module in vae.modules():
                if isinstance(module, (WanDistResidualBlock, WanResidualBlock)):
                    module.norm1.fuse_for_inference(fuse_silu=True)
                    module.norm2.fuse_for_inference(fuse_silu=True)
                    _silu_norms.add(id(module.norm1))
                    _silu_norms.add(id(module.norm2))
            # Second pass: fuse remaining norms (e.g. in attention blocks)
            for module in vae.modules():
                if isinstance(module, WanRMS_norm) and id(module) not in _silu_norms:
                    module.fuse_for_inference()
            del _silu_norms
            logger.info("WanRMS_norm modules fused and compiled for inference")

            # _decode() is single-pass — every latent frame (including the
            # first) goes through two temporal upsample stages (×2×2 = ×4),
            # producing 9×4 = 36 pixel frames.  But the first latent frame
            # should only map to 1 pixel frame (not 4): feature-cache
            # decode() handles this via first_chunk=True, while _decode()
            # cannot.  The 3 extra frames appear at the START as zero-
            # padding artifacts from the causal convolutions, so we trim
            # them to get the correct 33 frames.
            _vae_decode_temporal_trim = vae.temporal_compression_ratio - 1

            # Capture decode graph
            sample_dec = torch.randn(
                1,
                _z_dim,
                chunk_latent_num_frames,
                _H_lat,
                _W_lat,
                device=device,
                dtype=_vae_dtype,
            )
            _vae_decode_runner = VAECudaGraphRunner()
            _vae_decode_runner.capture(vae._decode, sample_dec)
            del sample_dec

            logger.info("VAE CUDA graph captured for decode")

            # Capture encode graph — _encode() is the single-pass path.
            # The encoder has two downsample3d layers with time_conv
            # (kernel=3, stride=2).  Without extra causal padding the
            # temporal dimension shrinks below kernel size for 5-frame
            # input.  Padding fix: set _padding[4]=2 (2 frames of left
            # causal padding) so T=5 → 3 → 2 through the two stages.
            #
            # NOTE: this permanently mutates time_conv._padding on the
            # model.  Do NOT call vae.encode() (feature-cache path) after
            # this — the extra padding would produce wrong temporal dims.
            # When use_vae_cuda_graph=True only _vae_encode_runner is used.
            for block in vae.encoder.down_blocks:
                if (
                    isinstance(block, (WanResample, WanDistResample))
                    and block.mode == "downsample3d"
                ):
                    _padding = list(block.time_conv._padding)
                    _padding[4] = 2
                    block.time_conv._padding = tuple(_padding)

            sample_enc = torch.randn(
                1,
                3,
                motion_frames_num,
                batch.height,
                batch.width,
                device=device,
                dtype=_vae_dtype,
            )
            _vae_encode_runner = VAECudaGraphRunner()
            _vae_encode_runner.capture(vae._encode, sample_enc)
            del sample_enc

            logger.info("VAE CUDA graph captured for encode")

        # Optional debug: save per-chunk outputs (set FLASHTALK_DEBUG_CHUNKS=1)
        _debug_save_chunks = os.environ.get("FLASHTALK_DEBUG_CHUNKS", "0") == "1"
        _debug_out_dir = os.environ.get(
            "FLASHTALK_DEBUG_DIR", "/sgl-workspace/outputs/debug_chunks"
        )

        # Disable Python's cyclic garbage collector during the chunk loop.
        # Reference-counting still frees objects immediately via `del`, but
        # the periodic GC sweep (which traverses all objects and can trigger
        # CUDA synchronisation points) is deferred until after the loop.
        _gc_was_enabled = gc.isenabled()
        gc.disable()

        # Background thread pool for streaming JPEG frame saving.
        # JPEG encoding + disk I/O is pure CPU work (~60-100ms per chunk)
        # that would otherwise block the main thread and leave the GPU idle.
        _frame_executor: ThreadPoolExecutor | None = None
        _frame_futures: list[Future] = []

        # File-based progress / cancellation IPC
        _request_id = batch.request_id
        _progress_dir = os.path.join(server_args.output_path, ".progress")
        os.makedirs(_progress_dir, exist_ok=True)
        _progress_file = (
            os.path.join(_progress_dir, _request_id) if _request_id else None
        )
        _cancel_file = (
            os.path.join(_progress_dir, f"{_request_id}.cancel")
            if _request_id
            else None
        )
        _cancelled = False

        # Streaming frame IPC: write per-chunk JPEG frames for MJPEG streaming
        _frame_dir = None
        _frames_per_chunk = slice_len  # typically 28
        if _request_id:
            _frame_dir = os.path.join(server_args.output_path, ".frames", _request_id)
            try:
                os.makedirs(_frame_dir, exist_ok=True)
                _meta = {
                    "num_chunks": num_chunks if not is_session else None,
                    "fps": batch.fps or 25,
                    "frames_per_chunk": _frames_per_chunk,
                    "width": batch.width,
                    "height": batch.height,
                    "session": is_session,
                }
                with open(os.path.join(_frame_dir, "meta.json"), "w") as _mf:
                    json.dump(_meta, _mf)
            except Exception as e:
                logger.warning("Failed to set up streaming frame dir: %s", e)
                _frame_dir = None

        if _frame_dir:
            _frame_executor = ThreadPoolExecutor(max_workers=1)

        # ================================================================
        # SESSION MODE: open-ended chunk loop with audio from files
        # ================================================================
        if is_session:
            all_chunk_frames = []
            chunk_idx = 0

            # Pre-compute loop-invariant values
            dit_dtype = PRECISION_TO_TYPE[pipeline_config.precision]
            gen = (
                batch.generator[0]
                if isinstance(batch.generator, list)
                else batch.generator
            )
            z_dim = pipeline_config.vae_config.arch_config.z_dim
            vae_spatial_stride = 8
            vae_dtype = PRECISION_TO_TYPE[pipeline_config.vae_precision]
            sf_dev = shift_factor.to(device=device)
            sc_dev = scaling_factor.to(device=device)

            while True:
                # Wait for audio chunk or end sentinel
                logger.info("Session: waiting for audio chunk %d...", chunk_idx)
                chunk_audio_data = _wait_for_session_audio_chunk(
                    session_dir,
                    chunk_idx,
                    cancel_file=_cancel_file,
                )
                if chunk_audio_data is None:
                    if _cancel_file and os.path.exists(_cancel_file):
                        logger.info("Session cancelled at chunk %d", chunk_idx)
                        _cancelled = True
                    else:
                        logger.info("Session ended at chunk %d", chunk_idx)
                    break

                chunk_start = time.time()
                logger.info("Session: generating chunk %d", chunk_idx)

                # a. Per-chunk audio processing (same as streaming mode)
                if use_streaming_audio:
                    audio_dq.extend(chunk_audio_data.tolist())
                    audio_array = np.array(audio_dq)

                    if _pyln is not None and _pyln_meter is not None:
                        loudness = _pyln_meter.integrated_loudness(audio_array)
                        if abs(loudness) <= 100:
                            audio_array = _pyln.normalize.loudness(
                                audio_array, loudness, -23.0
                            )

                    audio_feature_np = np.squeeze(
                        wav2vec_feature_extractor(
                            audio_array, sampling_rate=sample_rate
                        ).input_values
                    )
                    audio_feature_t = (
                        torch.from_numpy(audio_feature_np)
                        .float()
                        .to(device=device)
                        .unsqueeze(0)
                    )
                    if _wav2vec_graph_runner is not None:
                        chunk_wav2vec = _wav2vec_graph_runner.replay(audio_feature_t)
                    else:
                        audio_encoder.to(device)
                        chunk_wav2vec = audio_encoder(
                            audio_feature_t, num_video_frames=audio_end_idx
                        )

                    half_w = audio_proj.audio_window_first // 2
                    window_offsets = torch.arange(-half_w, half_w + 1, device=device)
                    frame_indices = torch.arange(
                        audio_start_idx, audio_end_idx, device=device
                    )
                    windowed_idx = frame_indices.unsqueeze(
                        1
                    ) + window_offsets.unsqueeze(0)
                    windowed_idx = windowed_idx.clamp(0, audio_end_idx - 1)
                    windowed_features = chunk_wav2vec[:, windowed_idx]

                    audio_proj.to(device)
                    batch.extra["audio_context"] = audio_proj.forward_prewindowed(
                        windowed_features, vae_temporal_factor=vae_temporal_factor
                    )

                    if (
                        server_args.audio_encoder_cpu_offload
                        and not skip_audio_offload
                        and _wav2vec_graph_runner is None
                    ):
                        audio_encoder.to("cpu", non_blocking=True)

                # b. Fresh noise latents
                latent_shape = (
                    1,
                    z_dim,
                    chunk_latent_num_frames,
                    batch.height // vae_spatial_stride,
                    batch.width // vae_spatial_stride,
                )
                batch.latents = torch.randn(
                    latent_shape, dtype=dit_dtype, device=device, generator=gen
                )

                # c. Denoise
                batch.timesteps = None
                batch = denoising_stage.forward(batch, server_args)

                # d. VAE decode + color correct + motion carry
                if not use_vae_cuda_graph:
                    vae.to(device=device, dtype=vae_dtype)

                denoised_latents = batch.latents.to(device=device, dtype=torch.float32)
                denorm_latents = denoised_latents / sc_dev + sf_dev

                denorm_latents = denorm_latents.to(vae_dtype)
                if use_vae_cuda_graph:
                    videos = _vae_decode_runner.replay(denorm_latents)
                    videos = videos[:, :, _vae_decode_temporal_trim:].clone()
                else:
                    videos = vae.decode(denorm_latents)
                del denorm_latents

                batch.output = videos
                batch = color_correction_stage.forward(batch, server_args)
                videos_corrected = batch.output
                del videos

                cond_frame = videos_corrected[:, :, -motion_frames_num:].clone()
                cond_frame = cond_frame.to(vae_dtype)
                if _vae_encode_runner is not None:
                    enc_raw = _vae_encode_runner.replay(cond_frame)
                    motion_latent_raw = enc_raw[:, :z_dim].clone()
                else:
                    latent_dist = vae.encode(cond_frame)
                    if isinstance(latent_dist, DiagonalGaussianDistribution):
                        motion_latent_raw = latent_dist.mode()
                    elif hasattr(latent_dist, "latent_dist"):
                        motion_latent_raw = latent_dist.latent_dist.mode()
                    else:
                        motion_latent_raw = latent_dist
                del cond_frame

                corrected = (videos_corrected + 1) / 2
                corrected = corrected.clamp(0, 1)
                del videos_corrected

                motion_latent_f32 = motion_latent_raw.float()
                motion_latent_normalized = (motion_latent_f32 - sf_dev) * sc_dev
                batch.extra["motion_latent"] = motion_latent_normalized

                if (
                    server_args.vae_cpu_offload
                    and not skip_vae_offload
                    and not use_vae_cuda_graph
                ):
                    vae.to("cpu", non_blocking=True)

                # e. Collect frames
                chunk_frames = corrected[:, :, motion_frames_num:].clone()
                del corrected
                # Session mode: frames are already streamed via fMP4,
                # skip accumulating on GPU to avoid OOM on long sessions.
                if not is_session:
                    all_chunk_frames.append(chunk_frames)

                # f. Progress + streaming frames
                if _progress_file:
                    try:
                        with open(_progress_file, "w") as _pf:
                            _pf.write(f"{chunk_idx + 1} -1")
                    except Exception:
                        pass
                if _frame_dir and _frame_executor is not None:
                    try:
                        frames_np = _chunk_frames_to_numpy(chunk_frames)
                        _frame_futures.append(
                            _frame_executor.submit(
                                _save_chunk_frames_for_streaming,
                                frames_np,
                                _frame_dir,
                                chunk_idx,
                                _frames_per_chunk,
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            "Session frame save failed for chunk %d: %s", chunk_idx, e
                        )
                del chunk_frames

                _t_chunk = time.time() - chunk_start
                logger.info("Session chunk %d: %.3fs", chunk_idx, _t_chunk)
                chunk_idx += 1

                # Periodically drain completed futures to avoid unbounded list growth
                if chunk_idx % 50 == 0 and _frame_futures:
                    _frame_futures = [f for f in _frame_futures if not f.done()]

            # --- Session post-loop ---
            if _gc_was_enabled:
                gc.enable()
                gc.collect()

            # Wait for all background frame saves to finish
            for fut in _frame_futures:
                try:
                    fut.result()
                except Exception as e:
                    logger.warning("Background frame save error: %s", e)
            _frame_futures.clear()
            if _frame_executor is not None:
                _frame_executor.shutdown(wait=False)

            if _frame_dir:
                try:
                    with open(os.path.join(_frame_dir, "done"), "w") as _df:
                        pass
                except Exception:
                    pass

            if not all_chunk_frames:
                # Session mode: frames already streamed via fMP4, nothing to save.
                # Set output_file_paths to signal completion without triggering save_outputs.
                return OutputBatch(
                    output=None,
                    output_file_paths=[],
                    metrics=batch.metrics,
                )

            final_video = torch.cat(all_chunk_frames, dim=2)
            audio_path = batch.extra.get("audio_path")
            return OutputBatch(
                output=final_video,
                metrics=batch.metrics,
                audio_path=audio_path,
            )

        # ================================================================
        # NORMAL MULTI-CHUNK MODE (existing code)
        # ================================================================
        for chunk_idx in range(num_chunks):
            # Check for cancellation at the start of each chunk
            if _cancel_file and os.path.exists(_cancel_file):
                logger.info("Cancelled at chunk %d/%d", chunk_idx + 1, num_chunks)
                _cancelled = True
                break

            chunk_start = time.time()
            logger.info("Generating chunk %d/%d", chunk_idx + 1, num_chunks)

            # a. Per-chunk audio: sliding window wav2vec2 processing
            #    (matches original FlashTalk's "stream" mode exactly)
            if use_streaming_audio:
                # Add this chunk's audio samples to the sliding deque
                if chunk_idx < len(speech_slices):
                    audio_dq.extend(speech_slices[chunk_idx].tolist())
                audio_array = np.array(audio_dq)

                # Per-chunk loudness normalization (matching original)
                if _pyln is not None and _pyln_meter is not None:
                    loudness = _pyln_meter.integrated_loudness(audio_array)
                    if abs(loudness) <= 100:
                        audio_array = _pyln.normalize.loudness(
                            audio_array, loudness, -23.0
                        )

                # Process through wav2vec2 feature extractor + encoder
                audio_feature_np = np.squeeze(
                    wav2vec_feature_extractor(
                        audio_array, sampling_rate=sample_rate
                    ).input_values
                )
                audio_feature_t = (
                    torch.from_numpy(audio_feature_np)
                    .float()
                    .to(device=device)
                    .unsqueeze(0)
                )
                if _wav2vec_graph_runner is not None:
                    chunk_wav2vec = _wav2vec_graph_runner.replay(audio_feature_t)
                else:
                    audio_encoder.to(device)
                    chunk_wav2vec = audio_encoder(
                        audio_feature_t, num_video_frames=audio_end_idx
                    )
                # chunk_wav2vec: (1, audio_end_idx, num_layers, feat_dim)
                #              = (1, 200, 12, 768)

                # Window on the FULL wav2vec output (matching original's
                # get_audio_embedding), then slice the last frame_num frames.
                half_w = audio_proj.audio_window_first // 2  # 2
                window_offsets = torch.arange(-half_w, half_w + 1, device=device)
                frame_indices = torch.arange(
                    audio_start_idx, audio_end_idx, device=device
                )
                windowed_idx = frame_indices.unsqueeze(1) + window_offsets.unsqueeze(0)
                windowed_idx = windowed_idx.clamp(0, audio_end_idx - 1)
                # (frame_num, window_size) indices into (1, 200, ...)
                windowed_features = chunk_wav2vec[:, windowed_idx]
                # (1, frame_num, window_size, num_layers, feat_dim)

                # Project through AudioProjModel with pre-windowed features
                audio_proj.to(device)
                batch.extra["audio_context"] = audio_proj.forward_prewindowed(
                    windowed_features, vae_temporal_factor=vae_temporal_factor
                )

                if (
                    server_args.audio_encoder_cpu_offload
                    and not skip_audio_offload
                    and _wav2vec_graph_runner is None
                ):
                    audio_encoder.to("cpu", non_blocking=True)
            else:
                # Fallback: slice pre-computed features (legacy path)
                start = chunk_idx * slice_len
                chunk_audio = audio_features_all[:, start : start + frame_num]
                if chunk_audio.shape[1] < frame_num:
                    pad_len = frame_num - chunk_audio.shape[1]
                    chunk_audio = torch.nn.functional.pad(
                        chunk_audio, (0, 0, 0, 0, 0, pad_len), mode="replicate"
                    )
                chunk_audio = chunk_audio.to(device)
                audio_proj.to(device)
                batch.extra["audio_context"] = audio_proj(
                    chunk_audio, vae_temporal_factor=vae_temporal_factor
                )

            # b. Create fresh noise latents (matching original FlashTalk dtype/shape)
            dit_dtype = PRECISION_TO_TYPE[pipeline_config.precision]
            gen = (
                batch.generator[0]
                if isinstance(batch.generator, list)
                else batch.generator
            )
            z_dim = pipeline_config.vae_config.arch_config.z_dim
            vae_spatial_stride = 8  # WanVAE spatial stride
            latent_shape = (
                1,
                z_dim,
                chunk_latent_num_frames,
                batch.height // vae_spatial_stride,
                batch.width // vae_spatial_stride,
            )
            batch.latents = torch.randn(
                latent_shape, dtype=dit_dtype, device=device, generator=gen
            )

            # c. Denoise — force FlashTalk-specific timesteps (with shift)
            batch.timesteps = None
            batch = denoising_stage.forward(batch, server_args)

            # d+e+f. Decode → color correct → motion carry
            # Match the original FlashTalk flow:
            # 1. Denormalize latents  2. VAE decode → [-1, 1]
            # 3. Color correct in [-1, 1]  4. Motion carry → VAE encode → normalize
            vae_dtype = PRECISION_TO_TYPE[pipeline_config.vae_precision]
            if not use_vae_cuda_graph:
                vae.to(device=device, dtype=vae_dtype)

            # Denormalize latents (like DecodingStage.scale_and_shift)
            denoised_latents = batch.latents.to(device=device, dtype=torch.float32)
            sf_dev = shift_factor.to(device=device)
            sc_dev = scaling_factor.to(device=device)
            denorm_latents = denoised_latents / sc_dev + sf_dev

            # VAE decode
            denorm_latents = denorm_latents.to(vae_dtype)
            if use_vae_cuda_graph:
                videos = _vae_decode_runner.replay(denorm_latents)
                videos = videos[:, :, _vae_decode_temporal_trim:].clone()
            else:
                videos = vae.decode(denorm_latents)  # [-1, 1]
            del denorm_latents

            # Color correction (operates on [-1, 1])
            batch.output = videos
            batch = color_correction_stage.forward(batch, server_args)
            videos_corrected = batch.output  # [-1, 1]
            del videos

            # Motion carry: last 5 frames in [-1, 1] → VAE encode
            # .clone() ensures contiguous memory for VAE encode and
            # releases the reference to the full decoded video tensor.
            cond_frame = videos_corrected[:, :, -motion_frames_num:].clone()
            cond_frame = cond_frame.to(vae_dtype)
            if _vae_encode_runner is not None:
                enc_raw = _vae_encode_runner.replay(cond_frame)
                motion_latent_raw = enc_raw[:, :z_dim].clone()
            else:
                latent_dist = vae.encode(cond_frame)
                if isinstance(latent_dist, DiagonalGaussianDistribution):
                    motion_latent_raw = latent_dist.mode()
                elif hasattr(latent_dist, "latent_dist"):
                    motion_latent_raw = latent_dist.latent_dist.mode()
                else:
                    motion_latent_raw = latent_dist
            del cond_frame

            # Convert to [0, 1] for chunk frame collection
            corrected = (videos_corrected + 1) / 2  # [0, 1]
            corrected = corrected.clamp(0, 1)
            del videos_corrected

            # Normalize the motion latent in float32
            motion_latent_f32 = motion_latent_raw.float()
            motion_latent_normalized = (motion_latent_f32 - sf_dev) * sc_dev
            batch.extra["motion_latent"] = motion_latent_normalized

            # Offload VAE if configured (skip when keeping on GPU for SP)
            if (
                server_args.vae_cpu_offload
                and not skip_vae_offload
                and not use_vae_cuda_graph
            ):
                vae.to("cpu", non_blocking=True)

            # g. Collect non-overlapping frames (skip first motion_frames_num)
            # .clone() releases the full corrected tensor (33 frames) and
            # keeps only the 28 new frames, reducing memory pressure.
            chunk_frames = corrected[:, :, motion_frames_num:].clone()
            del corrected
            all_chunk_frames.append(chunk_frames)

            # Write progress file for HTTP polling
            if _progress_file:
                try:
                    with open(_progress_file, "w") as _pf:
                        _pf.write(f"{chunk_idx + 1} {num_chunks}")
                except Exception:
                    pass

            # Save per-chunk JPEG frames for real-time MJPEG streaming
            # D2H transfer happens here on the main thread (~1-2ms), then JPEG
            # encoding + disk I/O (~60-100ms) runs in a background thread so
            # the GPU can start the next chunk's audio/denoising immediately.
            if _frame_dir and _frame_executor is not None:
                try:
                    frames_np = _chunk_frames_to_numpy(chunk_frames)
                    _frame_futures.append(
                        _frame_executor.submit(
                            _save_chunk_frames_for_streaming,
                            frames_np,
                            _frame_dir,
                            chunk_idx,
                            _frames_per_chunk,
                        )
                    )
                except Exception as e:
                    logger.warning(
                        "Streaming frame save failed for chunk %d: %s", chunk_idx, e
                    )

            # Per-chunk timing summary (skip chunk 1 which includes compile)
            _t_chunk = time.time() - chunk_start
            if chunk_idx > 0:
                logger.info(
                    "Chunk %d/%d: total=%.3fs",
                    chunk_idx + 1,
                    num_chunks,
                    _t_chunk,
                )

            # Optional: save per-chunk video for debugging
            if _debug_save_chunks and chunk_idx < 10:
                try:
                    import imageio

                    os.makedirs(_debug_out_dir, exist_ok=True)
                    _dbg = (chunk_frames[0] * 255).clamp(0, 255).to(torch.uint8)
                    _dbg = _dbg.permute(1, 2, 3, 0).cpu().numpy()
                    _dbg_path = os.path.join(
                        _debug_out_dir, f"chunk_{chunk_idx:03d}.mp4"
                    )
                    imageio.mimsave(_dbg_path, list(_dbg), fps=25)
                except Exception as e:
                    logger.warning("Debug chunk save failed: %s", e)

        # --- Concatenate all chunks & return ---
        # Re-enable garbage collection (was disabled during chunk loop)
        if _gc_was_enabled:
            gc.enable()
            gc.collect()

        # Wait for all background frame saves to finish before writing "done"
        for fut in _frame_futures:
            try:
                fut.result()
            except Exception as e:
                logger.warning("Background frame save error: %s", e)
        _frame_futures.clear()
        if _frame_executor is not None:
            _frame_executor.shutdown(wait=False)

        # Write done sentinel for streaming clients
        if _frame_dir:
            try:
                with open(os.path.join(_frame_dir, "done"), "w") as _df:
                    pass
            except Exception:
                pass

        if _cancelled and not all_chunk_frames:
            logger.info("Cancelled before any chunk completed — returning empty output")
            return OutputBatch(
                output=None,
                metrics=batch.metrics,
            )

        if _cancelled:
            logger.info(
                "Returning partial result: %d/%d chunks",
                len(all_chunk_frames),
                num_chunks,
            )

        final_video = torch.cat(all_chunk_frames, dim=2)

        audio_path = batch.extra.get("audio_path")

        return OutputBatch(
            output=final_video,
            metrics=batch.metrics,
            audio_path=audio_path,
        )


EntryClass = FlashTalkPipeline
