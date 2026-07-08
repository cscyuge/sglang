# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-S2V flat-checkpoint pipeline loader."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from transformers import AutoTokenizer

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
    get_param_names_mapping,
    set_default_torch_dtype,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.models.dits.wan_s2v import WanS2VTransformer3DModel
from sglang.multimodal_gen.runtime.models.encoders.wav2vec2 import Wav2Vec2AudioEncoder
from sglang.multimodal_gen.runtime.models.schedulers.wan_s2v_scheduler import (
    build_wan_s2v_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines.flashtalk_pipeline import (
    FlashTalkPipeline,
    _apply_fp8_quant_to_model,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DecodingStage,
    ImageVAEEncodingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_s2v import (
    WanS2VAudioEncodingStage,
    WanS2VDenoisingDispatchStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime.text_encoding import (
    RealtimeTextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.stream_r1_checkpoint import (
    load_stream_r1_generator_checkpoint,
)
from sglang.multimodal_gen.runtime.pipelines.wan_s2v_realtime import (
    WanS2VRealtimeSessionRunner,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class WanS2VPipeline(FlashTalkPipeline):
    """Loader for official WanModel_S2V checkpoints.

    The model-level forward is implemented in ``WanS2VTransformer3DModel``.
    This pipeline intentionally avoids FlashTalk's CLIP and AudioProj modules;
    S2V consumes Wav2Vec hidden states directly through its in-transformer
    causal audio encoder.
    """

    pipeline_name = "WanS2VPipeline"
    is_video_pipeline = True

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        flow_shift = server_args.pipeline_config.flow_shift
        self.modules["scheduler"] = build_wan_s2v_scheduler(
            bool(getattr(server_args.pipeline_config, "stream_r1_mode", False)),
            flow_shift,
        )

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        if loaded_modules is not None:
            return loaded_modules

        model_path = self.model_path
        if not os.path.isdir(model_path):
            raise ValueError(f"Wan S2V model path does not exist: {model_path}")
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {model_path}")

        logger.info("Wan S2V model path: %s", model_path)
        device = get_local_torch_device()
        loaded_components: dict[str, Any] = {}

        t5_path = os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth")
        vae_path = os.path.join(model_path, "Wan2.1_VAE.pth")
        io_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="s2v_pth")
        t5_future = (
            io_executor.submit(
                torch.load, t5_path, map_location="cpu", weights_only=True
            )
            if os.path.exists(t5_path)
            else None
        )
        vae_future = (
            io_executor.submit(
                torch.load, vae_path, map_location="cpu", weights_only=True
            )
            if os.path.exists(vae_path)
            else None
        )

        logger.info("Loading Wan S2V transformer...")
        loaded_components["transformer"] = self._load_transformer(
            model_path, server_args, device
        )
        logger.info("Loading Wan S2V VAE...")
        loaded_components["vae"] = self._load_vae(
            model_path, server_args, device, preloaded_state_dict=vae_future
        )
        logger.info("Loading Wan S2V T5 text encoder...")
        loaded_components["text_encoder"] = self._load_text_encoder(
            model_path, server_args, device, preloaded_state_dict=t5_future
        )
        io_executor.shutdown(wait=False)

        tokenizer_path = server_args.component_paths.get("tokenizer") or getattr(
            server_args.pipeline_config, "tokenizer_path", None
        )
        if not tokenizer_path:
            tokenizer_path = os.path.join(model_path, "google", "umt5-xxl")
        loaded_components["tokenizer"] = AutoTokenizer.from_pretrained(
            tokenizer_path if os.path.isdir(tokenizer_path) else model_path
        )

        audio_encoder_path = getattr(
            server_args.pipeline_config, "audio_encoder_path", None
        ) or server_args.component_paths.get("audio_encoder")
        if not audio_encoder_path:
            local_wav2vec = os.path.join(model_path, "wav2vec2-large-xlsr-53-english")
            audio_encoder_path = local_wav2vec if os.path.isdir(local_wav2vec) else None
        if audio_encoder_path:
            audio_encoder = Wav2Vec2AudioEncoder(
                model_path=audio_encoder_path,
                num_hidden_layers=25,
                include_embedding_layer=True,
            )
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
        else:
            logger.info(
                "No Wan S2V wav2vec path found; pass --audio-encoder-path for audio preprocessing."
            )

        logger.info("Wan S2V modules loaded: %s", list(loaded_components.keys()))
        return loaded_components

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )
        text_stage_cls = (
            RealtimeTextEncodingStage
            if bool(
                getattr(
                    getattr(server_args, "pipeline_config", None),
                    "wan_s2v_realtime",
                    False,
                )
            )
            else TextEncodingStage
        )
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=text_stage_cls(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )
        audio_encoder = self.get_module("audio_encoder")
        if audio_encoder is not None:
            self.add_stage(
                stage_name="audio_encoding_stage",
                stage=WanS2VAudioEncodingStage(
                    audio_encoder=audio_encoder,
                    wav2vec_feature_extractor=self.get_module(
                        "wav2vec_feature_extractor"
                    ),
                ),
            )
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")),
        )
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )
        self.add_stage(
            stage_name="image_latent_preparation_stage",
            stage=ImageVAEEncodingStage(vae=self.get_module("vae")),
        )
        self.add_stage(
            stage_name="denoising_stage",
            stage=WanS2VDenoisingDispatchStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        self.add_stage(
            stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae"))
        )

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        if batch.extra.get("wan_s2v_realtime_per_chunk", False):
            if not bool(getattr(server_args.pipeline_config, "wan_s2v_realtime", True)):
                raise RuntimeError(
                    "Wan S2V realtime chunks are disabled. Set "
                    "wan_s2v_realtime=true."
                )
            return WanS2VRealtimeSessionRunner(self).run_chunk(batch, server_args)
        if batch.extra.get("session_mode", False):
            if not bool(getattr(server_args.pipeline_config, "wan_s2v_realtime", True)):
                raise RuntimeError(
                    "Wan S2V realtime session is disabled. Use /v1/videos with "
                    "audio_path/audio_url or set wan_s2v_realtime=true."
                )
            return WanS2VRealtimeSessionRunner(self).run(batch, server_args)
        output = ComposedPipelineBase.forward(self, batch, server_args)
        if getattr(batch, "is_warmup", False):
            WanS2VRealtimeSessionRunner(self).prewarm_realtime_cuda_graphs(
                batch, server_args
            )
        return output

    def _load_transformer(
        self, model_path: str, server_args: ServerArgs, device: torch.device
    ) -> torch.nn.Module:
        from torch.distributed.fsdp import MixedPrecisionPolicy

        from sglang.multimodal_gen.runtime.loader.fsdp_load import (
            load_model_from_full_model_state_dict,
        )
        from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
        from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
            ModelOptFp4Config,
        )
        from sglang.multimodal_gen.utils import set_mixed_precision_policy

        with open(os.path.join(model_path, "config.json")) as f:
            raw_config = json.load(f)
        quant_config_dict = raw_config.get("quantization_config") or {}
        dit_quant_config = None
        if (
            quant_config_dict.get("quant_method") == "modelopt"
            and quant_config_dict.get("quant_algo") == "NVFP4"
        ):
            dit_quant_config = ModelOptFp4Config.from_config(quant_config_dict)

        dit_config = server_args.pipeline_config.dit_config
        arch = dit_config.arch_config
        field_map = {
            "in_dim": "in_channels",
            "out_dim": "out_channels",
            "dim": "hidden_size",
            "num_heads": "num_attention_heads",
            "num_layers": "num_layers",
            "ffn_dim": "ffn_dim",
            "text_dim": "text_dim",
            "freq_dim": "freq_dim",
            "cond_dim": "cond_dim",
            "audio_dim": "audio_dim",
            "num_audio_token": "num_audio_token",
            "enable_adain": "enable_adain",
            "adain_mode": "adain_mode",
            "audio_inject_layers": "audio_inject_layers",
            "zero_timestep": "zero_timestep",
            "add_last_motion": "add_last_motion",
            "enable_motioner": "enable_motioner",
            "enable_framepack": "enable_framepack",
            "framepack_drop_mode": "framepack_drop_mode",
            "motion_frames": "motion_frames",
        }
        for src, dst in field_map.items():
            if src in raw_config and dst != "hidden_size":
                setattr(arch, dst, raw_config[src])
        if "patch_size" in raw_config:
            arch.patch_size = tuple(raw_config["patch_size"])
        if "dim" in raw_config:
            arch.attention_head_dim = raw_config["dim"] // arch.num_attention_heads
        arch.__post_init__()

        safetensors_list = [
            f
            for f in _list_safetensors_files(model_path)
            if os.path.basename(f).startswith("diffusion_pytorch_model")
        ]
        if not safetensors_list:
            raise ValueError(f"No diffusion safetensors files found in {model_path}")

        default_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]
        mp_policy = MixedPrecisionPolicy(
            default_dtype, torch.float32, None, cast_forward_inputs=False
        )
        set_mixed_precision_policy(
            param_dtype=default_dtype,
            reduce_dtype=torch.float32,
            output_dtype=None,
            mp_policy=mp_policy,
        )

        with set_default_torch_dtype(default_dtype), torch.device("meta"):
            model = WanS2VTransformer3DModel(
                config=dit_config,
                quant_config=dit_quant_config,
            )

        if quant_config_dict and quant_config_dict.get("quant_method") == "fp8":
            fp8_config = Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                weight_block_size=quant_config_dict.get(
                    "weight_block_size", [128, 128]
                ),
            )
            _apply_fp8_quant_to_model(model, fp8_config)

        load_model_from_full_model_state_dict(
            model,
            safetensors_weights_iterator(safetensors_list),
            device,
            default_dtype,
            strict=False,
            cpu_offload=server_args.dit_cpu_offload,
            param_names_mapping=get_param_names_mapping(model.param_names_mapping),
        )

        stream_r1_checkpoint_path = getattr(
            server_args.pipeline_config, "stream_r1_generator_checkpoint_path", None
        )
        if stream_r1_checkpoint_path:
            checkpoint_info = load_stream_r1_generator_checkpoint(
                model,
                stream_r1_checkpoint_path,
                use_ema=bool(
                    getattr(server_args.pipeline_config, "use_stream_r1_ema", False)
                ),
                strict=False,
                param_names_mapping=get_param_names_mapping(model.param_names_mapping),
            )
            logger.info(
                "Loaded Stream-R1 generator checkpoint from %s "
                "(source=%s, tensors=%d, skipped=%d, missing=%d, unexpected=%d)",
                checkpoint_info.checkpoint_path,
                checkpoint_info.source_key or "<root>",
                checkpoint_info.num_tensors,
                len(checkpoint_info.skipped_keys),
                len(checkpoint_info.missing_keys),
                len(checkpoint_info.unexpected_keys),
            )
            if checkpoint_info.missing_keys or checkpoint_info.unexpected_keys:
                logger.warning(
                    "Stream-R1 checkpoint load used strict=False; "
                    "first missing keys=%s, first unexpected keys=%s",
                    list(checkpoint_info.missing_keys[:8]),
                    list(checkpoint_info.unexpected_keys[:8]),
                )

        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None and hasattr(
                quant_method, "process_weights_after_loading"
            ):
                quant_method.process_weights_after_loading(module)

        target = torch.device("cpu") if server_args.dit_cpu_offload else device
        d = model.hidden_size // model.num_attention_heads
        from sglang.multimodal_gen.runtime.models.dits.wan_s2v import rope_params

        model.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        ).to(target)
        if hasattr(model, "frame_packer"):
            model.frame_packer.freqs = model.freqs
            model.frame_packer.zip_frame_buckets = torch.tensor(
                [1, 2, 16], dtype=torch.long, device=target
            )

        for name, buf in list(model.named_buffers()):
            if not buf.is_meta:
                continue
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            parent.register_buffer(
                parts[-1],
                torch.zeros(buf.shape, dtype=buf.dtype, device=target),
                persistent=False,
            )

        for n, p in list(model.named_parameters()) + list(model.named_buffers()):
            if p.is_meta:
                raise RuntimeError(f"Unexpected Wan S2V tensor on meta device: {n}")
            if isinstance(p, torch.nn.Parameter):
                p.requires_grad = False
        logger.info(
            "Loaded Wan S2V transformer with %.2fB parameters",
            sum(p.numel() for p in model.parameters()) / 1e9,
        )
        return model


EntryClass = WanS2VPipeline
