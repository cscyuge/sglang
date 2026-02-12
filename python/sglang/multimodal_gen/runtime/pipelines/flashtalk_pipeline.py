# SPDX-License-Identifier: Apache-2.0
"""
FlashTalk pipeline implementation.

Audio-driven talking face video generation pipeline based on Wan 14B I2V,
extended with Wav2Vec2 audio encoder, AudioProjModel, and audio cross-attention.
"""

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    ConditioningStage,
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
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class FlashTalkPipeline(LoRAPipeline, ComposedPipelineBase):
    """FlashTalk audio-driven talking face video generation pipeline.

    Pipeline stages:
    1. InputValidation - validate input parameters
    2. TextEncoding - T5 encode text prompt
    3. ImageEncoding - CLIP encode condition face image
    4. AudioEncoding - Wav2Vec2 + AudioProj encode audio
    5. Conditioning - combine text + image conditions
    6. TimestepPreparation - prepare timesteps
    7. LatentPreparation - initialize noise latents
    8. ImageVAEEncoding - VAE encode condition image to latent
    9. FlashTalkDenoising - no-CFG denoising with audio context
    10. Decoding - VAE decode to pixel space
    11. FlashTalkColorCorrection - Lab color correction
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

        # 5. Conditioning
        self.add_stage(
            stage_name="conditioning_stage",
            stage=ConditioningStage(),
        )

        # 6. Timestep preparation
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler"),
            ),
        )

        # 7. Latent preparation
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        # 8. Image VAE encoding
        self.add_stage(
            stage_name="image_latent_preparation_stage",
            stage=ImageVAEEncodingStage(vae=self.get_module("vae")),
        )

        # 9. FlashTalk denoising (no CFG, with audio context)
        self.add_stage(
            stage_name="denoising_stage",
            stage=FlashTalkDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # 10. Decoding
        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(vae=self.get_module("vae")),
        )

        # 11. Color correction
        self.add_stage(
            stage_name="color_correction_stage",
            stage=FlashTalkColorCorrectionStage(),
        )


EntryClass = FlashTalkPipeline
