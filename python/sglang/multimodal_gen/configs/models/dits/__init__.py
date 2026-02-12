# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from sglang.multimodal_gen.configs.models.dits.hunyuanvideo import HunyuanVideoConfig
from sglang.multimodal_gen.configs.models.dits.mova_audio import MOVAAudioConfig
from sglang.multimodal_gen.configs.models.dits.mova_video import MOVAVideoConfig
from sglang.multimodal_gen.configs.models.dits.wanvideo import (
    FlashTalkWanVideoConfig,
    WanVideoConfig,
)

__all__ = [
    "FlashTalkWanVideoConfig",
    "HunyuanVideoConfig",
    "MOVAAudioConfig",
    "MOVAVideoConfig",
    "WanVideoConfig",
]
