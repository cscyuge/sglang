# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler,
)


def build_wan_s2v_scheduler(stream_r1_mode: bool, flow_shift: float | None):
    if stream_r1_mode:
        return SelfForcingFlowMatchScheduler(
            num_inference_steps=1000,
            shift=flow_shift if flow_shift is not None else 5.0,
            sigma_min=0.0,
            extra_one_step=True,
        )
    return FlowUniPCMultistepScheduler(shift=flow_shift)
