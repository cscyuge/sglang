# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Timestep preparation stages for diffusion pipelines.

This module contains implementations of timestep preparation stages for diffusion pipelines.
"""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    clone_scheduler_runtime,
    get_or_create_request_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.pipelines_core.workflow_runtime import (
    WorkflowSamplerPlan,
    build_workflow_scheduler_override,
    workflow_denoising_plan_from_extra,
    workflow_sampler_plan_from_extra,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class TimestepPreparationFingerprint:
    num_inference_steps: int
    timesteps: Any
    sigmas: Any
    n_tokens: int | None
    height: int | None
    width: int | None
    num_frames: int | None
    workflow_sampler: WorkflowSamplerPlan | None


class TimestepPreparationStage(PipelineStage):
    """
    Stage for preparing timesteps for the diffusion process.

    This stage handles the preparation of the timestep sequence that will be used
    during the diffusion process.
    """

    deduplicated_tensor_tree_output_fields = ("timesteps", "sigmas")
    deduplicated_deepcopy_output_fields = ("scheduler", "workflow_schedulers")
    deduplicated_extra_tensor_tree_output_keys = ("mu",)

    def __init__(
        self,
        scheduler,
        prepare_extra_set_timesteps_kwargs: list[
            Callable[[Req, ServerArgs], Tuple[str, Any]]
        ] = [],
    ) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.prepare_extra_set_timesteps_kwargs = (
            prepare_extra_set_timesteps_kwargs or []
        )

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.REPLICATED

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Prepare timesteps for the diffusion process.



        Returns:
            The batch with prepared timesteps.
        """
        if batch.scheduler is not None and batch.timesteps is not None:
            return batch

        sampler_plan = workflow_sampler_plan_from_extra(batch.extra)
        workflow_scheduler = build_workflow_scheduler_override(
            self.scheduler, sampler_plan
        )
        if workflow_scheduler is not None:
            assert sampler_plan is not None
            logger.info(
                "Workflow %s using scheduler override: %s "
                "(sampler=%s, schedule=%s, flow_shift=%s)",
                sampler_plan.workflow_name,
                workflow_scheduler.__class__.__name__,
                sampler_plan.sampler,
                sampler_plan.schedule,
                sampler_plan.flow_shift,
            )
            batch.scheduler = workflow_scheduler

        scheduler = get_or_create_request_scheduler(batch, self.scheduler)
        device = get_local_torch_device()
        num_inference_steps = batch.num_inference_steps
        timesteps = batch.timesteps
        sigmas = batch.sigmas
        custom_timesteps_requested = timesteps is not None
        custom_sigmas_requested = sigmas is not None
        n_tokens = batch.n_tokens

        sigmas = server_args.pipeline_config.prepare_sigmas(sigmas, num_inference_steps)
        batch.sigmas = sigmas

        # Prepare extra kwargs for set_timesteps
        extra_set_timesteps_kwargs = {}
        if (
            n_tokens is not None
            and "n_tokens" in inspect.signature(scheduler.set_timesteps).parameters
        ):
            extra_set_timesteps_kwargs["n_tokens"] = n_tokens

        for callee in self.prepare_extra_set_timesteps_kwargs:
            key, value = callee(batch, server_args)
            assert isinstance(key, str)
            extra_set_timesteps_kwargs[key] = value
            if key == "mu":
                batch.extra["mu"] = value

        # Handle custom timesteps or sigmas
        if timesteps is not None and sigmas is not None:
            raise ValueError(
                "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
            )

        if timesteps is not None:
            accepts_timesteps = (
                "timesteps" in inspect.signature(scheduler.set_timesteps).parameters
            )
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(
                timesteps=timesteps, device=device, **extra_set_timesteps_kwargs
            )
            timesteps = scheduler.timesteps
        elif sigmas is not None:
            accept_sigmas = (
                "sigmas" in inspect.signature(scheduler.set_timesteps).parameters
            )
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(
                sigmas=sigmas, device=device, **extra_set_timesteps_kwargs
            )
            timesteps = scheduler.timesteps
        else:
            scheduler.set_timesteps(
                num_inference_steps, device=device, **extra_set_timesteps_kwargs
            )
            timesteps = scheduler.timesteps

        workflow_plan = workflow_denoising_plan_from_extra(batch.extra)
        if workflow_plan is not None and workflow_plan.has_expert_flow_shifts:
            if custom_timesteps_requested or custom_sigmas_requested:
                raise ValueError(
                    "Workflow per-expert flow_shift is incompatible with custom "
                    "timesteps or sigmas"
                )
            workflow_schedulers = self._prepare_workflow_expert_schedulers(
                scheduler=scheduler,
                sampler_plan=sampler_plan,
                workflow_plan=workflow_plan,
                num_inference_steps=num_inference_steps,
                device=device,
                extra_set_timesteps_kwargs=extra_set_timesteps_kwargs,
            )
            batch.workflow_schedulers = workflow_schedulers
            timesteps = self._compose_workflow_timesteps(
                workflow_plan=workflow_plan,
                workflow_schedulers=workflow_schedulers,
                num_inference_steps=num_inference_steps,
            )
            logger.info(
                "Workflow %s using per-expert flow shifts: %s",
                workflow_plan.workflow_name,
                {
                    expert.name: expert.flow_shift
                    for expert in workflow_plan.experts
                    if expert.flow_shift is not None
                },
            )

        # Update batch with prepared timesteps
        batch.timesteps = timesteps
        batch.scheduler = scheduler
        if not batch.is_warmup:
            self.log_debug("timesteps: %s", timesteps)
        return batch

    def _prepare_workflow_expert_schedulers(
        self,
        *,
        scheduler,
        sampler_plan: WorkflowSamplerPlan | None,
        workflow_plan,
        num_inference_steps: int,
        device: torch.device,
        extra_set_timesteps_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if sampler_plan is None or sampler_plan.normalized_sampler != "euler":
            raise ValueError(
                "Workflow per-expert flow_shift currently requires Euler sampler"
            )
        if sampler_plan.normalized_schedule not in {None, "simple"}:
            raise ValueError(
                "Workflow per-expert flow_shift currently requires simple schedule"
            )

        workflow_schedulers = {}
        for expert in workflow_plan.experts:
            expert_scheduler = clone_scheduler_runtime(scheduler)
            if expert.flow_shift is not None:
                if not hasattr(expert_scheduler, "set_shift"):
                    raise ValueError(
                        "Workflow per-expert flow_shift requires a scheduler with "
                        "set_shift()"
                    )
                expert_scheduler.set_shift(expert.flow_shift)
            expert_scheduler.set_timesteps(
                num_inference_steps, device=device, **extra_set_timesteps_kwargs
            )
            workflow_schedulers[expert.name] = expert_scheduler
        return workflow_schedulers

    def _compose_workflow_timesteps(
        self,
        *,
        workflow_plan,
        workflow_schedulers: dict[str, Any],
        num_inference_steps: int,
    ) -> torch.Tensor:
        step_timesteps = []
        for step_index in range(num_inference_steps):
            expert = workflow_plan.select_expert(step_index, num_inference_steps)
            scheduler = workflow_schedulers[expert.name]
            step_timesteps.append(scheduler.timesteps[step_index])
        return torch.stack(step_timesteps)

    def build_dedup_fingerprint(
        self, batch: Req, server_args: ServerArgs
    ) -> TimestepPreparationFingerprint:
        return TimestepPreparationFingerprint(
            num_inference_steps=batch.num_inference_steps,
            timesteps=self.freeze_for_dedup(batch.timesteps),
            sigmas=self.freeze_for_dedup(batch.sigmas),
            n_tokens=batch.n_tokens,
            height=batch.height,
            width=batch.width,
            num_frames=batch.num_frames,
            workflow_sampler=workflow_sampler_plan_from_extra(batch.extra),
        )

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify timestep preparation stage inputs."""
        result = VerificationResult()
        result.add_check(
            "num_inference_steps", batch.num_inference_steps, V.positive_int
        )
        result.add_check("timesteps", batch.timesteps, V.none_or_tensor)
        result.add_check("sigmas", batch.sigmas, V.none_or_list)
        result.add_check("n_tokens", batch.n_tokens, V.none_or_positive_int)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify timestep preparation stage outputs."""
        if (
            batch.is_warmup
            and isinstance(batch.timesteps, torch.Tensor)
            and torch.isnan(batch.timesteps).any()
        ):
            # diffusers flow-match scheduler can emit NaN for one-step warmup
            batch.timesteps = torch.ones(
                (1,), dtype=torch.float32, device=get_local_torch_device()
            )

        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.with_dims(1)])
        return result
