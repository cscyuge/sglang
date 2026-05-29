# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.workflow_runtime import (
    build_workflow_scheduler_override,
    workflow_denoising_plan_from_extra,
    workflow_sampler_plan_from_extra,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import (
    TimestepPreparationStage,
)


def _workflow_extra(experts=None, sampler=None):
    effective_parameters = {}
    if experts is not None:
        effective_parameters["experts"] = experts
    if sampler is not None:
        effective_parameters.update(sampler)
    return {
        "workflow": {
            "name": "test/workflow",
            "effective_parameters": effective_parameters,
        }
    }


def test_no_workflow_or_range_returns_none():
    assert workflow_denoising_plan_from_extra({}) is None
    assert (
        workflow_denoising_plan_from_extra(
            _workflow_extra(
                [
                    {"name": "high", "component": "transformer"},
                    {"name": "low", "component": "transformer_2"},
                ]
            )
        )
        is None
    )


def test_select_expert_by_step_range():
    plan = workflow_denoising_plan_from_extra(
        _workflow_extra(
            [
                {
                    "name": "high",
                    "component": "transformer",
                    "start_step": 0,
                    "end_step": 10,
                    "guidance_param": "guidance_scale",
                },
                {
                    "name": "low",
                    "component": "transformer_2",
                    "start_step": 10,
                    "end_step": None,
                    "guidance_param": "guidance_scale_2",
                },
            ]
        )
    )

    assert plan is not None
    assert plan.select_expert(0, 12).name == "high"
    assert plan.select_expert(9, 12).component == "transformer"
    assert plan.select_expert(10, 12).name == "low"
    assert plan.select_expert(11, 12).component == "transformer_2"
    assert plan.count_dual_transformer_steps(12) == (10, 2)


def test_expert_flow_shift_is_parsed():
    plan = workflow_denoising_plan_from_extra(
        _workflow_extra(
            [
                {
                    "name": "high",
                    "component": "transformer",
                    "start_step": 0,
                    "end_step": 2,
                    "flow_shift": 5,
                },
                {
                    "name": "low",
                    "component": "transformer_2",
                    "start_step": 2,
                    "flow_shift": 8,
                },
            ]
        )
    )

    assert plan is not None
    assert plan.has_expert_flow_shifts
    assert plan.experts[0].flow_shift == 5.0
    assert plan.experts[1].flow_shift == 8.0


def test_explicit_ranges_must_cover_each_step():
    plan = workflow_denoising_plan_from_extra(
        _workflow_extra(
            [
                {"name": "high", "component": "transformer", "end_step": 2},
                {
                    "name": "low",
                    "component": "transformer_2",
                    "start_step": 3,
                },
            ]
        )
    )

    assert plan is not None
    with pytest.raises(ValueError, match="no expert covering"):
        plan.select_expert(2, 4)


def test_mixed_implicit_and_explicit_ranges_are_rejected():
    with pytest.raises(ValueError, match="must all define"):
        workflow_denoising_plan_from_extra(
            _workflow_extra(
                [
                    {"name": "high", "component": "transformer", "end_step": 2},
                    {"name": "low", "component": "transformer_2"},
                ]
            )
        )


def test_unknown_component_is_rejected_for_dual_transformer_counts():
    plan = workflow_denoising_plan_from_extra(
        _workflow_extra(
            [
                {"name": "high", "component": "transformer", "end_step": 1},
                {"name": "other", "component": "custom_dit", "start_step": 1},
            ]
        )
    )

    assert plan is not None
    with pytest.raises(ValueError, match="only support transformer"):
        plan.count_dual_transformer_steps(2)


def test_sampler_plan_from_workflow_extra():
    plan = workflow_sampler_plan_from_extra(
        _workflow_extra(
            sampler={
                "sampler": "euler",
                "schedule": "simple",
                "flow_shift": 5,
                "boundary_ratio": 0.875,
            }
        )
    )

    assert plan is not None
    assert plan.workflow_name == "test/workflow"
    assert plan.normalized_sampler == "euler"
    assert plan.normalized_schedule == "simple"
    assert plan.flow_shift == 5.0
    assert plan.boundary_ratio == 0.875


def test_euler_simple_sampler_builds_request_local_scheduler():
    template = FlowUniPCMultistepScheduler(shift=12.0)
    sampler_plan = workflow_sampler_plan_from_extra(
        _workflow_extra(
            sampler={
                "sampler": "euler",
                "schedule": "simple",
                "flow_shift": 5.0,
            }
        )
    )

    scheduler = build_workflow_scheduler_override(template, sampler_plan)

    assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
    assert scheduler is not template
    assert scheduler.config.num_train_timesteps == template.config.num_train_timesteps
    assert scheduler.shift == 5.0


def test_timestep_preparation_builds_per_expert_flow_shift_schedulers(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages import base as stage_base

    monkeypatch.setattr(
        stage_base,
        "get_global_server_args",
        lambda: SimpleNamespace(comfyui_mode=False),
    )
    batch = Req(
        request_id="test",
        num_inference_steps=4,
        extra=_workflow_extra(
            experts=[
                {
                    "name": "high",
                    "component": "transformer",
                    "start_step": 0,
                    "end_step": 2,
                    "flow_shift": 5.0,
                },
                {
                    "name": "low",
                    "component": "transformer_2",
                    "start_step": 2,
                    "end_step": None,
                    "flow_shift": 8.0,
                },
            ],
            sampler={"sampler": "euler", "schedule": "simple", "flow_shift": 5.0},
        ),
    )
    stage = TimestepPreparationStage(FlowMatchEulerDiscreteScheduler(shift=1.0))
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            prepare_sigmas=lambda sigmas, _num_inference_steps: sigmas
        )
    )

    result = stage.forward(batch, server_args)

    assert result.workflow_schedulers is not None
    high = result.workflow_schedulers["high"]
    low = result.workflow_schedulers["low"]
    assert high.shift == 5.0
    assert low.shift == 8.0
    assert torch.equal(result.timesteps[:2], high.timesteps[:2])
    assert torch.equal(result.timesteps[2:], low.timesteps[2:4])


def test_flow_unipc_sampler_keeps_pipeline_scheduler():
    template = FlowUniPCMultistepScheduler(shift=12.0)
    sampler_plan = workflow_sampler_plan_from_extra(
        _workflow_extra(sampler={"sampler": "flow_unipc", "flow_shift": 12.0})
    )

    assert build_workflow_scheduler_override(template, sampler_plan) is None


def test_unsupported_sampler_schedule_is_rejected():
    template = FlowUniPCMultistepScheduler(shift=12.0)
    sampler_plan = workflow_sampler_plan_from_extra(
        _workflow_extra(sampler={"sampler": "euler", "schedule": "karras"})
    )

    with pytest.raises(ValueError, match="unsupported schedule"):
        build_workflow_scheduler_override(template, sampler_plan)
