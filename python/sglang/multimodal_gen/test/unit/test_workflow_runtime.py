# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang.multimodal_gen.runtime.pipelines_core.workflow_runtime import (
    workflow_denoising_plan_from_extra,
)


def _workflow_extra(experts):
    return {
        "workflow": {
            "name": "test/workflow",
            "effective_parameters": {"experts": experts},
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
