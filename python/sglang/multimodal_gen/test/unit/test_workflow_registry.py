# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.workflows import get_workflow_registry
from sglang.multimodal_gen.configs.workflows.schema import WorkflowPreset


def test_builtin_wan2_2_remix_presets_load():
    registry = get_workflow_registry()

    names = [preset.name for preset in registry.list("wan2.2-remix")]

    assert names == [
        "wan2.2-remix/nsfw-i2v-comfy-v1",
        "wan2.2-remix/nsfw-i2v-sglang-v1",
        "wan2.2-remix/nsfw-t2v-comfy-v1",
        "wan2.2-remix/nsfw-t2v-sglang-v1",
        "wan2.2-remix/sfw-t2v-sglang-v1",
    ]
    summary = registry.get("wan2.2-remix/nsfw-i2v-sglang-v1").summary()
    assert summary["experts"][0]["component"] == "transformer"
    assert summary["execution_status"] == "ready"


def test_comfy_presets_are_discoverable_but_not_executable():
    registry = get_workflow_registry()
    preset = registry.get("wan2.2-remix/nsfw-t2v-comfy-v1")

    summary = preset.summary()

    assert summary["execution_status"] == "unsupported"
    assert "Euler/simple" in summary["unsupported_reason"]
    assert summary["sampler"] == {
        "sampler": "euler",
        "schedule": "simple",
        "flow_shift": 5.0,
    }
    assert summary["experts"][0]["end_step"] == 10
    assert summary["experts"][1]["start_step"] == 10


def test_resolve_t2v_workflow_plan_with_overrides():
    preset = get_workflow_registry().get("wan2.2-remix/nsfw-t2v-sglang-v1")

    plan = preset.resolve(
        input_values={
            "prompt": "a short video prompt",
            "negative_prompt": "low quality",
        },
        parameters={
            "seed": 123,
            "num_inference_steps": 8,
            "guidance_scale": 2.0,
        },
        output={"response_format": "url"},
    )

    assert plan.workflow_name == "wan2.2-remix/nsfw-t2v-sglang-v1"
    assert plan.prompt == "a short video prompt"
    assert plan.negative_prompt == "low quality"
    assert plan.parameters["num_inference_steps"] == 8
    assert plan.parameters["guidance_scale"] == 2.0
    assert plan.parameters["guidance_scale_2"] == 3.0
    assert plan.effective_parameters()["sampler"] == "flow_unipc"
    assert plan.effective_parameters()["boundary_ratio"] == 0.875

    video_kwargs = plan.to_video_request_kwargs()
    assert video_kwargs["prompt"] == "a short video prompt"
    assert video_kwargs["negative_prompt"] == "low quality"
    assert video_kwargs["num_inference_steps"] == 8
    assert "sampler" not in video_kwargs


def test_i2v_workflow_accepts_image_reference():
    preset = get_workflow_registry().get("wan2.2-remix/nsfw-i2v-sglang-v1")

    plan = preset.resolve(
        input_values={
            "prompt": "animate this image",
            "image_url": "https://example.com/input.png",
        },
        parameters={"num_frames": 33},
    )

    assert plan.task == "i2v"
    assert plan.image_reference == "https://example.com/input.png"
    assert plan.parameters["num_frames"] == 33

    video_kwargs = plan.to_video_request_kwargs(input_reference="/tmp/input.png")
    assert video_kwargs["input_reference"] == "/tmp/input.png"


def test_override_policy_rejects_denied_and_unknown_fields():
    preset = get_workflow_registry().get("wan2.2-remix/nsfw-t2v-sglang-v1")

    with pytest.raises(ValueError, match="denied fields"):
        preset.resolve(
            input_values={"prompt": "hello"},
            parameters={"components": {"transformer": "other"}},
        )

    with pytest.raises(ValueError, match="unsupported fields"):
        preset.resolve(
            input_values={"prompt": "hello"},
            parameters={"sampler": "euler"},
        )


def test_registry_validates_served_task_type():
    registry = get_workflow_registry()
    t2v_preset = registry.get("wan2.2-remix/nsfw-t2v-sglang-v1")
    i2v_preset = registry.get("wan2.2-remix/nsfw-i2v-sglang-v1")

    t2v_server = SimpleNamespace(
        pipeline_config=SimpleNamespace(task_type=ModelTaskType.T2V)
    )
    registry.validate_for_server(t2v_preset, t2v_server)
    with pytest.raises(ValueError, match="requires task"):
        registry.validate_for_server(i2v_preset, t2v_server)

    ti2v_server = SimpleNamespace(
        pipeline_config=SimpleNamespace(task_type=ModelTaskType.TI2V)
    )
    registry.validate_for_server(t2v_preset, ti2v_server)
    registry.validate_for_server(i2v_preset, ti2v_server)


def test_registry_rejects_unsupported_workflow():
    registry = get_workflow_registry()
    comfy_preset = registry.get("wan2.2-remix/nsfw-t2v-comfy-v1")
    server = SimpleNamespace(
        pipeline_config=SimpleNamespace(task_type=ModelTaskType.T2V)
    )

    with pytest.raises(ValueError, match="not executable"):
        registry.validate_for_server(comfy_preset, server)


def test_expert_range_overlap_is_rejected():
    data = {
        "name": "test/workflow",
        "task": "t2v",
        "model_family": "test",
        "pipeline": "TestPipeline",
        "components": {},
        "defaults": {},
        "sampler": {"type": "test"},
        "experts": [
            {"name": "a", "component": "transformer", "start_step": 0, "end_step": 4},
            {"name": "b", "component": "transformer_2", "start_step": 3, "end_step": 8},
        ],
    }

    with pytest.raises(ValueError, match="overlap"):
        WorkflowPreset.from_dict(data)
