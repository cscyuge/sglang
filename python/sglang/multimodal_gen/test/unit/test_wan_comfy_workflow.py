# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from sglang.multimodal_gen.configs.workflows import get_workflow_registry
from sglang.multimodal_gen.tools.wan_comfy_workflow import (
    compare_wan_comfy_semantics_to_preset,
    extract_wan_comfy_workflow_semantics,
    load_comfy_workflow,
)


def _node(node_id, node_type, widgets_values):
    return {
        "id": node_id,
        "type": node_type,
        "widgets_values": widgets_values,
    }


def _ksampler(node_id, start_step, end_step, *, steps=12, cfg=1.0):
    return _node(
        node_id,
        "KSamplerAdvanced",
        [
            "enable" if start_step == 0 else "disable",
            123,
            "fixed",
            steps,
            cfg,
            "euler",
            "simple",
            start_step,
            end_step,
            "enable" if end_step < steps else "disable",
        ],
    )


def _workflow(*nodes):
    return {"nodes": list(nodes)}


def test_extract_t2v_comfy_workflow_semantics_matches_builtin_preset():
    semantics = extract_wan_comfy_workflow_semantics(
        _workflow(
            _node(59, "EmptyHunyuanLatentVideo", [720, 1280, 81, 1]),
            _node(54, "ModelSamplingSD3", [5]),
            _node(55, "ModelSamplingSD3", [5]),
            _ksampler(57, 0, 10),
            _ksampler(58, 10, 10000),
            _node(99, "VHS_VideoCombine", {"frame_rate": 16}),
        )
    )

    assert semantics.task == "t2v"
    assert semantics.width == 1280
    assert semantics.height == 720
    assert semantics.num_frames == 81
    assert semantics.num_inference_steps == 12
    assert semantics.sampler == "euler"
    assert semantics.schedule == "simple"
    assert semantics.flow_shift == 5.0
    assert semantics.experts[0].start_step == 0
    assert semantics.experts[0].end_step == 10
    assert semantics.experts[1].start_step == 10
    assert semantics.experts[1].end_step is None

    preset = get_workflow_registry().get("wan2.2-remix/nsfw-t2v-comfy-v1")
    assert compare_wan_comfy_semantics_to_preset(semantics, preset) == []


def test_extract_i2v_comfy_workflow_semantics_matches_builtin_preset():
    semantics = extract_wan_comfy_workflow_semantics(
        _workflow(
            _node(107, "WanImageToVideo", [720, 1280, 33, 1]),
            _node(54, "ModelSamplingSD3", [8]),
            _node(55, "ModelSamplingSD3", [8]),
            _ksampler(57, 0, 10),
            _ksampler(58, 10, 10000),
            _node(99, "VHS_VideoCombine", {"frame_rate": 16}),
        )
    )

    assert semantics.task == "i2v"
    assert semantics.num_frames == 33
    assert semantics.flow_shift == 8.0

    preset = get_workflow_registry().get("wan2.2-remix/nsfw-i2v-comfy-v1")
    assert compare_wan_comfy_semantics_to_preset(semantics, preset) == []


def test_comfy_workflow_mismatch_reports_field():
    semantics = extract_wan_comfy_workflow_semantics(
        _workflow(
            _node(59, "EmptyHunyuanLatentVideo", [720, 1280, 81, 1]),
            _node(54, "ModelSamplingSD3", [5]),
            _node(55, "ModelSamplingSD3", [5]),
            _ksampler(57, 0, 10, steps=16),
            _ksampler(58, 10, 10000, steps=16),
            _node(99, "VHS_VideoCombine", {"frame_rate": 16}),
        )
    )

    preset = get_workflow_registry().get("wan2.2-remix/nsfw-t2v-comfy-v1")
    mismatches = compare_wan_comfy_semantics_to_preset(semantics, preset)

    assert any("defaults.num_inference_steps" in mismatch for mismatch in mismatches)


def test_disagreeing_sampler_shift_is_rejected():
    with pytest.raises(ValueError, match="shift values disagree"):
        extract_wan_comfy_workflow_semantics(
            _workflow(
                _node(59, "EmptyHunyuanLatentVideo", [720, 1280, 81, 1]),
                _node(54, "ModelSamplingSD3", [5]),
                _node(55, "ModelSamplingSD3", [8]),
                _ksampler(57, 0, 10),
                _ksampler(58, 10, 10000),
                _node(99, "VHS_VideoCombine", {"frame_rate": 16}),
            )
        )


def test_load_comfy_workflow_rejects_non_object(tmp_path):
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps([]))

    with pytest.raises(ValueError, match="must be a JSON object"):
        load_comfy_workflow(path)
