# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs import WanI2V480PConfig
from sglang.multimodal_gen.configs.workflows import get_workflow_registry
from sglang.multimodal_gen.runtime.entrypoints.openai.workflow_api import (
    _resolve_effective_workflow_parameters,
)


def test_i2v_workflow_effective_parameters_include_resolved_output_size(tmp_path):
    image_path = tmp_path / "portrait.png"
    Image.new("RGB", (1489, 2592), color="red").save(image_path)
    preset = get_workflow_registry().get("wan2.2-remix/nsfw-i2v-comfy-v1")
    plan = preset.resolve(
        input_values={
            "prompt": "animate this image",
            "input_reference": str(image_path),
        },
        parameters={"width": 480, "height": 832},
    )

    effective = _resolve_effective_workflow_parameters(
        plan,
        SimpleNamespace(pipeline_config=WanI2V480PConfig()),
        str(image_path),
    )

    assert effective["width"] == 480
    assert effective["height"] == 832
    assert effective["output_width"] == 464
    assert effective["output_height"] == 832


def test_t2v_workflow_effective_parameters_echo_requested_output_size():
    preset = get_workflow_registry().get("wan2.2-remix/nsfw-t2v-comfy-v1")
    plan = preset.resolve(
        input_values={"prompt": "a short video"},
        parameters={"width": 832, "height": 480},
    )

    effective = _resolve_effective_workflow_parameters(
        plan,
        SimpleNamespace(pipeline_config=None),
        None,
    )

    assert effective["output_width"] == 832
    assert effective["output_height"] == 480
