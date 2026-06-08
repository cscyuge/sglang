# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    Wan2_2_I2V_A14B_Config,
    _apply_painter_i2v_advanced_conditioning,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


def _workflow_batch() -> Req:
    return Req(
        height=16,
        width=16,
        num_frames=9,
        extra={
            "workflow": {
                "effective_parameters": {
                    "image_conditioning": {
                        "type": "painter_i2v_advanced",
                        "motion_amplitude": 1.3,
                        "color_protect": True,
                        "correct_strength": 0.05,
                        "enhanced_experts": ["high_noise"],
                        "original_experts": ["low_noise"],
                    }
                }
            }
        },
    )


def _workflow_batch_with_parameter_overrides() -> Req:
    batch = _workflow_batch()
    batch.extra["workflow"]["effective_parameters"].update(
        {
            "motion_amplitude": 1.0,
            "color_protect": False,
            "correct_strength": 0.0,
        }
    )
    return batch


def test_painter_i2v_motion_amplitude_matches_node_formula():
    latent_condition = torch.arange(1 * 16 * 3 * 2 * 2, dtype=torch.float32).reshape(
        1, 16, 3, 2, 2
    )
    motion_amplitude = 1.3

    base_latent = latent_condition[:, :, 0:1]
    gray_latent = latent_condition[:, :, 1:]
    diff = gray_latent - base_latent
    diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
    expected = torch.cat(
        [
            base_latent,
            torch.clamp(
                base_latent + (diff - diff_mean) * motion_amplitude + diff_mean,
                -6,
                6,
            ),
        ],
        dim=2,
    )

    actual = _apply_painter_i2v_advanced_conditioning(
        latent_condition,
        motion_amplitude=motion_amplitude,
        color_protect=False,
        correct_strength=0.0,
    )

    assert torch.equal(actual, expected)


def test_wan_painter_i2v_builds_high_and_low_image_latent_variants():
    config = Wan2_2_I2V_A14B_Config()
    latent_condition = torch.linspace(-1.0, 1.0, 1 * 16 * 3 * 2 * 2).reshape(
        1, 16, 3, 2, 2
    )

    base_batch = Req(height=16, width=16, num_frames=9)
    original_image_latents = config.postprocess_image_latent(
        latent_condition, base_batch
    )

    workflow_batch = _workflow_batch()
    default_image_latents = config.postprocess_image_latent(
        latent_condition, workflow_batch
    )
    config.finalize_image_latent_variants(workflow_batch)

    variants = workflow_batch.extra["workflow_image_latents"]
    assert set(variants) == {"high_noise", "low_noise"}
    assert torch.equal(variants["low_noise"], original_image_latents)
    assert torch.equal(variants["high_noise"], default_image_latents)
    assert not torch.equal(variants["high_noise"], variants["low_noise"])
    assert variants["high_noise"].shape == original_image_latents.shape


def test_wan_painter_i2v_honors_top_level_workflow_parameter_overrides():
    config = Wan2_2_I2V_A14B_Config()
    latent_condition = torch.linspace(-1.0, 1.0, 1 * 16 * 3 * 2 * 2).reshape(
        1, 16, 3, 2, 2
    )

    workflow_batch = _workflow_batch_with_parameter_overrides()
    default_image_latents = config.postprocess_image_latent(
        latent_condition, workflow_batch
    )
    config.finalize_image_latent_variants(workflow_batch)

    variants = workflow_batch.extra["workflow_image_latents"]
    assert torch.equal(variants["high_noise"], variants["low_noise"])
    assert torch.equal(variants["high_noise"], default_image_latents)
