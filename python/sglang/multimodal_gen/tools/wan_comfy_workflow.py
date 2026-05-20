"""Extract Wan ComfyUI workflow semantics used by SGLang presets.

This tool does not execute ComfyUI graphs. It parses ComfyUI UI workflow JSON
files and extracts the small set of generation parameters that SGLang maps into
Wan workflow presets.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class WanComfyExpertSemantics:
    name: str
    component: str
    start_step: int
    end_step: int | None
    guidance_param: str


@dataclass(frozen=True)
class WanComfyWorkflowSemantics:
    task: str
    width: int
    height: int
    num_frames: int
    fps: int
    num_inference_steps: int
    guidance_scale: float
    guidance_scale_2: float
    sampler: str
    schedule: str
    flow_shift: float
    experts: tuple[WanComfyExpertSemantics, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _KSamplerAdvancedSemantics:
    node_id: int
    add_noise: str
    steps: int
    cfg: float
    sampler: str
    schedule: str
    start_step: int
    end_step: int
    return_with_leftover_noise: str


def load_comfy_workflow(path: pathlib.Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"ComfyUI workflow must be a JSON object: {path}")
    return data


def extract_wan_comfy_workflow_semantics(
    workflow: dict[str, Any],
) -> WanComfyWorkflowSemantics:
    nodes = _workflow_nodes(workflow)
    latent_node = _select_latent_node(nodes)
    task, height, width, num_frames = _parse_latent_node(latent_node)
    fps = _parse_fps(nodes)
    flow_shift = _parse_flow_shift(nodes)
    ksamplers = sorted(
        (
            _parse_ksampler_advanced(node)
            for node in nodes
            if node.get("type") == "KSamplerAdvanced"
        ),
        key=lambda sampler: (sampler.start_step, sampler.end_step),
    )
    if len(ksamplers) != 2:
        raise ValueError(
            f"Wan2.2-Remix Comfy workflow must contain two KSamplerAdvanced nodes, got {len(ksamplers)}"
        )

    high_sampler, low_sampler = ksamplers
    _validate_ksampler_pair(high_sampler, low_sampler)

    steps = high_sampler.steps
    return WanComfyWorkflowSemantics(
        task=task,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        num_inference_steps=steps,
        guidance_scale=high_sampler.cfg,
        guidance_scale_2=low_sampler.cfg,
        sampler=high_sampler.sampler,
        schedule=high_sampler.schedule,
        flow_shift=flow_shift,
        experts=(
            WanComfyExpertSemantics(
                name="high_noise",
                component="transformer",
                start_step=high_sampler.start_step,
                end_step=_normalize_end_step(high_sampler.end_step, steps),
                guidance_param="guidance_scale",
            ),
            WanComfyExpertSemantics(
                name="low_noise",
                component="transformer_2",
                start_step=low_sampler.start_step,
                end_step=_normalize_end_step(low_sampler.end_step, steps),
                guidance_param="guidance_scale_2",
            ),
        ),
    )


def compare_wan_comfy_semantics_to_preset(
    semantics: WanComfyWorkflowSemantics,
    preset: Any,
) -> list[str]:
    mismatches: list[str] = []
    _compare("task", semantics.task, preset.task, mismatches)

    defaults = preset.defaults
    _compare("defaults.width", semantics.width, defaults.get("width"), mismatches)
    _compare("defaults.height", semantics.height, defaults.get("height"), mismatches)
    _compare(
        "defaults.num_frames",
        semantics.num_frames,
        defaults.get("num_frames"),
        mismatches,
    )
    _compare("defaults.fps", semantics.fps, defaults.get("fps"), mismatches)
    _compare(
        "defaults.num_inference_steps",
        semantics.num_inference_steps,
        defaults.get("num_inference_steps"),
        mismatches,
    )
    _compare_float(
        "defaults.guidance_scale",
        semantics.guidance_scale,
        defaults.get("guidance_scale"),
        mismatches,
    )
    _compare_float(
        "defaults.guidance_scale_2",
        semantics.guidance_scale_2,
        defaults.get("guidance_scale_2"),
        mismatches,
    )

    _compare("sampler.type", semantics.sampler, preset.sampler.type, mismatches)
    _compare(
        "sampler.schedule", semantics.schedule, preset.sampler.schedule, mismatches
    )
    _compare_float(
        "sampler.flow_shift",
        semantics.flow_shift,
        preset.sampler.flow_shift,
        mismatches,
    )

    preset_experts = tuple(preset.experts)
    _compare("experts.length", len(semantics.experts), len(preset_experts), mismatches)
    for idx, (actual, expected) in enumerate(
        zip(semantics.experts, preset_experts, strict=False)
    ):
        prefix = f"experts[{idx}]"
        _compare(f"{prefix}.name", actual.name, expected.name, mismatches)
        _compare(
            f"{prefix}.component", actual.component, expected.component, mismatches
        )
        _compare(
            f"{prefix}.start_step",
            actual.start_step,
            expected.start_step,
            mismatches,
        )
        _compare(f"{prefix}.end_step", actual.end_step, expected.end_step, mismatches)
        _compare(
            f"{prefix}.guidance_param",
            actual.guidance_param,
            expected.guidance_param,
            mismatches,
        )

    return mismatches


def assert_wan_comfy_semantics_match_preset(
    semantics: WanComfyWorkflowSemantics,
    preset: Any,
) -> None:
    mismatches = compare_wan_comfy_semantics_to_preset(semantics, preset)
    if mismatches:
        raise ValueError(
            f"Comfy workflow does not match preset {preset.name!r}:\n"
            + "\n".join(f"- {mismatch}" for mismatch in mismatches)
        )


def _workflow_nodes(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("ComfyUI workflow JSON must contain a nodes list")
    if not all(isinstance(node, dict) for node in nodes):
        raise ValueError("ComfyUI workflow nodes must be objects")
    return nodes


def _select_latent_node(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    latent_nodes = [
        node
        for node in nodes
        if node.get("type") in {"EmptyHunyuanLatentVideo", "WanImageToVideo"}
    ]
    if len(latent_nodes) != 1:
        raise ValueError(
            "Wan Comfy workflow must contain exactly one EmptyHunyuanLatentVideo "
            f"or WanImageToVideo node, got {len(latent_nodes)}"
        )
    return latent_nodes[0]


def _parse_latent_node(node: dict[str, Any]) -> tuple[str, int, int, int]:
    node_type = node.get("type")
    widgets = _widgets_list(node)
    if len(widgets) < 3:
        raise ValueError(f"{node_type} node must define height, width, and frames")

    task = "i2v" if node_type == "WanImageToVideo" else "t2v"
    height = int(widgets[0])
    width = int(widgets[1])
    num_frames = int(widgets[2])
    return task, height, width, num_frames


def _parse_fps(nodes: list[dict[str, Any]]) -> int:
    video_nodes = [node for node in nodes if node.get("type") == "VHS_VideoCombine"]
    if not video_nodes:
        raise ValueError("ComfyUI workflow must contain a VHS_VideoCombine node")

    widgets = video_nodes[0].get("widgets_values")
    if not isinstance(widgets, dict):
        raise ValueError("VHS_VideoCombine widgets_values must be an object")
    if "frame_rate" not in widgets:
        raise ValueError("VHS_VideoCombine widgets_values.frame_rate is required")
    return int(widgets["frame_rate"])


def _parse_flow_shift(nodes: list[dict[str, Any]]) -> float:
    shifts = {
        float(_widgets_list(node)[0])
        for node in nodes
        if node.get("type") == "ModelSamplingSD3"
    }
    if not shifts:
        raise ValueError("ComfyUI workflow must contain a ModelSamplingSD3 node")
    if len(shifts) != 1:
        raise ValueError(f"ModelSamplingSD3 shift values disagree: {sorted(shifts)}")
    return shifts.pop()


def _parse_ksampler_advanced(node: dict[str, Any]) -> _KSamplerAdvancedSemantics:
    widgets = _widgets_list(node)
    if len(widgets) < 10:
        raise ValueError("KSamplerAdvanced widgets_values must contain 10 entries")

    return _KSamplerAdvancedSemantics(
        node_id=int(node.get("id", -1)),
        add_noise=str(widgets[0]),
        steps=int(widgets[3]),
        cfg=float(widgets[4]),
        sampler=str(widgets[5]),
        schedule=str(widgets[6]),
        start_step=int(widgets[7]),
        end_step=int(widgets[8]),
        return_with_leftover_noise=str(widgets[9]),
    )


def _validate_ksampler_pair(
    high_sampler: _KSamplerAdvancedSemantics,
    low_sampler: _KSamplerAdvancedSemantics,
) -> None:
    if high_sampler.start_step != 0:
        raise ValueError(
            f"High-noise KSamplerAdvanced must start at step 0, got {high_sampler.start_step}"
        )
    if low_sampler.start_step != high_sampler.end_step:
        raise ValueError(
            "Low-noise KSamplerAdvanced must start where high-noise sampler ends, "
            f"got {low_sampler.start_step} != {high_sampler.end_step}"
        )

    for field_name in ("steps", "sampler", "schedule"):
        high_value = getattr(high_sampler, field_name)
        low_value = getattr(low_sampler, field_name)
        if high_value != low_value:
            raise ValueError(
                f"KSamplerAdvanced {field_name} values disagree: {high_value!r} != {low_value!r}"
            )


def _normalize_end_step(end_step: int, total_steps: int) -> int | None:
    if end_step >= total_steps:
        return None
    return end_step


def _widgets_list(node: dict[str, Any]) -> list[Any]:
    widgets = node.get("widgets_values")
    if not isinstance(widgets, list):
        raise ValueError(f"Node {node.get('id')} widgets_values must be a list")
    return widgets


def _compare(
    field_name: str,
    actual: Any,
    expected: Any,
    mismatches: list[str],
) -> None:
    if actual != expected:
        mismatches.append(f"{field_name}: workflow={actual!r}, preset={expected!r}")


def _compare_float(
    field_name: str,
    actual: Any,
    expected: Any,
    mismatches: list[str],
) -> None:
    if actual is None or expected is None:
        _compare(field_name, actual, expected, mismatches)
        return
    if abs(float(actual) - float(expected)) > 1e-6:
        mismatches.append(f"{field_name}: workflow={actual!r}, preset={expected!r}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Wan ComfyUI workflow semantics and optionally compare them with an SGLang workflow preset."
    )
    parser.add_argument(
        "--workflow",
        required=True,
        type=pathlib.Path,
        help="ComfyUI workflow JSON path",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Optional SGLang workflow preset name to compare against",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    semantics = extract_wan_comfy_workflow_semantics(load_comfy_workflow(args.workflow))
    print(json.dumps(semantics.to_dict(), indent=2, sort_keys=True))

    if args.preset:
        from sglang.multimodal_gen.configs.workflows import get_workflow_registry

        preset = get_workflow_registry().get(args.preset)
        assert_wan_comfy_semantics_match_preset(semantics, preset)
        print(f"Preset {args.preset!r} matches extracted Comfy workflow semantics.")


if __name__ == "__main__":
    main()
