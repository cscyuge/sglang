# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkflowStepExpert:
    name: str
    component: str
    start_step: int | None = None
    end_step: int | None = None
    guidance_param: str | None = None

    @property
    def has_step_range(self) -> bool:
        return self.start_step is not None or self.end_step is not None

    def contains(self, step_index: int, total_steps: int) -> bool:
        start_step = 0 if self.start_step is None else self.start_step
        end_step = total_steps if self.end_step is None else self.end_step
        return start_step <= step_index < end_step


@dataclass(frozen=True)
class WorkflowDenoisingPlan:
    workflow_name: str | None
    experts: tuple[WorkflowStepExpert, ...]

    def select_expert(self, step_index: int, total_steps: int) -> WorkflowStepExpert:
        matches = [
            expert
            for expert in self.experts
            if expert.contains(step_index, total_steps)
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(
                f"Workflow {self.workflow_name!r} has no expert covering denoising "
                f"step {step_index} of {total_steps}"
            )
        names = ", ".join(expert.name for expert in matches)
        raise ValueError(
            f"Workflow {self.workflow_name!r} has overlapping experts for denoising "
            f"step {step_index}: {names}"
        )

    def count_components(self, total_steps: int) -> Counter[str]:
        counts: Counter[str] = Counter()
        for step_index in range(total_steps):
            counts[self.select_expert(step_index, total_steps).component] += 1
        return counts

    def count_dual_transformer_steps(self, total_steps: int) -> tuple[int, int]:
        counts = self.count_components(total_steps)
        unsupported = set(counts) - {"transformer", "transformer_2"}
        if unsupported:
            raise ValueError(
                "Workflow explicit denoising ranges only support transformer and "
                f"transformer_2 components, got: {', '.join(sorted(unsupported))}"
            )
        return counts["transformer"], counts["transformer_2"]


def workflow_denoising_plan_from_extra(
    extra: dict[str, Any] | None,
) -> WorkflowDenoisingPlan | None:
    if not isinstance(extra, dict):
        return None
    workflow = extra.get("workflow")
    if not isinstance(workflow, dict):
        return None

    effective_parameters = workflow.get("effective_parameters")
    if not isinstance(effective_parameters, dict):
        effective_parameters = {}

    expert_specs = workflow.get("experts", effective_parameters.get("experts"))
    if not expert_specs:
        return None
    if not isinstance(expert_specs, list):
        raise ValueError("workflow experts must be a list")

    experts = tuple(_parse_expert(spec) for spec in expert_specs)
    has_ranges = [expert.has_step_range for expert in experts]
    if not any(has_ranges):
        return None
    if not all(has_ranges):
        raise ValueError(
            "workflow experts must all define start_step or end_step when explicit "
            "denoising ranges are used"
        )

    return WorkflowDenoisingPlan(
        workflow_name=workflow.get("name"),
        experts=experts,
    )


def _parse_expert(spec: Any) -> WorkflowStepExpert:
    if not isinstance(spec, dict):
        raise ValueError("workflow expert entries must be objects")

    name = spec.get("name")
    component = spec.get("component")
    if not name:
        raise ValueError("workflow expert.name is required")
    if not component:
        raise ValueError("workflow expert.component is required")

    start_step = _optional_non_negative_int(spec.get("start_step"), "start_step", name)
    end_step = _optional_non_negative_int(spec.get("end_step"), "end_step", name)
    if start_step is not None and end_step is not None and end_step < start_step:
        raise ValueError(f"workflow expert {name!r} end_step is before start_step")

    return WorkflowStepExpert(
        name=str(name),
        component=str(component),
        start_step=start_step,
        end_step=end_step,
        guidance_param=spec.get("guidance_param"),
    )


def _optional_non_negative_int(
    value: Any, field_name: str, expert_name: Any
) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError(f"workflow expert {expert_name!r} has negative {field_name}")
    return value
