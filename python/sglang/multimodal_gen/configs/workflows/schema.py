# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_INPUT_IMAGE_KEYS = ("image", "image_url", "input_reference", "reference_url")


def _as_dict(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return dict(value)


def _as_list(value: Any, field_name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return list(value)


@dataclass(frozen=True)
class WorkflowSamplerSpec:
    type: str
    schedule: str | None = None
    flow_shift: float | None = None
    boundary_ratio: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "WorkflowSamplerSpec":
        data = _as_dict(data, "sampler")
        sampler_type = data.get("type")
        if not sampler_type:
            raise ValueError("sampler.type is required")
        return cls(
            type=str(sampler_type),
            schedule=data.get("schedule"),
            flow_shift=data.get("flow_shift"),
            boundary_ratio=data.get("boundary_ratio"),
        )

    def to_effective_parameters(self) -> dict[str, Any]:
        result: dict[str, Any] = {"sampler": self.type}
        if self.schedule is not None:
            result["schedule"] = self.schedule
        if self.flow_shift is not None:
            result["flow_shift"] = self.flow_shift
        if self.boundary_ratio is not None:
            result["boundary_ratio"] = self.boundary_ratio
        return result


@dataclass(frozen=True)
class WorkflowExpertRange:
    name: str
    component: str
    start_step: int | None = None
    end_step: int | None = None
    guidance_param: str | None = None
    flow_shift: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowExpertRange":
        data = _as_dict(data, "expert")
        name = data.get("name")
        component = data.get("component")
        if not name:
            raise ValueError("expert.name is required")
        if not component:
            raise ValueError("expert.component is required")
        start_step = data.get("start_step")
        end_step = data.get("end_step")
        if start_step is not None and int(start_step) < 0:
            raise ValueError(f"expert {name!r} has negative start_step")
        if end_step is not None and int(end_step) < 0:
            raise ValueError(f"expert {name!r} has negative end_step")
        if (
            start_step is not None
            and end_step is not None
            and int(end_step) < int(start_step)
        ):
            raise ValueError(f"expert {name!r} end_step is before start_step")
        return cls(
            name=str(name),
            component=str(component),
            start_step=int(start_step) if start_step is not None else None,
            end_step=int(end_step) if end_step is not None else None,
            guidance_param=data.get("guidance_param"),
            flow_shift=(
                float(data["flow_shift"])
                if data.get("flow_shift") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "component": self.component,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "guidance_param": self.guidance_param,
            "flow_shift": self.flow_shift,
        }


@dataclass(frozen=True)
class WorkflowOverridePolicy:
    allow: frozenset[str] = field(default_factory=frozenset)
    deny: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "WorkflowOverridePolicy":
        data = _as_dict(data, "override_policy")
        return cls(
            allow=frozenset(str(item) for item in _as_list(data.get("allow"), "allow")),
            deny=frozenset(str(item) for item in _as_list(data.get("deny"), "deny")),
        )

    def validate_keys(self, keys: set[str]) -> None:
        denied = keys & self.deny
        if denied:
            raise ValueError(
                "Workflow request overrides denied fields: " + ", ".join(sorted(denied))
            )
        if not self.allow:
            return
        disallowed = keys - self.allow
        if disallowed:
            raise ValueError(
                "Workflow request overrides unsupported fields: "
                + ", ".join(sorted(disallowed))
            )


@dataclass(frozen=True)
class WorkflowPreset:
    schema_version: int
    name: str
    task: str
    model_family: str
    pipeline: str
    base_model_id: str | None
    components: dict[str, str]
    defaults: dict[str, Any]
    sampler: WorkflowSamplerSpec
    experts: tuple[WorkflowExpertRange, ...] = ()
    decode: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    override_policy: WorkflowOverridePolicy = field(
        default_factory=WorkflowOverridePolicy
    )
    execution_status: str = "ready"
    unsupported_reason: str | None = None
    description: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowPreset":
        data = _as_dict(data, "workflow preset")
        name = data.get("name")
        task = data.get("task")
        model_family = data.get("model_family")
        pipeline = data.get("pipeline")
        if not name:
            raise ValueError("workflow preset name is required")
        if not task:
            raise ValueError(f"workflow preset {name!r} task is required")
        if not model_family:
            raise ValueError(f"workflow preset {name!r} model_family is required")
        if not pipeline:
            raise ValueError(f"workflow preset {name!r} pipeline is required")

        experts = tuple(
            WorkflowExpertRange.from_dict(item)
            for item in _as_list(data.get("experts"), "experts")
        )
        cls._validate_experts(experts)
        execution_status = str(data.get("execution_status", "ready"))
        if execution_status not in {"ready", "unsupported"}:
            raise ValueError(
                f"workflow preset {name!r} has invalid execution_status: "
                f"{execution_status!r}"
            )

        return cls(
            schema_version=int(data.get("schema_version", 1)),
            name=str(name),
            task=str(task).lower(),
            model_family=str(model_family),
            pipeline=str(pipeline),
            base_model_id=data.get("base_model_id"),
            components={
                str(k): str(v)
                for k, v in _as_dict(data.get("components"), "components").items()
            },
            defaults=_as_dict(data.get("defaults"), "defaults"),
            sampler=WorkflowSamplerSpec.from_dict(data.get("sampler")),
            experts=experts,
            decode=_as_dict(data.get("decode"), "decode"),
            output=_as_dict(data.get("output"), "output"),
            override_policy=WorkflowOverridePolicy.from_dict(
                data.get("override_policy")
            ),
            execution_status=execution_status,
            unsupported_reason=data.get("unsupported_reason"),
            description=data.get("description"),
        )

    @staticmethod
    def _validate_experts(experts: tuple[WorkflowExpertRange, ...]) -> None:
        fixed_ranges: list[tuple[int, int, str]] = []
        for expert in experts:
            if expert.start_step is None or expert.end_step is None:
                continue
            fixed_ranges.append((expert.start_step, expert.end_step, expert.name))
        fixed_ranges.sort()
        for idx in range(1, len(fixed_ranges)):
            prev_start, prev_end, prev_name = fixed_ranges[idx - 1]
            start, _end, name = fixed_ranges[idx]
            if start < prev_end:
                raise ValueError(
                    f"expert ranges overlap: {prev_name!r} [{prev_start}, {prev_end}) "
                    f"and {name!r}"
                )

    def resolve(
        self,
        *,
        input_values: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
    ) -> "WorkflowExecutionPlan":
        input_values = _as_dict(input_values, "input")
        parameters = _as_dict(parameters, "parameters")
        output = _as_dict(output, "output")

        requested_keys = set(input_values) | set(parameters)
        requested_keys.update(f"output.{key}" for key in output)
        self.override_policy.validate_keys(requested_keys)

        effective_parameters = dict(self.defaults)
        effective_parameters.update(parameters)

        effective_output = dict(self.output)
        effective_output.update(output)

        prompt = input_values.get("prompt")
        if not prompt:
            raise ValueError("input.prompt is required")

        image_reference = None
        for key in _INPUT_IMAGE_KEYS:
            if input_values.get(key):
                image_reference = input_values[key]
                break

        negative_prompt = input_values.get("negative_prompt")
        if negative_prompt is None:
            negative_prompt = effective_parameters.get("negative_prompt")

        return WorkflowExecutionPlan(
            preset=self,
            prompt=str(prompt),
            negative_prompt=negative_prompt,
            image_reference=image_reference,
            parameters=effective_parameters,
            output=effective_output,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "model_family": self.model_family,
            "pipeline": self.pipeline,
            "base_model_id": self.base_model_id,
            "description": self.description,
            "defaults": self.defaults,
            "sampler": self.sampler.to_effective_parameters(),
            "experts": [expert.to_dict() for expert in self.experts],
            "execution_status": self.execution_status,
            "unsupported_reason": self.unsupported_reason,
        }


@dataclass(frozen=True)
class WorkflowExecutionPlan:
    preset: WorkflowPreset
    prompt: str
    negative_prompt: str | None
    image_reference: Any | None
    parameters: dict[str, Any]
    output: dict[str, Any]

    @property
    def workflow_name(self) -> str:
        return self.preset.name

    @property
    def task(self) -> str:
        return self.preset.task

    def effective_parameters(self) -> dict[str, Any]:
        effective = dict(self.parameters)
        if self.negative_prompt is not None:
            effective["negative_prompt"] = self.negative_prompt
        effective.update(self.preset.sampler.to_effective_parameters())
        if self.preset.experts:
            effective["experts"] = [expert.to_dict() for expert in self.preset.experts]
        return effective

    def to_video_request_kwargs(
        self, *, input_reference: str | None = None
    ) -> dict[str, Any]:
        kwargs = dict(self.parameters)
        kwargs["prompt"] = self.prompt
        if self.negative_prompt is not None:
            kwargs["negative_prompt"] = self.negative_prompt
        if input_reference:
            kwargs["input_reference"] = input_reference
        return kwargs
