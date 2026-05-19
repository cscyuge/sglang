# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.workflows.schema import WorkflowPreset


class WorkflowRegistry:
    def __init__(self, presets: list[WorkflowPreset] | None = None) -> None:
        self._presets: dict[str, WorkflowPreset] = {}
        for preset in presets or []:
            self.register(preset)

    @classmethod
    def from_package_presets(cls) -> "WorkflowRegistry":
        registry = cls()
        preset_package = "sglang.multimodal_gen.configs.workflows.presets"
        for resource in resources.files(preset_package).iterdir():
            if resource.name.startswith("_") or resource.suffix != ".json":
                continue
            data = json.loads(resource.read_text(encoding="utf-8"))
            if isinstance(data, list):
                items = data
            else:
                items = data.get("presets", [data])
            for item in items:
                registry.register(WorkflowPreset.from_dict(item))
        return registry

    def register(self, preset: WorkflowPreset) -> None:
        if preset.name in self._presets:
            raise ValueError(f"Duplicate workflow preset: {preset.name}")
        self._presets[preset.name] = preset

    def get(self, name: str) -> WorkflowPreset:
        try:
            return self._presets[name]
        except KeyError as exc:
            raise KeyError(f"Unknown workflow preset: {name}") from exc

    def list(self, model_family: str | None = None) -> list[WorkflowPreset]:
        presets = list(self._presets.values())
        if model_family is not None:
            presets = [
                preset for preset in presets if preset.model_family == model_family
            ]
        return sorted(presets, key=lambda preset: preset.name)

    def validate_for_server(self, preset: WorkflowPreset, server_args: Any) -> None:
        pipeline_config = getattr(server_args, "pipeline_config", None)
        task_type = getattr(pipeline_config, "task_type", None)
        if task_type is None:
            return

        if isinstance(task_type, ModelTaskType):
            task_name = task_type.name.lower()
        else:
            task_name = str(task_type).lower()

        compatible_tasks = {task_name}
        if task_name == "ti2v":
            compatible_tasks.update({"t2v", "i2v"})
        if preset.task not in compatible_tasks:
            raise ValueError(
                f"Workflow {preset.name!r} requires task {preset.task!r}, "
                f"but the served model task is {task_name!r}"
            )


@lru_cache(maxsize=1)
def get_workflow_registry() -> WorkflowRegistry:
    return WorkflowRegistry.from_package_presets()
