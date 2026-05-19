# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import shutil
import tempfile
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Path, Query, Request
from fastapi.responses import FileResponse

from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.multimodal_gen.configs.workflows import get_workflow_registry
from sglang.multimodal_gen.configs.workflows.schema import WorkflowExecutionPlan
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
    WorkflowListResponse,
    WorkflowRunListResponse,
    WorkflowRunRequest,
    WorkflowRunResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.storage import cloud_storage
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import WORKFLOW_STORE
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    add_common_data_to_response,
    merge_image_input_list,
    process_generation_batch,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    _build_video_sampling_params,
    _save_first_input_image,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.observability.trace import extract_trace_headers

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/workflows", tags=["workflows"])


def _is_remote_or_data_image(value: Any) -> bool:
    return isinstance(value, str) and value.lower().startswith(
        ("http://", "https://", "data:image")
    )


async def _resolve_workflow_image_reference(
    plan: WorkflowExecutionPlan,
    request_id: str,
    uploads_dir: str,
    *,
    prefer_remote_source: bool,
) -> str | None:
    image_reference = plan.image_reference
    if image_reference is None:
        return None
    if isinstance(image_reference, str) and not _is_remote_or_data_image(
        image_reference
    ):
        return image_reference

    image_sources = merge_image_input_list(image_reference)
    return await _save_first_input_image(
        image_sources,
        request_id,
        uploads_dir,
        prefer_remote_source=prefer_remote_source,
    )


def _workflow_job_from_sampling(
    request_id: str,
    workflow_request: WorkflowRunRequest,
    plan: WorkflowExecutionPlan,
    sampling,
) -> Dict[str, Any]:
    output_format = str(plan.output.get("format") or "mp4")
    return {
        "id": request_id,
        "object": "workflow.run",
        "status": "queued",
        "progress": 0,
        "created_at": int(time.time()),
        "workflow": plan.workflow_name,
        "model": workflow_request.model,
        "output": {
            "type": "video",
            "url": None,
            "format": output_format,
            "file_path": os.path.abspath(sampling.output_file_path()),
            "file_paths": None,
        },
        "effective_parameters": plan.effective_parameters(),
    }


def _cleanup_temp_dirs(temp_dirs: list[str]) -> None:
    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def _dispatch_workflow_job_async(
    job_id: str,
    batch: Req,
    *,
    temp_dirs: list[str] | None = None,
    output_persistent: bool = True,
) -> None:
    from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client

    try:
        save_file_path_list, result = await process_generation_batch(
            async_scheduler_client, batch
        )
        save_file_path = save_file_path_list[0]
        cloud_url = await cloud_storage.upload_and_cleanup(save_file_path)
        persistent_path = (
            os.path.abspath(save_file_path)
            if not cloud_url and output_persistent
            else None
        )
        content_url = (
            cloud_url
            if cloud_url
            else f"/v1/workflows/runs/{job_id}/content" if output_persistent else None
        )
        update_fields = {
            "status": "completed",
            "progress": 100,
            "completed_at": int(time.time()),
            "output": {
                "type": "video",
                "url": content_url,
                "format": "mp4",
                "file_path": persistent_path,
                "file_paths": (
                    [os.path.abspath(path) for path in save_file_path_list]
                    if output_persistent
                    else None
                ),
            },
        }
        update_fields = add_common_data_to_response(
            update_fields, request_id=job_id, result=result
        )
        await WORKFLOW_STORE.update_fields(job_id, update_fields)
    except Exception as e:
        logger.error("%s", e)
        await WORKFLOW_STORE.update_fields(
            job_id, {"status": "failed", "error": {"message": str(e)}}
        )
    finally:
        for temp_dir in temp_dirs or []:
            shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("", response_model=WorkflowListResponse)
async def list_workflows(model_family: Optional[str] = Query(None)):
    registry = get_workflow_registry()
    return WorkflowListResponse(
        data=[preset.summary() for preset in registry.list(model_family=model_family)]
    )


@router.post("/runs", response_model=WorkflowRunResponse)
async def create_workflow_run(
    request: Request,
    workflow_request: WorkflowRunRequest,
):
    request_id = generate_request_id()
    server_args = get_global_server_args()
    registry = get_workflow_registry()

    try:
        preset = registry.get(workflow_request.workflow)
        registry.validate_for_server(preset, server_args)
        plan = preset.resolve(
            input_values=workflow_request.input.model_dump(exclude_none=True),
            parameters=workflow_request.parameters,
            output=workflow_request.output,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    output_format = str(plan.output.get("format") or "mp4").lower()
    if output_format != "mp4":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported workflow output format: {output_format!r}. Only 'mp4' is supported.",
        )

    if plan.task == "i2v" and plan.image_reference is None:
        raise HTTPException(
            status_code=400,
            detail="input.image, input.image_url, input.input_reference, or input.reference_url is required for image-to-video workflows",
        )

    temp_dirs: list[str] = []
    if server_args.input_save_path is not None:
        uploads_dir = server_args.input_save_path
        os.makedirs(uploads_dir, exist_ok=True)
    else:
        uploads_dir = tempfile.mkdtemp(prefix="sglang_input_")
        temp_dirs.append(uploads_dir)

    try:
        input_path = await _resolve_workflow_image_reference(
            plan,
            request_id,
            uploads_dir,
            prefer_remote_source=server_args.input_save_path is None,
        )
    except Exception as e:
        _cleanup_temp_dirs(temp_dirs)
        raise HTTPException(
            status_code=400, detail=f"Failed to process image source: {str(e)}"
        )

    effective_output_path = server_args.output_path
    output_persistent = True
    if effective_output_path is None:
        output_tmp = tempfile.mkdtemp(prefix="sglang_output_")
        temp_dirs.append(output_tmp)
        effective_output_path = output_tmp
        output_persistent = False

    video_kwargs = plan.to_video_request_kwargs(input_reference=input_path)
    video_kwargs["model"] = workflow_request.model
    video_kwargs["output_path"] = effective_output_path
    req = VideoGenerationsRequest(**video_kwargs)

    try:
        sampling_params = _build_video_sampling_params(request_id, req)
    except (ValueError, TypeError) as e:
        _cleanup_temp_dirs(temp_dirs)
        raise HTTPException(status_code=400, detail=str(e))

    job = _workflow_job_from_sampling(
        request_id, workflow_request, plan, sampling_params
    )
    if not output_persistent and job.get("output"):
        job["output"]["file_path"] = None
    await WORKFLOW_STORE.upsert(request_id, job)

    trace_headers = extract_trace_headers(request.headers)
    batch = prepare_request(
        server_args=server_args,
        sampling_params=sampling_params,
        external_trace_header=trace_headers,
    )
    effective_parameters = plan.effective_parameters()
    if plan.preset.sampler.boundary_ratio is not None and batch.boundary_ratio is None:
        batch.boundary_ratio = plan.preset.sampler.boundary_ratio
    batch.extra["workflow"] = {
        "name": plan.workflow_name,
        "effective_parameters": effective_parameters,
        "experts": effective_parameters.get("experts", []),
    }

    asyncio.create_task(
        _dispatch_workflow_job_async(
            request_id,
            batch,
            temp_dirs=temp_dirs or None,
            output_persistent=output_persistent,
        )
    )
    return WorkflowRunResponse(**job)


@router.get("/runs", response_model=WorkflowRunListResponse)
async def list_workflow_runs(
    after: Optional[str] = Query(None),
    limit: Optional[int] = Query(None, ge=1, le=100),
    order: Optional[str] = Query("desc"),
):
    order = (order or "desc").lower()
    if order not in ("asc", "desc"):
        order = "desc"
    jobs = await WORKFLOW_STORE.list_values()
    jobs.sort(key=lambda job: job.get("created_at", 0), reverse=order != "asc")

    if after is not None:
        try:
            idx = next(i for i, job in enumerate(jobs) if job["id"] == after)
            jobs = jobs[idx + 1 :]
        except StopIteration:
            jobs = []
    if limit is not None:
        jobs = jobs[:limit]
    return WorkflowRunListResponse(data=[WorkflowRunResponse(**job) for job in jobs])


@router.get("/runs/{run_id}", response_model=WorkflowRunResponse)
async def retrieve_workflow_run(run_id: str = Path(...)):
    job = await WORKFLOW_STORE.get(run_id)
    if not job:
        raise HTTPException(status_code=404, detail="Workflow run not found")
    return WorkflowRunResponse(**job)


@router.get("/runs/{run_id}/content")
async def download_workflow_run_content(run_id: str = Path(...)):
    job = await WORKFLOW_STORE.get(run_id)
    if not job:
        raise HTTPException(status_code=404, detail="Workflow run not found")
    output = job.get("output") or {}
    file_path = output.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Generation is still in-progress")
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=os.path.basename(file_path),
    )
