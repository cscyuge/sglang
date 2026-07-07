# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import pickle
import tempfile
from collections import deque
from typing import Any, List

import zmq

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin import (
    SchedulerDisaggMixin,
)
from sglang.multimodal_gen.runtime.distributed import get_world_group
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    _parse_size,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    GetWeightsChecksumReqInput,
    UpdateWeightFromDiskReqInput,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    GetDisaggStatsReq,
    ListLorasReq,
    MergeLoraWeightsReq,
    ProfileReqOutput,
    ReleaseRealtimeSessionReq,
    SetLoraReq,
    ShutdownReq,
    StartProfileReq,
    StopProfileReq,
    UnmergeLoraWeightsReq,
)
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.realtime.session import RealtimeSessionCache
from sglang.multimodal_gen.runtime.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args,
)
from sglang.multimodal_gen.runtime.server_warmup import should_return_warmup_result
from sglang.multimodal_gen.runtime.utils.common import get_zmq_socket
from sglang.multimodal_gen.runtime.utils.distributed import broadcast_pyobj
from sglang.multimodal_gen.runtime.utils.logging_utils import GREEN, RESET, init_logger
from sglang.multimodal_gen.runtime.warmup_request_builder import (
    resolve_stream_r1_warmup_num_frames,
)

logger = init_logger(__name__)

MINIMUM_PICTURE_BASE64_FOR_WARMUP = "data:image/jpg;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="


class Scheduler(SchedulerDisaggMixin):
    """
    Runs the main event loop for the rank 0 worker.
    It listens for external requests via ZMQ and coordinates with other workers.
    This class does NOT manage worker processes.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        port_args: PortArgs,
        task_pipes_to_slaves: list = None,
        result_pipes_from_slaves: list = None,
        local_rank: int | None = None,
    ):
        self.server_args = server_args
        self.port_args = port_args

        # local_rank is the physical GPU index for torch.cuda.set_device.
        # In non-disagg mode, it equals gpu_id. In disagg mode, it may differ
        # (e.g., denoiser rank 0 on physical GPU 1).
        if local_rank is None:
            local_rank = gpu_id

        set_global_server_args(server_args=server_args)

        # Inter-process Communication
        self.context = zmq.Context(io_threads=2)
        endpoint = server_args.scheduler_endpoint
        if gpu_id == 0:
            # router allocates identify (envelope) for each connection
            self.receiver, actual_endpoint = get_zmq_socket(
                self.context, zmq.ROUTER, endpoint, True
            )
            logger.info(f"Scheduler bind at endpoint: {actual_endpoint}")
        else:
            self.receiver = None

        worker = GPUWorker(
            local_rank=local_rank,
            master_port=port_args.master_port,
            rank=gpu_id,
            server_args=server_args,
        )
        self.worker = worker
        self.task_pipes_to_slaves = task_pipes_to_slaves
        self.result_pipes_from_slaves = result_pipes_from_slaves
        self.gpu_id = gpu_id
        self._running = True
        self.realtime_session_cache = RealtimeSessionCache()

        self.request_handlers = {
            SetLoraReq: self._handle_set_lora,
            MergeLoraWeightsReq: self._handle_merge_lora,
            UnmergeLoraWeightsReq: self._handle_unmerge_lora,
            Req: self._handle_generation,
            List[Req]: self._handle_generation,
            ListLorasReq: self._handle_list_loras,
            ShutdownReq: self._handle_shutdown,
            GetDisaggStatsReq: self._handle_get_disagg_stats,
            ReleaseRealtimeSessionReq: self._handle_release_realtime_session,
            UpdateWeightFromDiskReqInput: self._handle_update_weights_from_disk,
            GetWeightsChecksumReqInput: self._handle_get_weights_checksum,
            StartProfileReq: self._handle_start_profile,
            StopProfileReq: self._handle_stop_profile,
        }

        # Profiler state
        self.torch_profiler = None
        self.profile_in_progress = False
        self.torch_profiler_output_dir = None

        # FIFO, new reqs are appended
        self.waiting_queue: deque[tuple[bytes, Req]] = deque()

        # whether we've send the necessary warmup reqs
        self.warmed_up = False
        # warmup progress tracking
        self._warmup_total = 0
        self._warmup_processed = 0

        self.prepare_server_warmup_reqs()

        # Maximum consecutive errors before terminating the event loop
        self._max_consecutive_errors = 3
        self._consecutive_error_count = 0

        self._init_disagg_state(server_args, local_rank)

    def get_disagg_metrics(self) -> dict | None:
        """Return disagg role metrics snapshot, or None if not in disagg mode."""
        if self._disagg_metrics is None:
            return None
        return self._disagg_metrics.snapshot().to_dict()

    def _handle_get_disagg_stats(self, _reqs: List[Any]) -> OutputBatch:
        """Handle stats request — return disagg metrics via OutputBatch.output."""
        stats = self.get_disagg_metrics()
        return OutputBatch(
            output=stats or {"role": "monolithic", "message": "not in disagg mode"}
        )

    def _handle_set_lora(self, reqs: List[Any]) -> OutputBatch:
        # TODO: return set status
        # TODO: return with SetLoRAResponse or something more appropriate
        req = reqs[0]
        return self.worker.set_lora(
            req.lora_nickname, req.lora_path, req.target, req.strength
        )

    def _handle_merge_lora(self, reqs: List[Any]):
        req = reqs[0]
        return self.worker.merge_lora_weights(req.target, req.strength)

    def _handle_unmerge_lora(self, reqs: List[Any]) -> OutputBatch:
        req = reqs[0]
        return self.worker.unmerge_lora_weights(req.target)

    def _handle_list_loras(self, _reqs: List[Any]) -> OutputBatch:
        return self.worker.list_loras()

    def _handle_shutdown(self, _reqs: List[Any]) -> OutputBatch:
        self._running = False
        return OutputBatch()

    def _handle_release_realtime_session(self, reqs: List[Any]) -> OutputBatch:
        req = reqs[0]
        released = self.realtime_session_cache.release(req.session_id)
        return OutputBatch(
            output={
                "session_id": req.session_id,
                "released": released,
            }
        )

    def _handle_start_profile(self, reqs: List[Any]) -> OutputBatch:
        import time

        import torch.profiler

        req = reqs[0]
        if self.profile_in_progress:
            return OutputBatch(
                output=ProfileReqOutput(
                    success=False, message="Profiling is already in progress"
                )
            )

        activity_map = {
            "CPU": torch.profiler.ProfilerActivity.CPU,
            "CUDA": torch.profiler.ProfilerActivity.CUDA,
            "GPU": torch.profiler.ProfilerActivity.CUDA,
        }
        activities = []
        for a in req.activities:
            act = activity_map.get(a.upper())
            if act is not None:
                activities.append(act)
        if not activities:
            activities = [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]

        output_dir = req.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.torch_profiler_output_dir = output_dir

        self.torch_profiler = torch.profiler.profile(
            activities=activities,
            with_stack=req.with_stack,
            record_shapes=req.record_shapes,
        )
        self.torch_profiler.start()
        self.profile_in_progress = True

        logger.info(
            "Started torch profiler on rank %d (output_dir=%s)",
            self.gpu_id,
            output_dir,
        )
        return OutputBatch(
            output=ProfileReqOutput(
                success=True, message=f"Profiling started (output_dir={output_dir})"
            )
        )

    def _handle_stop_profile(self, _reqs: List[Any]) -> OutputBatch:
        import time

        if not self.profile_in_progress or self.torch_profiler is None:
            return OutputBatch(
                output=ProfileReqOutput(
                    success=False, message="No profiling in progress"
                )
            )

        self.torch_profiler.stop()

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        trace_path = os.path.join(
            self.torch_profiler_output_dir,
            f"profile-{timestamp}-rank{self.gpu_id}.trace.json.gz",
        )
        self.torch_profiler.export_chrome_trace(trace_path)
        logger.info("Exported trace to %s", trace_path)

        self.torch_profiler = None
        self.profile_in_progress = False
        self.torch_profiler_output_dir = None

        return OutputBatch(
            output=ProfileReqOutput(
                success=True, message=f"Profiling stopped, trace saved to {trace_path}"
            )
        )

    def _handle_update_weights_from_disk(self, reqs: List[Any]) -> OutputBatch:
        """Handle update_weights_from_disk request for RL workflows."""
        req = reqs[0]
        success, message = self.worker.update_weights_from_disk(
            model_path=req.model_path,
            flush_cache=req.flush_cache,
            target_modules=req.target_modules,
        )
        return OutputBatch(
            output={"success": success, "message": message},
            error=None if success else message,
        )

    def _handle_get_weights_checksum(self, reqs: List[Any]) -> OutputBatch:
        """Handle get_weights_checksum request."""
        req = reqs[0]
        checksums = self.worker.get_weights_checksum(module_names=req.module_names)
        return OutputBatch(output=checksums)

    def _handle_generation(self, reqs: List[Req]):
        for req in reqs:
            self.realtime_session_cache.attach(req)

        warmup_reqs = [req for req in reqs if req.is_warmup]
        if warmup_reqs:
            self._warmup_processed += len(warmup_reqs)
            if self._warmup_total > 0:
                logger.info(
                    f"Processing warmup req... ({self._warmup_processed}/{self._warmup_total})"
                )
            else:
                logger.info("Processing warmup req...")
        return self.worker.execute_forward(reqs)

    def return_result(
        self,
        output_batch: OutputBatch,
        identity: bytes | None = None,
        is_warmup: bool = False,
        req_or_group: Any | None = None,
    ):
        """
        replies to client, only on rank 0
        """
        from sglang.multimodal_gen.runtime.utils.realtime_frame_store import (
            discard_raw_rgb_frame_store_writer_request,
            pop_raw_rgb_frame_store_writer_request,
            start_raw_rgb_frame_store_writer_request,
        )

        frame_store_write_request = pop_raw_rgb_frame_store_writer_request(
            output_batch
        )
        should_send = not is_warmup or should_return_warmup_result(req_or_group)
        if not should_send or self.receiver is None or identity is None:
            if frame_store_write_request is not None:
                discard_raw_rgb_frame_store_writer_request(frame_store_write_request)
            return

        sent = False
        try:
            payload = pickle.dumps(output_batch)
            self.receiver.send_multipart([identity, b"", payload])
            sent = True
        finally:
            if frame_store_write_request is not None:
                if sent:
                    try:
                        start_raw_rgb_frame_store_writer_request(
                            frame_store_write_request
                        )
                    except Exception:
                        logger.exception(
                            "failed to start realtime frame store writer"
                        )
                        discard_raw_rgb_frame_store_writer_request(
                            frame_store_write_request
                        )
                else:
                    discard_raw_rgb_frame_store_writer_request(
                        frame_store_write_request
                    )

    def get_next_batch_to_run(self) -> list[tuple[bytes, Req]] | None:
        """pull a req from waiting_queue"""
        if not self.waiting_queue:
            return None

        # pop the first (earliest)
        item = self.waiting_queue.popleft()

        return [item]

    def prepare_server_warmup_reqs(self):
        if (
            self.server_args.warmup
            and not self.warmed_up
            and self.server_args.warmup_resolutions is not None
        ):
            # insert warmup reqs constructed with each warmup-resolution
            self._warmup_total = len(self.server_args.warmup_resolutions)
            self._warmup_processed = 0
            task_type = self.server_args.pipeline_config.task_type

            requires_warmup_image = task_type.accepts_image_input()
            warmup_input_path = None
            if requires_warmup_image:
                warmup_input_path = self._prepare_shared_warmup_image_path()

            for resolution in self.server_args.warmup_resolutions:
                width, height = _parse_size(resolution)

                if requires_warmup_image:
                    req = Req(
                        data_type=task_type.data_type(),
                        width=width,
                        height=height,
                        prompt="",
                        negative_prompt="",
                        image_path=[warmup_input_path],
                    )
                    # Req() lazily creates SamplingParams with width/height=None
                    # (triggering __post_init__ to set *_not_provided=True), then
                    # sets width/height AFTER construction.  Fix the stale flags
                    # so InputValidationStage respects the warmup resolution.
                    req.width_not_provided = False
                    req.height_not_provided = False
                else:
                    req = Req(
                        data_type=task_type.data_type(),
                        width=width,
                        height=height,
                        prompt="",
                    )
                    req.width_not_provided = False
                    req.height_not_provided = False

                # FlashTalk-specific warmup: set correct frame count and
                # provide dummy audio so the audio encoder, audio cross-
                # attention, and DeepGEMM kernels are all exercised.
                pipeline_config = self.server_args.pipeline_config
                audio_encoder_path = getattr(
                    pipeline_config, "audio_encoder_path", None
                ) or self.server_args.component_paths.get("audio_encoder")
                if audio_encoder_path is not None:
                    import numpy as np

                    chunk_frame_num = resolve_stream_r1_warmup_num_frames(
                        self.server_args,
                        int(getattr(pipeline_config, "chunk_frame_num", 33)),
                    )
                    motion_frames_num = getattr(pipeline_config, "motion_frames_num", 5)
                    fps = 25
                    req.num_frames = chunk_frame_num
                    req.fps = fps
                    req.adjust_frames = False
                    # Audio must produce MORE than chunk_frame_num video frames
                    # so forward() enters the multi-chunk path (which initialises
                    # VAE torch.compile, VAE CUDA graph, Wav2Vec2 CUDA graph).
                    # Two chunks is enough: chunk_frame_num + slice_len frames.
                    slice_len = chunk_frame_num - motion_frames_num
                    warmup_video_frames = chunk_frame_num + slice_len
                    n_audio_samples = int(warmup_video_frames / fps * 16000)
                    req.extra["audio_tensor"] = np.zeros(
                        n_audio_samples, dtype=np.float32
                    )

                req.set_as_warmup(self.server_args.warmup_steps)
                self.waiting_queue.append((None, req))
            # if server is warmed-up, set this flag to avoid req-based warmup
            self.warmed_up = True

    def process_warmup(self):
        """Drain warmup queue synchronously before signaling ready.

        All ranks call this simultaneously after prepare_server_warmup_reqs().
        NCCL collectives inside forward keep ranks in sync — no broadcast needed.
        """
        if not self.waiting_queue:
            return
        logger.info(
            "Processing %d warmup request(s) synchronously...",
            len(self.waiting_queue),
        )
        while self.waiting_queue:
            items = self.get_next_batch_to_run()
            if not items:
                break
            reqs = [item[1] for item in items]
            processed_req = reqs[0]
            if not isinstance(processed_req, Req):
                continue
            try:
                output_batch = self._handle_generation(reqs)
                is_warmup = getattr(processed_req, "is_warmup", False)
                if is_warmup:
                    if output_batch.error is None:
                        logger.info(
                            "Warmup (%d/%d) done in %.2fs",
                            self._warmup_processed,
                            self._warmup_total,
                            output_batch.metrics.total_duration_s,
                        )
                    else:
                        logger.warning(
                            "Warmup (%d/%d) failed: %s",
                            self._warmup_processed,
                            self._warmup_total,
                            output_batch.error,
                        )
            except Exception as e:
                logger.error("Warmup failed: %s", e, exc_info=True)

    def _prepare_shared_warmup_image_path(self) -> str:
        world_group = get_world_group()
        src_rank = world_group.ranks[0]

        warmup_sync: dict[str, str | None]
        if world_group.rank == src_rank:
            try:
                if self.server_args.input_save_path is not None:
                    uploads_dir = self.server_args.input_save_path
                    os.makedirs(uploads_dir, exist_ok=True)
                else:
                    uploads_dir = tempfile.mkdtemp(prefix="sglang_input_")
                warmup_image_base = os.path.join(uploads_dir, "warmup_image")
                input_path = asyncio.run(
                    save_image_to_path(
                        MINIMUM_PICTURE_BASE64_FOR_WARMUP,
                        warmup_image_base,
                    )
                )
                warmup_sync = {"input_path": input_path, "error": None}
            except Exception as e:
                warmup_sync = {"input_path": None, "error": str(e)}
        else:
            warmup_sync = {}

        # Sync rank 0's warmup-image write result (path or error) to all ranks.
        warmup_sync = broadcast_pyobj(
            warmup_sync,
            world_group.rank,
            world_group.cpu_group,
            src=src_rank,
        )
        if not isinstance(warmup_sync, dict):
            raise RuntimeError("Invalid warmup sync payload received across ranks")

        error = warmup_sync.get("error")
        if error is not None:
            raise RuntimeError(
                f"Warmup image preparation failed on rank {src_rank}: {error}"
            )

        input_path = warmup_sync.get("input_path")
        if not isinstance(input_path, str) or not input_path:
            raise RuntimeError("Warmup image preparation returned empty input path")

        return input_path

    def process_received_reqs_with_req_based_warmup(
        self, recv_reqs: List[tuple[bytes, Any]]
    ) -> List[tuple[bytes, Any]]:
        if (
            self.warmed_up
            or not self.server_args.warmup
            or not recv_reqs
            or self.server_args.warmup_resolutions is not None
        ):
            return recv_reqs

        # handle server req-based warmup by inserting an identical req to the beginning of the waiting queue
        # only the very first req through server's lifetime will be warmed up
        identity, req = recv_reqs[0]
        if isinstance(req, Req):
            warmup_req = req.copy_as_warmup(self.server_args.warmup_steps)
            recv_reqs.insert(0, (identity, warmup_req))
            self._warmup_total = 1
            self._warmup_processed = 0
            self.warmed_up = True
        return recv_reqs

    def recv_reqs(self) -> List[tuple[bytes, Any]]:
        """
        For non-main schedulers, reqs are broadcasted from main using broadcast_pyobj
        """
        if self.receiver is not None:
            try:
                try:
                    # Accept valid REQ envelopes only, ignore malformed/probe frames.
                    parts = self.receiver.recv_multipart(zmq.NOBLOCK)
                    identity, payload = parts[0], parts[-1]

                    # Ignore malformed probes or non-pickle data
                    recv_reqs = pickle.loads(payload) if len(parts) > 2 else []
                except (zmq.Again, pickle.UnpicklingError, IndexError, EOFError):
                    recv_reqs = []
            except zmq.ZMQError:
                # re-raise or handle appropriately to let the outer loop continue
                raise

            if recv_reqs:
                # Ensure recv_reqs is a list
                if not isinstance(recv_reqs, list):
                    recv_reqs = [recv_reqs]

                logger.info(
                    "Scheduler: received %d request(s) from ZMQ", len(recv_reqs)
                )

                # Pack with identity for rank 0
                recv_reqs = [(identity, req) for req in recv_reqs]
        else:
            recv_reqs = None

        # TODO: fix this condition
        if self.server_args.sp_degree != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.worker.sp_group.rank,
                self.worker.sp_cpu_group,
                src=self.worker.sp_group.ranks[0],
            )

        if self.server_args.enable_cfg_parallel:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.worker.cfg_group.rank,
                self.worker.cfg_cpu_group,
                src=self.worker.cfg_group.ranks[0],
            )

        if self.server_args.tp_size > 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.worker.tp_group.rank,
                self.worker.tp_cpu_group,
                src=self.worker.tp_group.ranks[0],
            )

        assert recv_reqs is not None

        return recv_reqs

    def event_loop(self) -> None:
        """
        The main event loop that listens for ZMQ requests.
        Handles abortion
        """
        # Pool mode: all roles use the pool event loop
        if self._disagg_role != RoleType.MONOLITHIC:
            self._disagg_event_loop()
            return

        logger.debug(
            f"Rank 0 scheduler listening on tcp://*:{self.server_args.scheduler_port}"
        )

        while self._running:
            # Update queue depth for metrics
            if self._disagg_metrics:
                self._disagg_metrics.update_queue_depth(len(self.waiting_queue))

            # 1: receive requests
            try:
                new_reqs = self.recv_reqs()
                new_reqs = self.process_received_reqs_with_req_based_warmup(new_reqs)
                self.waiting_queue.extend(new_reqs)
                # Reset error count on success
                self._consecutive_error_count = 0
            except Exception as e:
                self._consecutive_error_count += 1
                logger.error(
                    f"Error receiving requests in scheduler event loop "
                    f"(attempt {self._consecutive_error_count}/{self._max_consecutive_errors}): {e}",
                    exc_info=True,
                )
                if self._consecutive_error_count >= self._max_consecutive_errors:
                    logger.error(
                        f"Maximum consecutive errors ({self._max_consecutive_errors}) reached. "
                        "Terminating scheduler event loop."
                    )
                    raise RuntimeError(
                        f"Scheduler terminated after {self._max_consecutive_errors} "
                        f"consecutive errors. Last error: {e}"
                    ) from e
                continue

            # 2: execute, make sure a reply is always sent
            items = self.get_next_batch_to_run()
            if not items:
                continue

            identities = [item[0] for item in items]
            reqs = [item[1] for item in items]
            logger.info(
                "Scheduler: dequeued request type=%s queue_remaining=%d",
                type(reqs[0]).__name__ if reqs else "?",
                len(self.waiting_queue),
            )

            try:
                processed_req = reqs[0]
                is_warmup = (
                    processed_req.is_warmup if isinstance(processed_req, Req) else False
                )

                handler = self.request_handlers.get(type(processed_req))
                if handler:
                    output_batch = handler(reqs)
                else:
                    output_batch = OutputBatch(
                        error=f"Unknown request type: {type(processed_req)}"
                    )
            except Exception as e:
                logger.error(
                    f"Error executing request in scheduler event loop: {e}",
                    exc_info=True,
                )
                output_batch = OutputBatch(error=str(e))

            # 3. return results
            try:
                is_warmup = (
                    processed_req.is_warmup if isinstance(processed_req, Req) else False
                )
                if is_warmup:
                    if output_batch.error is None:
                        if self._warmup_total > 0:
                            logger.info(
                                f"Warmup req ({self._warmup_processed}/{self._warmup_total}) processed in {GREEN}%.2f{RESET} seconds",
                                output_batch.metrics.total_duration_s,
                            )
                        else:
                            logger.info(
                                f"Warmup req processed in {GREEN}%.2f{RESET} seconds",
                                output_batch.metrics.total_duration_s,
                            )
                    else:
                        if self._warmup_total > 0:
                            logger.info(
                                f"Warmup req ({self._warmup_processed}/{self._warmup_total}) processing failed"
                            )
                        else:
                            logger.info("Warmup req processing failed")

                # TODO: Support sending back to multiple identities if batched
                self.return_result(
                    output_batch,
                    identities[0],
                    is_warmup=is_warmup,
                    req_or_group=processed_req,
                )
            except zmq.ZMQError as e:
                # Reply failed; log and keep loop alive to accept future requests
                logger.error(f"ZMQ error sending reply: {e}")
                continue

        if self.receiver is not None:
            self.receiver.close()
        self._cleanup_disagg()
        self.context.destroy(linger=0)

    def _broadcast_task(self, payload: dict[str, Any]) -> None:
        """Broadcast a task to all slave worker processes."""
        method = payload["method"]
        kwargs = {k: v for k, v in payload.items() if k != "method"}
        task = {"method": method, "kwargs": kwargs}
        for pipe in self.task_pipes_to_slaves:
            pipe.send(task)

    def _collect_slave_results(self) -> List[dict[str, Any]]:
        """Collect results from all slave worker processes."""
        results = []
        for pipe in self.result_pipes_from_slaves:
            results.append(pipe.recv())
        return results
