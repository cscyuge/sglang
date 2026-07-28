#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

NUM_PROCESSES=${NUM_PROCESSES:-4}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29581}
STAGE_RANKS=${STAGE_RANKS:-0,1,2,3}
STAGE_PARALLEL_SIZE=${STAGE_PARALLEL_SIZE:-1}
BLOCKS=${BLOCKS:-1000}
WARMUP_BLOCKS=${WARMUP_BLOCKS:-20}
SHAPE=${SHAPE:-16,1,60,104}
DTYPE=${DTYPE:-bf16}
VALIDATE_EVERY=${VALIDATE_EVERY:-1}
MATMUL_SIZE=${MATMUL_SIZE:-0}
MATMUL_REPEATS=${MATMUL_REPEATS:-1}
SP_COLLECTIVE_ELEMENTS=${SP_COLLECTIVE_ELEMENTS:-0}
SP_COLLECTIVE_REPEATS=${SP_COLLECTIVE_REPEATS:-1}
SP_COLLECTIVE_PATTERN=${SP_COLLECTIVE_PATTERN:-}
SP_COLLECTIVE_PATTERN_REPEATS=${SP_COLLECTIVE_PATTERN_REPEATS:-1}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-180}
JSON_OUTPUT=${JSON_OUTPUT:-/tmp/wan_s2v_tpp_nccl_microbench.json}

export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-86400}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

exec torchrun \
    --nproc-per-node="${NUM_PROCESSES}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/benchmark_wan_s2v_tpp_nccl.py" \
    --stage-ranks "${STAGE_RANKS}" \
    --stage-parallel-size "${STAGE_PARALLEL_SIZE}" \
    --blocks "${BLOCKS}" \
    --warmup-blocks "${WARMUP_BLOCKS}" \
    --shape "${SHAPE}" \
    --dtype "${DTYPE}" \
    --validate-every "${VALIDATE_EVERY}" \
    --matmul-size "${MATMUL_SIZE}" \
    --matmul-repeats "${MATMUL_REPEATS}" \
    --sp-collective-elements "${SP_COLLECTIVE_ELEMENTS}" \
    --sp-collective-repeats "${SP_COLLECTIVE_REPEATS}" \
    --sp-collective-pattern "${SP_COLLECTIVE_PATTERN}" \
    --sp-collective-pattern-repeats "${SP_COLLECTIVE_PATTERN_REPEATS}" \
    --timeout-seconds "${TIMEOUT_SECONDS}" \
    --json-output "${JSON_OUTPUT}"
