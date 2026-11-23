#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
ROOT="${HYPER_MULTICORE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ "$PYTHON_BIN" != */* ]]; then
    PYTHON_BIN=$(command -v "$PYTHON_BIN") || die "Python is not executable: $PYTHON_BIN"
fi
CANN_ENV_FILE="${CANN_ENV_FILE:-/usr/local/Ascend/cann/set_env.sh}"  # codespell:ignore cann
NATIVE_ENV_FILE="${NATIVE_ENV_FILE:-}"
if [[ -z "$NATIVE_ENV_FILE" && -f "$ROOT/build/native/payload/hyper_multicore/lib/set_env.bash" ]]; then
    NATIVE_ENV_FILE="$ROOT/build/native/payload/hyper_multicore/lib/set_env.bash"
fi
VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-}"
NNODES_VALUE="${NNODES:-1}"
NPROC_PER_NODE_VALUE="${NPROC_PER_NODE:-8}"
NODE_RANK_VALUE="${NODE_RANK:-0}"
MASTER_ADDR_VALUE="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_VALUE="${MASTER_PORT:-29500}"
SHMEM_PORT_VALUE="${SHMEM_PORT:-$((MASTER_PORT_VALUE + 1000))}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"

source_env() {
    local env_file=$1
    [[ -f "$env_file" ]] || die "environment file not found: $env_file"
    set +u
    # shellcheck disable=SC1090
    source "$env_file"
    set -u
}

[[ -x "$PYTHON_BIN" ]] || die "Python is not executable: $PYTHON_BIN"
[[ -f "$SCRIPT_DIR/benchmark.py" ]] || \
    die "benchmark runner is missing under $ROOT"
[[ "$NNODES_VALUE" =~ ^[0-9]+$ ]] || die "NNODES must be an integer"
[[ "$NPROC_PER_NODE_VALUE" =~ ^[0-9]+$ ]] || die "NPROC_PER_NODE must be an integer"
[[ "$NODE_RANK_VALUE" =~ ^[0-9]+$ ]] || die "NODE_RANK must be an integer"
[[ "$MASTER_PORT_VALUE" =~ ^[0-9]+$ ]] || die "MASTER_PORT must be an integer"
[[ "$SHMEM_PORT_VALUE" =~ ^[0-9]+$ ]] || die "SHMEM_PORT must be an integer"
((NNODES_VALUE > 0 && NPROC_PER_NODE_VALUE > 0)) || \
    die "NNODES and NPROC_PER_NODE must be positive"
((NODE_RANK_VALUE < NNODES_VALUE)) || die "NODE_RANK must be smaller than NNODES"

if [[ -z "$VISIBLE_DEVICES" ]]; then
    for ((device_index = 0; device_index < NPROC_PER_NODE_VALUE; device_index++)); do
        VISIBLE_DEVICES+="${VISIBLE_DEVICES:+,}$device_index"
    done
fi
[[ "$VISIBLE_DEVICES" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || \
    die "ASCEND_RT_VISIBLE_DEVICES must be a comma-separated list of nonnegative device IDs"
IFS=',' read -r -a VISIBLE_DEVICE_IDS <<< "$VISIBLE_DEVICES"
((${#VISIBLE_DEVICE_IDS[@]} >= NPROC_PER_NODE_VALUE)) || \
    die "ASCEND_RT_VISIBLE_DEVICES has fewer devices than NPROC_PER_NODE=$NPROC_PER_NODE_VALUE"
declare -A SEEN_DEVICE_IDS=()
for device_id in "${VISIBLE_DEVICE_IDS[@]}"; do
    [[ -z "${SEEN_DEVICE_IDS[$device_id]:-}" ]] || die "duplicate visible device ID: $device_id"
    SEEN_DEVICE_IDS[$device_id]=1
done

cd "$ROOT"
source_env "$CANN_ENV_FILE"
if [[ -n "$NATIVE_ENV_FILE" ]]; then
    source_env "$NATIVE_ENV_FILE"
fi

export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
SHMEM_HOST_VALUE="${SHMEM_HOST:-$MASTER_ADDR_VALUE}"
export HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT="${HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT:-tcp://$SHMEM_HOST_VALUE:$SHMEM_PORT_VALUE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

"$PYTHON_BIN" -c 'from hyper_multicore._loader import require_multicore_environment; require_multicore_environment()'

exec timeout --kill-after=60s "$TIMEOUT_SECONDS" \
    "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes="$NNODES_VALUE" \
    --nproc_per_node="$NPROC_PER_NODE_VALUE" \
    --node_rank="$NODE_RANK_VALUE" \
    --master_addr="$MASTER_ADDR_VALUE" \
    --master_port="$MASTER_PORT_VALUE" \
    -m examples.moe.benchmark \
    "$@"
