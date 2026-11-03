#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
# Explicit native build and wheel packaging; never turn component failure into success.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec python "${PROJECT_ROOT}/hyper_multicore/_build/build_native.py" "$@"
