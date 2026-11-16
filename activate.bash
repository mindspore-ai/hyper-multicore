#!/usr/bin/env bash
# Copyright 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Activate this source checkout and its completed native build without installation.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Source this script in the shell that will launch Python.' >&2
    exit 1
fi
_hmc_source_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) || return 1
_hmc_native_env="${_hmc_source_root}/build/native/payload/hyper_multicore/lib/set_env.bash"
if [[ ! -f "${_hmc_native_env}" ]]; then
    echo 'Missing source-built payload. Run bash build.sh for the target SoC first.' >&2
    unset _hmc_source_root _hmc_native_env
    return 1
fi
if ! source "${_hmc_native_env}"; then
    unset _hmc_source_root _hmc_native_env
    return 1
fi
export PYTHONPATH="${_hmc_source_root}${PYTHONPATH:+:${PYTHONPATH}}"
unset _hmc_source_root _hmc_native_env
