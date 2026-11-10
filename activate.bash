#!/usr/bin/env bash
# Copyright 2026 Huawei Technologies Co., Ltd. Licensed under the Apache License, Version 2.0.
# Activate this source checkout and its completed build without installing a wheel.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'ERROR: use source activate.bash to update the calling shell.' >&2
    exit 1
fi

_hyper_multicore_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P) || return 1
_hyper_multicore_payload="${_hyper_multicore_root}/build/native/payload/hyper_multicore"
if [[ ! -f "${_hyper_multicore_payload}/build_info.json" \
      || ! -f "${_hyper_multicore_payload}/adapters/manifest.json" \
      || ! -f "${_hyper_multicore_payload}/lib/set_env.bash" ]]; then
    echo "ERROR: no completed native build under ${_hyper_multicore_root}; run bash build.sh --native-only." >&2
    unset _hyper_multicore_root _hyper_multicore_payload
    return 1
fi
source "${_hyper_multicore_payload}/lib/set_env.bash" || {
    unset _hyper_multicore_root _hyper_multicore_payload
    return 1
}
export HYPER_MULTICORE_ROOT="${_hyper_multicore_root}"
case ":${PYTHONPATH:-}:" in
    *":${_hyper_multicore_root}:"*) ;;
    *) export PYTHONPATH="${_hyper_multicore_root}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
case ":${PATH}:" in
    *":${_hyper_multicore_root}/scripts:"*) ;;
    *) export PATH="${_hyper_multicore_root}/scripts:${PATH}" ;;
esac
unset _hyper_multicore_root _hyper_multicore_payload
