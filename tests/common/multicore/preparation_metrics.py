# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Extract device boundaries; never substitute Host return for completion."""
import re
from decimal import Decimal


def preparation_windows(trace, expected_steps):
    events = trace['traceEvents'] if isinstance(trace, dict) else trace
    hardware = {e['pid'] for e in events if e.get('name') == 'process_name'
                and e.get('args', {}).get('name') == 'Ascend Hardware'}
    events = [dict(e, ts=Decimal(str(e['ts'])), dur=Decimal(str(e.get('dur', 0))))
              for e in events if e.get('ph') == 'X']
    markers = [e for e in events if e.get('ph') == 'X' and e.get('name', '').startswith('hmc_prelude/')]
    if len(markers) != expected_steps:
        raise ValueError(f'Expected {expected_steps} invocation markers, found {len(markers)}')
    result = []
    for marker in sorted(markers, key=lambda e: e['ts']):
        _, variant, routing, iteration = marker['name'].split('/')
        left, right = marker['ts'], marker['ts'] + marker['dur']
        window = [e for e in events if e.get('ph') == 'X' and e.get('dur', 0) > 0
                  and left <= e['ts'] and e['ts'] + e['dur'] <= right]
        gathers = [e for e in window if re.match(r'^hcom_.*allgather', e.get('name', ''), re.I)]
        if len(gathers) != 1:
            raise ValueError(f"{marker['name']}: expected one count AllGather, got {len(gathers)}; "
                             'refuse to guess among FSDP/configuration/other collectives')
        gather = gathers[0]
        after = gather['ts'] + gather['dur']
        if variant == 'native':
            targets = [e for e in window if re.match(r'^hcom_.*alltoallv', e.get('name', ''), re.I)
                       and e['ts'] >= after]
        elif variant == 'mega':
            targets = [e for e in window if 'megamoe' in e.get('name', '').lower()
                       and (e.get('pid') in hardware or
                            (e.get('cat') or '').lower() in ('kernel', 'aicore_op', 'ai_core_op'))
                       and e['ts'] >= after]
        else:
            raise ValueError(f'Unsupported variant {variant}')
        if not targets:
            raise ValueError(f"{marker['name']}: missing Dispatch/core device activity")
        target = min(targets, key=lambda e: e['ts'])
        if variant == 'mega' and len(targets) != 1:
            raise ValueError(f"{marker['name']}: ambiguous MegaKernel device activities")
        result.append(dict(variant=variant, routing=routing, iteration=int(iteration),
                           allgather_name=gather['name'], allgather_end_us=str(after),
                           target_name=target['name'], target_start_us=str(target['ts']),
                           prepare_ms=float((target['ts'] - after) / 1000),
                           allgather_ms=float(gather['dur'] / 1000),
                           marker_host_ms=float(marker['dur'] / 1000)))
    return result
