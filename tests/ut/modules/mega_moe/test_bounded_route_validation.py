# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Check the production bounded-route validator against an independent count oracle."""
import random
import shutil
import subprocess
from pathlib import Path

import pytest


def _expected(payload, peers, experts, hidden, capacity, tokens, topk):
    rows = [payload[i * (experts + 3):(i + 1) * (experts + 3)] for i in range(peers)]
    mask = rows[0][2]
    if not 0 <= mask < 32:
        return False
    for row in rows:
        if row[:3] != [0, tokens, mask] or sum(row[3:]) != tokens * topk:
            return False
        if any(n < 0 or n > tokens or n * hidden > 2**31 - 1 for n in row[3:]):
            return False
    local = experts // peers
    return all(sum(sum(row[3 + dest * local:3 + (dest + 1) * local]) for row in rows) <= capacity
               for dest in range(peers))


def test_bounded_route_validation_preserves_all_checks(tmp_path):
    """Exercise actual scalar device logic, including non-local overflow and empty ranks."""
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('g++ is required for the device validation logic check')
    root = Path(__file__).resolve().parents[4]
    # Compile the real policy; the harness emulates independent source and
    # destination owners without an NPU runtime or a duplicate validator.
    (tmp_path / 'kernel_operator.h').write_text('''#pragma once
#include <cstdint>
#include <climits>
#include <stdexcept>
#include <vector>
#define __aicore__
namespace AscendC {
template<typename T> struct LocalTensor {
    std::vector<T>& values;
    T GetValue(int64_t index) { return values.at(index); }
};
inline void Trap() { throw std::runtime_error("invalid bounded route"); }
}
''')
    policy = root / 'hyper_multicore/ops/common/mega_moe/route_validation.h'
    program = tmp_path / 'validate.cpp'
    program.write_text('#include "' + str(policy) + '"\n' + '''
#include <iostream>
using namespace hyper_multicore::moe_route;
int main() {
    int32_t p, e, h, t, k;
    int64_t capacity;
    while (std::cin >> p >> e >> h >> capacity >> t >> k) {
        std::vector<int32_t> counts(p * (e + 3));
        for (auto& value : counts) std::cin >> value;
        try {
            BoundedRouteValidation validation{capacity, t, k};
            for (int source = 0; source < p; ++source) {
                const auto base = source * (e + 3);
                validation.Source({counts}, base);
                int64_t total = 0;
                for (int expert = 0; expert < e; ++expert) {
                    const auto n = counts[base + 3 + expert];
                    validation.Count(n, h);
                    total += n;
                }
                validation.SourceTotal(total);
            }
            for (int dest = 0; dest < p; ++dest) {
                int64_t total = 0;
                for (int source = 0; source < p; ++source)
                    for (int j = 0; j < e / p; ++j)
                        total += counts[source * (e + 3) + 3 + dest * (e / p) + j];
                validation.DestinationTotal(total);
            }
            std::cout << 1 << '\\n';
        } catch (const std::runtime_error&) {
            std::cout << 0 << '\\n';
        }
    }
}
''')
    executable = tmp_path / 'validate'
    subprocess.run([compiler, '-std=c++17', '-O2', '-I', str(tmp_path), str(program), '-o', str(executable)],
                   check=True, capture_output=True, text=True)
    rng = random.Random(20261128)
    cases = []
    for peers, local in ((1, 1), (2, 3), (3, 7), (8, 6), (16, 3), (64, 6)):
        experts = peers * local
        for tokens in (0, 1, 127, 257):
            topk = min(experts, 3)
            rows = []
            for _ in range(peers):
                counts = [0] * experts
                for _ in range(tokens):
                    for expert in rng.sample(range(experts), topk):
                        counts[expert] += 1
                rows.extend([0, tokens, 15, *counts])
            capacity = max(sum(sum(rows[s * (experts + 3) + 3 + d * local:
                                          s * (experts + 3) + 3 + (d + 1) * local])
                               for s in range(peers)) for d in range(peers))
            args = (peers, experts, 5120, capacity, tokens, topk)
            cases.append((args, rows))
            # Every header/count condition is tested on the final source, not
            # just on this rank's portion of the exchanged count table.
            for index, value in ((0, 1), (1, tokens + 1), (2, 7), (3, -1), (3, tokens + 1)):
                changed = rows.copy()
                changed[(peers - 1) * (experts + 3) + index] = value
                cases.append((args, changed))
            invalid_mask = rows.copy()
            invalid_mask[2] = 32
            cases.append((args, invalid_mask))
            if tokens:
                cases.append(((peers, experts, 5120, capacity - 1, tokens, topk), rows))
                changed = rows.copy()
                index = next(i for i in range(3, experts + 3) if changed[i] > 0)
                changed[index] -= 1
                cases.append((args, changed))
    # Valid counts can still exceed the element-offset integer bound.
    cases.append(((1, 1, 2, 2**31 - 1, 2**31 - 1, 1), [0, 2**31 - 1, 15, 2**31 - 1]))
    text = '\n'.join(' '.join(map(str, (*args, *rows))) for args, rows in cases)
    result = subprocess.run([str(executable)], input=text, check=True, capture_output=True, text=True)
    observed = [bool(int(value)) for value in result.stdout.splitlines()]
    expected = [_expected(rows, *args) for args, rows in cases]
    assert observed == expected
    assert any(expected) and not all(expected)
