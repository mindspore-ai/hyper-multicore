// Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

namespace ReplicaHost {
using PhasePrices = std::vector<std::pair<double, double>>;

inline double TransferCost(double sent, double received, const PhasePrices& phases) {
  double cost = 0.;
  for (auto [owner, guest] : phases) cost += std::max(owner * sent, guest * received);
  return cost;
}

struct PrefixMove {
  int owner, guest, expert;
  int64_t first, last, peak;
  double sent, received, charge;
};

// Record every legal move, including equal-peak prefixes. Selection and the hard-capacity
// repair use the same score and materializer in receive.py for all execution paths.
inline std::vector<PrefixMove> SearchPrefixes(
    std::vector<int64_t> remaining, std::vector<int64_t> loads, const std::vector<int64_t>& nodes,
    const std::vector<std::vector<int64_t>>& counts, int slots, int quantum, int64_t target,
    double multiplier, const PhasePrices& phases) {
  const int ranks = loads.size();
  const int home = remaining.size() / ranks;
  std::vector<int> occupied(ranks);
  std::vector<std::vector<bool>> guests(ranks, std::vector<bool>(remaining.size()));
  std::vector<double> sent(ranks), received(ranks);
  double max_sent = 0, max_received = 0;
  std::priority_queue<std::pair<int64_t, int>> donors;
  std::vector<PrefixMove> moves;
  for (int r = 0; r < ranks; ++r) {
    if (loads[r] > target) donors.emplace(loads[r], -r);
  }
  while (!donors.empty()) {
    const int owner = -donors.top().second;
    donors.pop();
    const auto excess = loads[owner] - target;
    using Key = std::tuple<double, double, int64_t, bool, int64_t, int, int>;
    Key best{};
    PrefixMove selected{};
    selected.guest = -1;
    for (int guest = 0; guest < ranks; ++guest) {
      const auto load = loads[guest], room = target - load;
      if (room <= 0 || occupied[guest] >= slots) continue;
      const double charge = nodes[owner] == nodes[guest] ? 1.0 : multiplier;
      const double outgoing = std::max(max_sent, sent[owner] + charge);
      const double incoming = std::max(max_received, received[guest] + charge);
      const auto transfer = TransferCost(outgoing, incoming, phases);
      if (selected.guest >= 0 && std::min(room, (loads[owner] - load) / 2) - transfer < std::get<0>(best)) continue;
      for (int e = owner * home; e < (owner + 1) * home; ++e) {
        const auto end = remaining[e];
        if (end == 0 || guests[guest][e]) continue;
        const auto cut = std::max<int64_t>(0, end - std::min(excess, room));
        const auto low = cut / quantum * quantum;
        const auto start = (end - low <= room && cut - low <= quantum / 2) ? low : std::min(end, low + quantum);
        const auto moved = end - start;
        if (moved <= 0) continue;
        const auto relief = loads[owner] - std::max(loads[owner] - moved, load + moved);
        Key key(relief - transfer, -charge, std::min(moved, counts[guest][e]), start == 0, -load, -guest, -e);
        if (selected.guest < 0 || key > best) {
          best = key;
          selected = {owner, guest, e, start, end, 0, outgoing, incoming, charge};
        }
      }
    }
    if (selected.guest < 0) continue;
    const auto moved = selected.last - selected.first;
    const auto guest = selected.guest;
    loads[owner] -= moved;
    loads[guest] += moved;
    ++occupied[guest];
    guests[guest][selected.expert] = true;
    remaining[selected.expert] = selected.first;
    sent[owner] += selected.charge;
    received[guest] += selected.charge;
    max_sent = selected.sent;
    max_received = selected.received;
    if (loads[owner] > target) donors.emplace(loads[owner], -owner);
    selected.peak = *std::max_element(loads.begin(), loads.end());
    moves.push_back(selected);
  }
  return moves;
}
}  // namespace ReplicaHost
