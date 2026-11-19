// Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <new>
#include <tuple>
#include <vector>

namespace ReplicaHost {
namespace {
using Rows = std::vector<int64_t>;
using Move = std::tuple<int64_t, int64_t, int, int, int, int64_t>;

bool ReadBoundedTuple(PyObject *value, Py_ssize_t max_size, int64_t limit, Rows *out) {
  if (!PyTuple_CheckExact(value) || PyTuple_GET_SIZE(value) == 0 || PyTuple_GET_SIZE(value) > max_size) {
    return false;
  }
  int64_t remaining = limit;
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(value); ++i) {
    PyObject *item = PyTuple_GET_ITEM(value, i);
    if (!PyLong_CheckExact(item)) return false;
    int overflow = 0;
    const auto current = PyLong_AsLongLongAndOverflow(item, &overflow);
    if (overflow || current < 0 || current > remaining) return false;
    remaining -= current;
    out->push_back(current);
  }
  return true;
}

bool ReadSearch(PyObject *totals, PyObject *topology, int ranks, int budget, Rows *weights, Rows *nodes) {
  // A total of 2^30 rows bounds every quadratic score by 2^60.
  return ranks > 0 && ranks <= 128 && budget >= 0 && budget <= 4096 &&
         ReadBoundedTuple(totals, 4096, int64_t{1} << 30, weights) &&
         weights->size() % ranks == 0 && ReadBoundedTuple(topology, ranks, INT64_MAX, nodes) &&
         nodes->size() == static_cast<size_t>(ranks);
}

Rows HomeLoads(const Rows &weights, int home, int ranks) {
  Rows loads(ranks);
  for (size_t expert = 0; expert < weights.size(); ++expert) loads[expert / home] += weights[expert];
  return loads;
}

struct LoadScore {
  explicit LoadScore(const Rows &loads) {
    for (size_t rank = 0; rank < loads.size(); ++rank) {
      square += loads[rank] * loads[rank];
      auto item = std::make_pair(loads[rank], static_cast<int>(rank));
      for (auto &leader : leaders) {
        if (item > leader) std::swap(leader, item);
      }
    }
  }

  int64_t Outside(int source, int target) const {
    for (const auto &entry : leaders) {
      if (entry.second != source && entry.second != target) return entry.first;
    }
    return 0;
  }

  std::array<std::pair<int64_t, int>, 3> leaders = {{{0, -1}, {0, -1}, {0, -1}}};
  int64_t square = 0;
};

template <typename Values>
PyObject *ToTuple(const Values &values) {
  PyObject *out = PyTuple_New(values.size());
  if (out == nullptr) return nullptr;
  for (size_t i = 0; i < values.size(); ++i) {
    PyObject *item = PyLong_FromLongLong(values[i]);
    if (item == nullptr) {
      Py_DECREF(out);
      return nullptr;
    }
    PyTuple_SET_ITEM(out, i, item);
  }
  return out;
}

PyObject *PackAligned(const Rows &loads, const std::vector<Move> &moves) {
  PyObject *load_tuple = ToTuple(loads);
  if (load_tuple == nullptr) return nullptr;
  PyObject *selected = PyTuple_New(moves.size());
  if (selected == nullptr) {
    Py_DECREF(load_tuple);
    return nullptr;
  }
  for (size_t i = 0; i < moves.size(); ++i) {
    const auto &[peak, square, remote, expert, target, cut] = moves[i];
    PyObject *entry = Py_BuildValue("(i(iL))", expert, target, static_cast<long long>(cut));
    if (entry == nullptr) {
      Py_DECREF(load_tuple);
      Py_DECREF(selected);
      return nullptr;
    }
    PyTuple_SET_ITEM(selected, i, entry);
  }
  return Py_BuildValue("NN", load_tuple, selected);
}
bool ReadCountMatrix(PyObject *matrix, int ranks, int experts, std::vector<Rows> *rows) {
  if (!PyTuple_CheckExact(matrix) || PyTuple_GET_SIZE(matrix) != ranks) return false;
  rows->resize(ranks);
  for (int rank = 0; rank < ranks; ++rank) {
    if (!ReadBoundedTuple(PyTuple_GET_ITEM(matrix, rank), experts, int64_t{1} << 30, &(*rows)[rank]) ||
        (*rows)[rank].size() != static_cast<size_t>(experts)) return false;
  }
  return true;
}

using Fragment = std::tuple<int64_t, int64_t, int>;

bool ReadFragments(PyObject *slots, PyObject *spans, int ranks, int home, int width,
                   std::vector<std::vector<Fragment>> *fragments) {
  if (!PyTuple_CheckExact(slots) || PyTuple_GET_SIZE(slots) != ranks ||
      !PyTuple_CheckExact(spans) || PyTuple_GET_SIZE(spans) != ranks * width) return false;
  const int experts = ranks * home;
  std::vector<int> last_rank(experts, -1);
  for (int rank = 0; rank < ranks; ++rank) {
    PyObject *row = PyTuple_GET_ITEM(slots, rank);
    if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != width) return false;
    for (int column = 0; column < width; ++column) {
      PyObject *item = PyTuple_GET_ITEM(row, column);
      int overflow = 0;
      if (!PyLong_CheckExact(item)) return false;
      const auto expert = PyLong_AsLongLongAndOverflow(item, &overflow);
      if (overflow || expert < -1 || expert >= experts ||
          (column < home && expert != rank * home + column)) return false;
      const int slot = rank * width + column;
      Rows span;
      if (!ReadBoundedTuple(PyTuple_GET_ITEM(spans, slot), 2, int64_t{1} << 32, &span) ||
          span.size() != 2 || span[0] > span[1]) return false;
      if (expert >= 0) {
        if (last_rank[expert] == rank) return false;
        last_rank[expert] = rank;
        if (span[0] < span[1]) (*fragments)[expert].emplace_back(span[0], span[1], slot);
      }
    }
  }
  return true;
}

bool FillAlignedDispatch(const std::vector<Rows> &counts, std::vector<std::vector<Fragment>> *fragments,
                         std::vector<Rows> *dispatch) {
  Rows seen(counts.size());
  for (size_t expert = 0; expert < fragments->size(); ++expert) {
    auto &ordered = (*fragments)[expert];
    std::sort(ordered.begin(), ordered.end());
    std::fill(seen.begin(), seen.end(), 0);
    int64_t end = 0;
    for (const auto &[start, stop, slot] : ordered) {
      if (start != end || start % 64 != 0) return false;
      int64_t prefix = 0;
      for (size_t rank = 0; rank < counts.size(); ++rank) {
        const auto next = prefix + counts[rank][expert];
        const auto rows = std::max(int64_t{0}, std::min(next, stop) - std::max(prefix, start));
        (*dispatch)[rank][slot] = rows;
        seen[rank] += rows;
        prefix = next;
      }
      end = stop;
    }
    int64_t total = 0;
    for (size_t rank = 0; rank < counts.size(); ++rank) {
      if (seen[rank] != counts[rank][expert]) return false;
      total += seen[rank];
    }
    if (end != total) return false;
  }
  return true;
}

PyObject *PackDispatch(const std::vector<Rows> &dispatch) {
  PyObject *result = PyTuple_New(dispatch.size());
  if (result == nullptr) return nullptr;
  for (size_t index = 0; index < dispatch.size(); ++index) {
    PyObject *row = ToTuple(dispatch[index]);
    if (row == nullptr) {
      Py_DECREF(result);
      return nullptr;
    }
    PyTuple_SET_ITEM(result, index, row);
  }
  return result;
}

int LightestWholeGuest(const std::vector<int> &available, const Rows &loads, const Rows &nodes,
                       int source, int owner, int64_t weight) {
  int guest = -1;
  for (int target : available) {
    if (loads[source] - loads[target] < weight) break;
    if (target == source || target == owner) continue;
    if (guest != -1 && loads[target] != loads[guest]) break;
    if (nodes[target] == nodes[owner]) return target;
    if (guest == -1) guest = target;
  }
  return guest;
}

}  // namespace

PyObject *AlignedDispatch(PyObject *, PyObject *args) {
  PyObject *matrix = nullptr, *slots = nullptr, *spans = nullptr;
  int ranks = 0, home = 0, width = 0;
  if (!PyArg_ParseTuple(args, "OOOiii", &matrix, &slots, &spans, &ranks, &home, &width)) return nullptr;
  if (ranks < 1 || ranks > 128 || home < 1 || width < home || width > 8192 / ranks) Py_RETURN_NONE;
  try {
    std::vector<Rows> counts;
    std::vector<std::vector<Fragment>> fragments(ranks * home);
    if (!ReadCountMatrix(matrix, ranks, ranks * home, &counts) ||
        !ReadFragments(slots, spans, ranks, home, width, &fragments)) Py_RETURN_NONE;
    std::vector<Rows> dispatch(ranks, Rows(ranks * width));
    if (!FillAlignedDispatch(counts, &fragments, &dispatch)) Py_RETURN_NONE;
    return PackDispatch(dispatch);
  } catch (const std::bad_alloc &) {
    return PyErr_NoMemory();
  }
}

PyObject *SearchAligned(PyObject *, PyObject *args) {
  PyObject *totals = nullptr, *topology = nullptr;
  int ranks = 0, budget = 0;
  long long minimum = 0;
  if (!PyArg_ParseTuple(args, "OiiLO", &totals, &ranks, &budget, &minimum, &topology)) return nullptr;
  try {
    Rows weights, nodes;
    if (minimum < 1 || !ReadSearch(totals, topology, ranks, budget, &weights, &nodes)) Py_RETURN_NONE;
    const int experts = weights.size();
    const int home = experts / ranks;
    auto loads = HomeLoads(weights, home, ranks);
    std::vector<int> guests(ranks);
    std::vector<bool> selected(experts);
    std::vector<Move> moves;
    while (budget > 0) {
      const LoadScore score(loads);
      const auto original = std::make_pair(score.leaders[0].first, score.square);
      Move best = {original.first, original.second, 0, experts, ranks, 0};
      const auto consider_owner = [&](int owner) {
        // The target only gains rows, so its old load can stay in this bound.
        const auto outside = score.Outside(owner, -1);
        for (int expert = owner * home; expert < (owner + 1) * home; ++expert) {
          const auto weight = weights[expert];
          if (selected[expert] || weight < minimum) continue;
          for (int target = 0; target < ranks; ++target) {
            const auto gap = loads[owner] - loads[target];
            if (gap <= minimum || guests[target] >= budget) continue;
            const auto lower = std::clamp(weight - gap / 2, int64_t{0}, weight) / 64 * 64;
            for (int64_t cut : {int64_t{0}, lower, lower + 64}) {
              const auto moved = weight - cut;
              if (moved < minimum || moved >= gap || (cut > 0 && cut < minimum)) continue;
              const auto peak = std::max({loads[owner] - moved, loads[target] + moved, outside});
              const auto square = score.square + 2 * moved * (moved - gap);
              const auto key = Move(peak, square, nodes[owner] != nodes[target], expert, target, cut);
              if (key < best) best = key;
            }
          }
        }
      };
      // A move from a different owner cannot reduce the heaviest rank's load.
      const int heaviest = score.leaders[0].second;
      consider_owner(heaviest);
      if (std::get<0>(best) == original.first) {
        for (int owner = 0; owner < ranks; ++owner) {
          if (owner != heaviest) consider_owner(owner);
        }
      }
      const auto &[peak, square, remote, expert, target, cut] = best;
      if (expert == experts) break;
      const auto moved = weights[expert] - cut;
      loads[expert / home] -= moved;
      loads[target] += moved;
      ++guests[target];
      selected[expert] = true;
      moves.push_back(best);
    }
    return PackAligned(loads, moves);
  } catch (const std::bad_alloc &) {
    return PyErr_NoMemory();
  }
}

PyObject *SearchWhole(PyObject *, PyObject *args) {
  PyObject *totals = nullptr, *topology = nullptr;
  int ranks = 0, budget = 0, capacity_required = 0;
  long long minimum = 0;
  if (!PyArg_ParseTuple(args, "OiiLpO", &totals, &ranks, &budget, &minimum, &capacity_required, &topology)) {
    return nullptr;
  }
  try {
    Rows weights, nodes;
    if (minimum < 0 || !ReadSearch(totals, topology, ranks, budget, &weights, &nodes)) Py_RETURN_NONE;
    const int experts = weights.size();
    const int home = experts / ranks;
    auto loads = HomeLoads(weights, home, ranks);
    std::vector<int> targets(experts), guests(ranks);
    std::vector<int> available;
    available.reserve(ranks);
    for (int expert = 0; expert < experts; ++expert) targets[expert] = expert / home;
    int cross = 0, guest_total = 0;
    for (int step = 0; step < experts * 2 + ranks * budget; ++step) {
      const LoadScore score(loads);
      auto best = std::make_tuple(score.leaders[0].first, score.square, cross, guest_total);
      int picked = -1, destination = -1;
      available.clear();
      for (int rank = 0; rank < ranks; ++rank) {
        if (guests[rank] < budget) available.push_back(rank);
      }
      std::sort(available.begin(), available.end(), [&loads](int left, int right) {
        return std::tie(loads[left], left) < std::tie(loads[right], right);
      });
      for (int expert = 0; expert < experts; ++expert) {
        const auto weight = weights[expert];
        if (weight == 0) continue;
        const int source = targets[expert], owner = expert / home;
        // A move heavier than the load gap increases the square score without
        // reducing the peak. Among the remaining guests, both scores are
        // minimized at the smallest target load; locality breaks equal loads.
        const int guest = (weight >= minimum || capacity_required)
                              ? LightestWholeGuest(available, loads, nodes, source, owner, weight)
                              : -1;
        for (int target : {std::min(owner, guest), std::max(owner, guest)}) {
          if (target < 0 || target == source || loads[source] - loads[target] < weight) continue;
          const auto peak = std::max({loads[source] - weight, loads[target] + weight, score.Outside(source, target)});
          const auto square = score.square + 2 * weight * (weight - (loads[source] - loads[target]));
          const auto key = std::make_tuple(peak, square,
              cross - (nodes[source] != nodes[owner]) + (nodes[target] != nodes[owner]),
              guest_total - (source != owner) + (target != owner));
          if (key < best) {
            best = key;
            picked = expert;
            destination = target;
          }
        }
      }
      if (picked < 0) break;
      const int source = targets[picked], owner = picked / home;
      loads[source] -= weights[picked];
      loads[destination] += weights[picked];
      guests[source] -= (source != owner);
      guests[destination] += (destination != owner);
      targets[picked] = destination;
      cross = std::get<2>(best);
      guest_total = std::get<3>(best);
    }
    PyObject *load_tuple = ToTuple(loads);
    if (load_tuple == nullptr) return nullptr;
    PyObject *target_tuple = ToTuple(targets);
    if (target_tuple == nullptr) {
      Py_DECREF(load_tuple);
      return nullptr;
    }
    return Py_BuildValue("NN", load_tuple, target_tuple);
  } catch (const std::bad_alloc &) {
    return PyErr_NoMemory();
  }
}
}  // namespace ReplicaHost

namespace {
PyMethodDef methods[] = {
  {"search_aligned", ReplicaHost::SearchAligned, METH_VARARGS, "Search aligned fragment placement."},
  {"aligned_dispatch", ReplicaHost::AlignedDispatch, METH_VARARGS, "Materialize checked aligned quotas."},
  {"search_whole", ReplicaHost::SearchWhole, METH_VARARGS, "Search whole expert placement."},
  {nullptr, nullptr, 0, nullptr},
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_replica_host", nullptr, -1, methods, nullptr, nullptr, nullptr};
}  // namespace

PyMODINIT_FUNC PyInit__replica_host() { return PyModule_Create(&module); }
