// Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <new>
#include <tuple>
#include <vector>

#include "csrc/prefix_search.h"

namespace ReplicaHost {
namespace {
using Rows = std::vector<int64_t>;
// The CPython L format requires the type returned by PyLong_AsLongLong.
using PyLongValue = decltype(PyLong_AsLongLong(nullptr));

bool ReadBoundedTuple(PyObject *value, Py_ssize_t max_size, int64_t limit, Rows *out) {
  if (!PyTuple_CheckExact(value) || PyTuple_GET_SIZE(value) == 0 || PyTuple_GET_SIZE(value) > max_size) {
    return false;
  }
  out->reserve(PyTuple_GET_SIZE(value));
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

bool InvalidPhase(const PhasePrices::value_type &phase) {
  return !std::isfinite(phase.first) || !std::isfinite(phase.second);
}

bool ReadPhases(PyObject *tuple, PhasePrices *out) {
  if (!PyTuple_Check(tuple)) return false;
  out->reserve(PyTuple_GET_SIZE(tuple));
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(tuple); ++i) {
    double sent, received;
    if (!PyArg_ParseTuple(PyTuple_GET_ITEM(tuple, i), "dd", &sent, &received)) return false;
    out->emplace_back(sent, received);
  }
  return std::none_of(out->begin(), out->end(), InvalidPhase);
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

using PrefixKey = std::tuple<double, int, double, size_t, int64_t>;

struct PrefixSelection {
  int64_t peak;
  int64_t best_seen;
  bool has_best;
  PrefixKey best;
  std::vector<PrefixMove> moves;
};

struct PrefixScore {
  PyLongValue capacity, resident;
  double compute, setup, resize;
};

bool ConsiderPrefixes(std::vector<PrefixMove> moves, int ranks, int slots, const PrefixScore &score,
                      const PhasePrices &phases, PrefixSelection *selection, int *width) {
  std::vector<int> occupied(ranks);
  size_t accepted = 0;
  double weighted_edges = 0.;
  for (size_t i = 0; i < moves.size(); ++i) {
    const auto &move = moves[i];
    selection->best_seen = std::min(selection->best_seen, move.peak);
    weighted_edges += move.charge;
    if (!occupied.empty() && ++occupied[move.guest] > *width) *width = std::min(slots, std::max(1, *width * 2));
    if (score.capacity >= 0 && move.peak > score.capacity) continue;
    double delta = (move.peak - selection->peak) * score.compute + score.setup +
        TransferCost(move.sent, move.received, phases);
    if (score.resize != 0. && move.peak <= score.resident) delta -= score.resize;
    if (!std::isfinite(delta) || !std::isfinite(weighted_edges)) return false;
    const auto key = std::make_tuple(delta, *width, weighted_edges, i + 1, move.peak);
    if (!selection->has_best || key < selection->best) {
      selection->best = key;
      selection->has_best = true;
      accepted = i + 1;
    }
  }
  if (accepted != 0) {
    moves.resize(accepted);
    selection->moves = std::move(moves);
  }
  return true;
}

PyObject *BuildPrefixResult(const PrefixSelection &selection) {
  PyObject *result = PyList_New(selection.moves.size());
  if (result == nullptr) return nullptr;
  for (size_t i = 0; i < selection.moves.size(); ++i) {
    const auto &move = selection.moves[i];
    PyObject *row = Py_BuildValue("(iiiLL)", move.owner, move.guest, move.expert,
                                  static_cast<PyLongValue>(move.first), static_cast<PyLongValue>(move.last));
    if (row == nullptr) {
      Py_DECREF(result);
      return nullptr;
    }
    PyList_SET_ITEM(result, i, row);
  }
  return Py_BuildValue("NL", result, static_cast<PyLongValue>(selection.best_seen));
}

}  // namespace

PyObject *SelectPrefix(PyObject *, PyObject *args) {
  PyObject *totals, *matrix, *levels, *topology, *phase_tuple;
  int ranks, slots, quantum, search_single_slot;
  PrefixScore score;
  double multiplier;
  if (!PyArg_ParseTuple(args, "OOiiiOOOddddLLp", &totals, &matrix, &ranks, &slots, &quantum,
                        &levels, &topology, &phase_tuple, &multiplier, &score.compute, &score.setup,
                        &score.resize, &score.capacity, &score.resident, &search_single_slot)) return nullptr;
  try {
    Rows weights, nodes, targets;
    std::vector<Rows> counts;
    PhasePrices phases;
    if (quantum < 1 || !std::isfinite(multiplier) || !std::isfinite(score.compute) ||
        !std::isfinite(score.setup) || !std::isfinite(score.resize) ||
        !ReadBoundedTuple(levels, 3, int64_t{3} << 30, &targets) ||
        !ReadSearch(totals, topology, ranks, slots, &weights, &nodes) ||
        !ReadCountMatrix(matrix, ranks, weights.size(), &counts)) Py_RETURN_NONE;
    if (!ReadPhases(phase_tuple, &phases)) {
      if (PyErr_Occurred()) return nullptr;
      Py_RETURN_NONE;
    }
    const auto loads = HomeLoads(weights, weights.size() / ranks, ranks);
    const auto peak = *std::max_element(loads.begin(), loads.end());
    PrefixSelection selection{peak, peak, score.capacity < 0, {0., 0, 0., 0, peak}, {}};
    const int compare_ranks = search_single_slot && slots > 1 ? ranks : 0;
    for (const auto target : targets) {
      for (const int budget : {slots, 1}) {
        auto moves = SearchPrefixes(weights, loads, nodes, counts, budget, quantum, target, multiplier, phases);
        int width = 0;
        if (!ConsiderPrefixes(std::move(moves), compare_ranks, slots, score, phases, &selection, &width)) {
          Py_RETURN_NONE;
        }
        if (width <= 1) break;
      }
    }
    return BuildPrefixResult(selection);
  } catch (const std::bad_alloc &) {
    return PyErr_NoMemory();
  }
}

}  // namespace ReplicaHost

namespace {
PyMethodDef methods[] = {
  {"select_prefix", ReplicaHost::SelectPrefix, METH_VARARGS, "Choose a capacity-legal prefix across target loads."},
  {nullptr, nullptr, 0, nullptr},
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_replica_host", nullptr, -1, methods, nullptr, nullptr, nullptr};
}  // namespace

PyMODINIT_FUNC PyInit__replica_host() {
  PyObject *result = PyModule_Create(&module);
  if (result != nullptr && PyModule_AddIntConstant(result, "PLANNER_ABI", 2) < 0) {
    Py_DECREF(result);
    return nullptr;
  }
  return result;
}
