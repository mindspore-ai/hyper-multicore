// Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "prefix_search.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <new>
#include <tuple>
#include <vector>

namespace ReplicaHost {
namespace {
using Rows = std::vector<int64_t>;

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

bool ReadPhases(PyObject *tuple, PhasePrices *out) {
  if (!PyTuple_Check(tuple)) return false;
  out->reserve(PyTuple_GET_SIZE(tuple));
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(tuple); ++i) {
    double sent, received;
    if (!PyArg_ParseTuple(PyTuple_GET_ITEM(tuple, i), "dd", &sent, &received)) return false;
    out->emplace_back(sent, received);
  }
  return true;
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

}  // namespace

PyObject *SelectPrefix(PyObject *, PyObject *args) {
  PyObject *totals, *matrix, *levels, *topology, *phase_tuple;
  int ranks, slots, quantum;
  long long capacity, resident;
  double multiplier, compute, setup, resize;
  if (!PyArg_ParseTuple(args, "OOiiiOOOddddLL", &totals, &matrix, &ranks, &slots, &quantum,
                        &levels, &topology, &phase_tuple, &multiplier, &compute, &setup,
                        &resize, &capacity, &resident)) return nullptr;
  try {
    Rows weights, nodes, targets;
    std::vector<Rows> counts;
    PhasePrices phases;
    if (quantum < 1 || !std::isfinite(multiplier) || !std::isfinite(compute) ||
        !std::isfinite(setup) || !std::isfinite(resize) ||
        !ReadBoundedTuple(levels, 3, int64_t{3} << 30, &targets) ||
        !ReadSearch(totals, topology, ranks, slots, &weights, &nodes) ||
        !ReadCountMatrix(matrix, ranks, weights.size(), &counts)) Py_RETURN_NONE;
    if (!ReadPhases(phase_tuple, &phases)) {
      if (PyErr_Occurred()) return nullptr;
      Py_RETURN_NONE;
    }
    if (std::any_of(phases.begin(), phases.end(), [](const auto &p) {
          return !std::isfinite(p.first) || !std::isfinite(p.second);
        })) Py_RETURN_NONE;
    const auto loads = HomeLoads(weights, weights.size() / ranks, ranks);
    const auto peak = *std::max_element(loads.begin(), loads.end());
    auto best_seen = peak;
    bool has_best = capacity < 0;
    auto best = std::make_tuple(0., 0., size_t{0}, peak);
    std::vector<PrefixMove> selected;
    for (const auto target : targets) {
      auto moves = SearchPrefixes(weights, loads, nodes, counts, slots, quantum, target, multiplier, phases);
      size_t accepted = 0;
      double weighted_edges = 0.;
      for (size_t i = 0; i < moves.size(); ++i) {
        const auto &move = moves[i];
        best_seen = std::min(best_seen, move.peak);
        weighted_edges += move.charge;
        if (capacity >= 0 && move.peak > capacity) continue;
        double delta = (move.peak - peak) * compute + setup + TransferCost(move.sent, move.received, phases);
        if (resize != 0. && move.peak <= resident) delta -= resize;
        if (!std::isfinite(delta) || !std::isfinite(weighted_edges)) Py_RETURN_NONE;
        const auto key = std::make_tuple(delta, weighted_edges, i + 1, move.peak);
        if (!has_best || key < best) {
          best = key;
          has_best = true;
          accepted = i + 1;
        }
      }
      if (accepted != 0) {
        moves.resize(accepted);
        selected = std::move(moves);
      }
    }
    PyObject *result = PyList_New(selected.size());
    if (result == nullptr) return nullptr;
    for (size_t i = 0; i < selected.size(); ++i) {
      const auto &m = selected[i];
      PyObject *row = Py_BuildValue("(iiiLL)", m.owner, m.guest, m.expert,
                                    static_cast<long long>(m.first), static_cast<long long>(m.last));
      if (row == nullptr) {
        Py_DECREF(result);
        return nullptr;
      }
      PyList_SET_ITEM(result, i, row);
    }
    return Py_BuildValue("NL", result, static_cast<long long>(best_seen));
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

PyMODINIT_FUNC PyInit__replica_host() { return PyModule_Create(&module); }
