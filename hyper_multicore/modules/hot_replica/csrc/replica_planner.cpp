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
PyObject *AlignedDispatch(PyObject *, PyObject *args);
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

bool ReadPeerEdges(PyObject *matrix, int ranks, std::vector<std::vector<bool>> *peers) {
  if (matrix == Py_None) return true;
  if (!PyTuple_CheckExact(matrix) || PyTuple_GET_SIZE(matrix) != ranks) return false;
  peers->resize(ranks);
  for (int owner = 0; owner < ranks; ++owner) {
    PyObject *row = PyTuple_GET_ITEM(matrix, owner);
    if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != ranks) return false;
    auto &allowed = (*peers)[owner];
    allowed.reserve(ranks);
    for (int guest = 0; guest < ranks; ++guest) {
      PyObject *edge = PyTuple_GET_ITEM(row, guest);
      if (!PyBool_Check(edge)) return false;
      allowed.push_back(edge == Py_True);
    }
  }
  return true;
}

}  // namespace

PyObject *SelectPrefix(PyObject *, PyObject *args) {
  PyObject *totals, *matrix, *levels, *topology, *phase_tuple, *peer_matrix = Py_None;
  int ranks, slots, quantum, minimum_rows = 1, search_single_slot = 0;
  long long capacity, resident;
  double multiplier, compute, setup, resize, matrix_copy = 0.;
  if (!PyArg_ParseTuple(args, "OOiiiOOOddddLL|idOp", &totals, &matrix, &ranks, &slots, &quantum,
                        &levels, &topology, &phase_tuple, &multiplier, &compute, &setup,
                        &resize, &capacity, &resident, &minimum_rows, &matrix_copy, &peer_matrix,
                        &search_single_slot)) return nullptr;
  try {
    Rows weights, nodes, targets;
    std::vector<Rows> counts;
    std::vector<std::vector<bool>> peers;
    PhasePrices phases;
    if (quantum < 1 || minimum_rows < 1 || !std::isfinite(multiplier) || !std::isfinite(compute) ||
        !std::isfinite(setup) || !std::isfinite(resize) || !std::isfinite(matrix_copy) || matrix_copy < 0. ||
        !ReadBoundedTuple(levels, 3, int64_t{3} << 30, &targets) ||
        !ReadSearch(totals, topology, ranks, slots, &weights, &nodes) ||
        !ReadCountMatrix(matrix, ranks, weights.size(), &counts) || !ReadPeerEdges(peer_matrix, ranks, &peers)) {
      Py_RETURN_NONE;
    }
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
    auto best = std::make_tuple(0., 0, 0., size_t{0}, peak);
    const bool compare_slots = search_single_slot && slots > 1;
    std::vector<PrefixMove> selected;
    for (const auto target : targets) {
      for (const int budget : {slots, 1}) {
        auto moves = SearchPrefixes(weights, loads, nodes, counts, budget, quantum, target, multiplier, phases,
                                    minimum_rows, matrix_copy, peers);
        std::vector<int> occupied(compare_slots ? ranks : 0);
        size_t accepted = 0;
        double weighted_edges = 0.;
        int width = 0;
        for (size_t i = 0; i < moves.size(); ++i) {
          const auto &move = moves[i];
          best_seen = std::min(best_seen, move.peak);
          weighted_edges += move.charge;
          if (compare_slots && ++occupied[move.guest] > width) {
            width = width == 0 ? 1 : std::min(slots, width * 2);
          }
          if (capacity >= 0 && move.peak > capacity) continue;
          double delta = (move.peak - peak) * compute + setup + TransferCost(move.sent, move.received, phases) +
                         matrix_copy * weighted_edges;
          if (resize != 0. && move.peak <= resident) delta -= resize;
          if (!std::isfinite(delta) || !std::isfinite(weighted_edges)) Py_RETURN_NONE;
          const auto key = std::make_tuple(delta, width, weighted_edges, i + 1, move.peak);
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
        // A wide trajectory that never used a second slot is also the narrow trajectory.
        if (width <= 1) break;
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
  {"select_prefix_with_minimum_rows", ReplicaHost::SelectPrefix, METH_VARARGS, "Set the minimum rows per optional move."},
  {"select_prefix_with_matrix_cost", ReplicaHost::SelectPrefix, METH_VARARGS, "Price matrix copies on reachable edges."},
  {"select_prefix_with_slot_search", ReplicaHost::SelectPrefix, METH_VARARGS, "Compare the full and single-slot searches."},
  {"aligned_dispatch", ReplicaHost::AlignedDispatch, METH_VARARGS, "Materialize checked aligned quotas."},
  {nullptr, nullptr, 0, nullptr},
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_replica_host", nullptr, -1, methods, nullptr, nullptr, nullptr};
}  // namespace

PyMODINIT_FUNC PyInit__replica_host() { return PyModule_Create(&module); }
