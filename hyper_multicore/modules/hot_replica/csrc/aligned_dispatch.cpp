// Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
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

}  // namespace ReplicaHost
