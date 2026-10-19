#pragma once

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <string>
#include <utility>

#include "cumlrkt/common.hpp"
#include "raftrkt/array.h"
#include "raftrkt/error.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt::cluster {

template <typename T, typename Index = int>
struct matrix {
  T* data;
  Index n;
  Index d;
};

template <typename T, typename Index = int>
matrix<T, Index> rows_of(const rr_view* view, const char* name, int64_t rows,
                         int64_t cols) {
  rr_view v = *raftrkt::require(view, name);
  if (v.rank == 2 && (v.shape[0] == 0 || v.shape[1] == 0)) {
    v.strides[0] = 0;
    v.strides[1] = 0;
  }
  T* data = raftrkt::matrix_data<T, Index>(&v, name, raftrkt::layout::row_major,
                                           rows, cols);
  return {data, static_cast<Index>(v.shape[0]), static_cast<Index>(v.shape[1])};
}

inline void require_at_least(const std::string& what, int64_t got,
                             int64_t least) {
  if (got < least) {
    throw logic_error(what + ": expected at least " + std::to_string(least) +
                      ", got " + std::to_string(got));
  }
}

inline std::string printed(double got) {
  if (std::isnan(got)) {
    return "nan";
  }
  if (got == 0) {
    return "0";
  }
  return std::format("{}", got);
}

inline void require_positive(const char* what, double got, bool positive) {
  if (!positive) {
    throw logic_error(
        std::format("{}: expected greater than 0, got {}", what, printed(got)));
  }
}

inline void require_positive(const char* what, double got) {
  require_positive(what, got, got > 0);
}

inline void require_non_negative(const char* what, double got) {
  if (!(got >= 0)) {
    throw logic_error(
        std::format("{}: expected at least 0, got {}", what, printed(got)));
  }
}

inline void settle(const raft::handle_t& handle) {
  raftrkt::cuda_check(
      cudaStreamSynchronize(raft::resource::get_cuda_stream(handle).value()),
      "cudaStreamSynchronize");
}

inline void require_some(const char* name, const char* axis, int64_t extent) {
  if (extent == 0) {
    throw logic_error(std::string(name) + ": expected at least 1 " + axis +
                      ", got 0");
  }
}

template <typename Fn>
void on_device(
    void* handle,
    std::initializer_list<std::pair<const char*, const rr_view*>> views,
    const rr_view* first, const char* first_name, Fn&& fn) {
  const raft::handle_t& h = raftrkt::handle_of(handle);
  const int device = raftrkt::device_of(h);
  const raftrkt::device_scope scope{device};
  raftrkt::require_on_device(device, views);
  raftrkt::sync_guard sync{raft::resource::get_cuda_stream(h)};
  cumlrkt::on_floats(first, first_name,
                     [&](auto tag) { fn(h, decltype(tag){}); });
  sync.finish();
}

}  // namespace cumlrkt::cluster
