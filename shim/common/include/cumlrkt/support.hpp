#pragma once

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <string>
#include <type_traits>
#include <utility>

#include "cumlrkt/common.hpp"
#include "cumlrkt/limits.hpp"
#include "cumlrkt/rows.hpp"
#include "raftrkt/array.h"
#include "raftrkt/error.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt {

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

template <typename T = double>
std::string printed(std::type_identity_t<T> got) {
  if (std::isnan(got)) {
    return "nan";
  }
  if (got == 0) {
    return "0";
  }
  return std::format("{}", got);
}

template <typename T = double>
void require_positive(const char* what, double got) {
  if (!(static_cast<T>(got) > 0)) {
    throw logic_error(
        std::format("{}: expected greater than 0, got {}", what, printed(got)));
  }
}

inline void require_non_negative(const char* what, double got) {
  if (!(got >= 0)) {
    throw logic_error(
        std::format("{}: expected at least 0, got {}", what, printed(got)));
  }
}

inline void require_finite_non_negative(const char* what, double got) {
  if (!std::isfinite(got) || got < 0) {
    throw logic_error(
        std::format("{}: expected a finite value of at least 0, got {}", what,
                    printed(got)));
  }
}

inline void require_at_most(const char* what, int64_t got, int64_t most) {
  if (got > most) {
    throw logic_error(
        std::format("{}: expected at most {}, got {}", what, most, got));
  }
}

inline void require_float(const char* what, double got) {
  if (got > most_float) {
    throw logic_error(std::format("{}: expected at most {}, got {}", what,
                                  printed(most_float), printed(got)));
  }
}

inline void require_index(const char* name, int64_t rows, int64_t cols) {
  if (cols > 0 && rows > most_index / cols) {
    throw logic_error(std::format(
        "{}: {}x{} elements do not fit the index type", name, rows, cols));
  }
}

inline void settle(const raft::handle_t& handle) {
  raftrkt::cuda_check(
      cudaStreamSynchronize(raft::resource::get_cuda_stream(handle).value()),
      "cudaStreamSynchronize");
}

template <typename T>
T read_one(const T* data, int64_t i, rmm::cuda_stream_view stream) {
  T value{};
  raftrkt::cuda_check(cudaMemcpyAsync(&value, data + i, sizeof(T),
                                      cudaMemcpyDeviceToHost, stream.value()),
                      "cudaMemcpyAsync");
  raftrkt::cuda_check(cudaStreamSynchronize(stream.value()),
                      "cudaStreamSynchronize");
  return value;
}

template <typename T>
void require_bounded_rows(const char* name, const matrix<T, int64_t>& x,
                          rmm::cuda_stream_view stream) {
  const double most = squared_norm_bound<T>;
  const auto [row, squared] = first_large_row(x.data, x.n, x.d, most, stream);
  if (row >= 0) {
    throw logic_error(
        std::format("{}: row {}: expected a squared norm of at most {}, got {}",
                    name, row, printed(most), printed(squared)));
  }
}

template <typename T>
void require_finite(const char* name, const matrix<T, int64_t>& x,
                    rmm::cuda_stream_view stream) {
  const int64_t i = first_non_finite(x.data, x.n * x.d, stream);
  if (i >= 0) {
    const T got = read_one(x.data, i, stream);
    throw logic_error(
        std::format("{}: row {}, column {}: expected a finite value, got {}",
                    name, i / x.d, i % x.d, printed(static_cast<double>(got))));
  }
}

template <typename T>
void require_magnitudes(const char* name, const matrix<T, int64_t>& x,
                        double most, rmm::cuda_stream_view stream) {
  const int64_t i = first_large_element(x.data, x.n * x.d, most, stream);
  if (i >= 0) {
    const T got = read_one(x.data, i, stream);
    throw logic_error(std::format(
        "{}: row {}, column {}: expected a magnitude of at most {}, got {}",
        name, i / x.d, i % x.d, printed(most), printed<T>(got)));
  }
}

inline void require_some(const char* name, const char* axis, int64_t extent,
                         int64_t least = 1) {
  if (extent < least) {
    throw logic_error(std::format("{}: expected at least {} {}{}, got {}", name,
                                  least, axis, least == 1 ? "" : "s", extent));
  }
}

template <typename Run>
void in_device_scope(
    void* handle,
    std::initializer_list<std::pair<const char*, const rr_view*>> views,
    Run&& run) {
  const raft::handle_t& h = raftrkt::handle_of(handle);
  const int device = raftrkt::device_of(h);
  const raftrkt::device_scope scope{device};
  raftrkt::require_on_device(device, views);
  raftrkt::sync_guard sync{raft::resource::get_cuda_stream(h)};
  run(h);
  sync.finish();
}

template <typename Run>
int guarded(void* handle,
            std::initializer_list<std::pair<const char*, const rr_view*>> views,
            Run&& run) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [&] {
    in_device_scope(handle, views, std::forward<Run>(run));
  });
}

template <typename Fn>
void on_device(
    void* handle,
    std::initializer_list<std::pair<const char*, const rr_view*>> views,
    const rr_view* first, const char* first_name, Fn&& fn) {
  in_device_scope(handle, views, [&](const raft::handle_t& h) {
    cumlrkt::on_floats(first, first_name,
                       [&](auto tag) { fn(h, decltype(tag){}); });
  });
}

template <typename Fn>
void on_float32_device(
    void* handle,
    std::initializer_list<std::pair<const char*, const rr_view*>> views,
    const rr_view* first, const char* first_name, Fn&& fn) {
  in_device_scope(handle, views, [&](const raft::handle_t& h) {
    const rr_view& view = *raftrkt::require(first, first_name);
    if (view.dtype != RR_DTYPE_FLOAT32) {
      throw logic_error(std::string(first_name) + ": expected float32, got " +
                        raftrkt::dtype_name(view.dtype));
    }
    fn(h);
  });
}

}  // namespace cumlrkt

#define CUMLRKT_JOB(entry)                                                 \
  void entry##_job(void* args, size_t args_len, rr_job_outcome* outcome) { \
    raftrkt::run_job<&(entry)>(args, args_len, outcome,                    \
                               cumlrkt::last_error_slot());                \
  }
