#pragma once

#include <cstdint>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <vector>

#include "cumlrkt/support.hpp"
#include "glm_staging.hpp"

namespace cumlrkt::glm {

struct moment {
  double square;
  double cross;
};

struct residual_terms {
  std::vector<moment> moments;
  double y_norm;
};

struct staged_columns {
  const int64_t* kept;
  const int* exponent;
  int64_t count;
};

template <typename T>
std::vector<T> to_host(const buffer<T>& b, rmm::cuda_stream_view stream) {
  std::vector<T> host(b.size());
  raftrkt::cuda_check(
      cudaMemcpyAsync(host.data(), b.data(), b.size() * sizeof(T),
                      cudaMemcpyDeviceToHost, stream.value()),
      "cudaMemcpyAsync");
  raftrkt::cuda_check(cudaStreamSynchronize(stream.value()),
                      "cudaStreamSynchronize");
  return host;
}

template <typename T>
buffer<T> to_device(const std::vector<T>& host, rmm::cuda_stream_view stream) {
  buffer<T> b = scratch<T>(static_cast<int64_t>(host.size()), stream);
  if (!host.empty()) {
    raftrkt::cuda_check(
        cudaMemcpyAsync(b.data(), host.data(), host.size() * sizeof(T),
                        cudaMemcpyHostToDevice, stream.value()),
        "cudaMemcpyAsync");
  }
  return b;
}

template <typename T>
std::vector<double> column_largest(const matrix<T, int64_t>& x, const T* w,
                                   rmm::cuda_stream_view stream);

template <typename T>
double largest_counted(const T* v, int64_t n, const T* w,
                       rmm::cuda_stream_view stream);

template <typename T>
void stage_matrix(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                  const T* w, row_selection rows, staged_columns columns,
                  T* out);

template <typename T>
void stage_elements(const raft::handle_t& handle, const T* v, const T* w,
                    row_selection rows, int exponent, T* out);

template <typename T>
void unstage(const raft::handle_t& handle, const T* staged, const int64_t* slot,
             const int* shift, int64_t d, T* coef);

template <typename T>
residual_terms residual_moments(const raft::handle_t& handle,
                                const matrix<T, int64_t>& x, const T* y,
                                const T* w, staged_columns columns,
                                const T* coef, double intercept,
                                bool fit_intercept);

}  // namespace cumlrkt::glm
