#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <string>
#include <vector>

#include "cumlrkt/scale.hpp"
#include "cumlrkt/support.hpp"

namespace cumlrkt::glm {

struct staging {
  std::vector<int> column_exponent;
  int y_exponent;
  std::vector<int64_t> kept;
};

struct ols_shape {
  int64_t n;
  int64_t d;
};

struct row_selection {
  const int64_t* index;
  int64_t count;
};

int even_floor(int x);

template <typename T>
int spread_exponent(int largest, double spread, int64_t n) {
  const int reach = std::numeric_limits<T>::max_exponent - 2 -
                    std::bit_width(static_cast<uint64_t>(n));
  const int e = spread > 0 ? largest + unit_exponent(spread) : largest;
  return std::min(e, largest + reach);
}

int64_t ols_workspace(int algorithm, ols_shape shape, int64_t cusolver_lwork);

std::string algorithm_name(int algorithm);

template <typename T>
buffer<T> staged_weights(const raft::handle_t& handle, const T* w, int64_t n);

template <typename T>
staging plan_staging(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                     const T* y, const T* w, row_selection rows,
                     bool fit_intercept);

staging uniform(staging s);

template <typename T>
std::vector<char> constant_columns(const raft::handle_t& handle,
                                   const matrix<T, int64_t>& x, const T* w,
                                   bool fit_intercept);

template <typename T>
bool constant_vector(const raft::handle_t& handle, const T* v, int64_t n,
                     const T* w, bool fit_intercept);

template <typename T>
int64_t first_positive(const T* w, int64_t n, rmm::cuda_stream_view stream);

template <typename T>
int64_t positive_count(const T* w, int64_t n, rmm::cuda_stream_view stream);

template <typename T>
void positive_rows(const T* w, int64_t n, int64_t* rows,
                   rmm::cuda_stream_view stream);

template <typename T>
void stage_columns(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                   const T* w, const staging& s, row_selection rows, T* out);

template <typename T>
void stage_targets(const raft::handle_t& handle, const T* y, const T* w,
                   const staging& s, row_selection rows, T* staged_y,
                   T* staged_w);

template <typename T>
double staged_mean(const raft::handle_t& handle, const T* y, const T* w,
                   int64_t n);

template <typename T>
int64_t unstage_coef(const raft::handle_t& handle, const T* staged,
                     const staging& s, int64_t d, T* coef);

template <typename T>
void require_finite_solution(const T* staged, int64_t d, T intercept,
                             int algorithm, rmm::cuda_stream_view stream);

template <typename T>
double residual_cosine(const raft::handle_t& handle,
                       const matrix<T, int64_t>& x, const staging& s,
                       const T* y, const T* w, const T* coef,
                       bool fit_intercept);

}  // namespace cumlrkt::glm
