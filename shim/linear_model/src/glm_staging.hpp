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
  std::vector<double> largest;
  std::vector<double> normalised_mean;
  std::vector<double> normalised_remainder;
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
buffer<T> staged_weights(const raft::handle_t& handle, const T* w, int64_t n,
                         double largest);

template <typename T>
staging plan_staging(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                     const T* y, const T* w, row_selection rows,
                     bool fit_intercept);

staging uniform(staging s);

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
struct staged_problem {
  buffer<T> x;
  buffer<T> y;
  buffer<T> w;
};

template <typename T>
staged_problem<T> stage_problem(const raft::handle_t& handle,
                                const matrix<T, int64_t>& x, const T* y,
                                const T* w, const staging& s,
                                row_selection rows);

template <typename T>
int64_t unstage_coef(const raft::handle_t& handle, const T* staged,
                     const staging& s, int64_t d, T* coef);

template <typename T>
void require_finite_solution(const T* staged, int64_t d, T intercept,
                             int algorithm, rmm::cuda_stream_view stream);

template <typename T>
void require_least_squares(double cosine, int algorithm);

template <typename T>
void require_in_range(int64_t column, T intercept);

template <typename T>
double residual_cosine(const raft::handle_t& handle,
                       const matrix<T, int64_t>& x, const T* y, const T* w,
                       row_selection rows, const staging& s, const T* coef,
                       bool fit_intercept);

}  // namespace cumlrkt::glm
