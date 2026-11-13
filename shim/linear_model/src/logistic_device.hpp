#pragma once

#include <cstdint>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <vector>

#include "cumlrkt/support.hpp"
#include "glm_staging.hpp"

namespace cumlrkt::glm::qn {

inline constexpr double constant_spread = 16;

struct column_stats {
  std::vector<double> sigma;
  std::vector<double> mean;
  std::vector<double> magnitude;
  std::vector<int64_t> kept;
  double largest;
};

struct residual_sums {
  std::vector<double> gradient;
  std::vector<double> intercept;
  double loss;
};

struct qn_shape {
  int64_t m;
  int64_t p;
  int64_t k;
  int64_t classes;
  bool fit_intercept;
};

template <typename L>
int64_t first_non_integral(const L* y, int64_t n, rmm::cuda_stream_view stream);

template <typename L>
void encode_against(const raft::handle_t& handle, const L* y, int64_t n,
                    const L* classes, int64_t k, int32_t* codes);

template <typename T>
buffer<double> effective_weights(const raft::handle_t& handle, const T* w,
                                 const int32_t* codes,
                                 const double* class_weight, int64_t n);

std::vector<double> class_totals(rmm::cuda_stream_view stream,
                                 const double* weights, const int32_t* codes,
                                 int64_t n, int64_t k);

int64_t positive_weights(const double* weights, int64_t n, int64_t* index,
                         rmm::cuda_stream_view stream);

bool equal_weights(const double* weights, row_selection rows,
                   rmm::cuda_stream_view stream);

double weight_sum(const double* weights, row_selection rows,
                  rmm::cuda_stream_view stream);

template <typename T>
column_stats column_stats_of(const matrix<T, int64_t>& x, const double* weights,
                             row_selection rows, bool fit_intercept,
                             rmm::cuda_stream_view stream);

template <typename T>
void stage_rows(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                row_selection rows, const int64_t* kept, int64_t p, int shift,
                T* out);

template <typename T>
void staged_targets(const raft::handle_t& handle, const int32_t* codes,
                    const double* weights, double scale, row_selection rows,
                    T* y, T* w);

template <typename T>
void params_to_coef(const raft::handle_t& handle, const T* params,
                    const int64_t* slot, int64_t d, int64_t k, int shift,
                    bool fit_intercept, int64_t p, T* coef, T* intercept);

template <typename T>
void coef_to_params(const raft::handle_t& handle, const T* coef,
                    const T* intercept, int64_t d, int64_t k, T* params);

template <typename T>
residual_sums residuals_of(const raft::handle_t& handle, const T* x, const T* y,
                           const T* w, const T* params, qn_shape shape);

template <typename T>
void scores_to(const raft::handle_t& handle, const T* scores, int64_t n,
               int64_t k, int32_t mode, T* out);

template <typename T, typename L>
void labels_of(const raft::handle_t& handle, const T* scores, int64_t n,
               int64_t k, const L* classes, L* out);

}  // namespace cumlrkt::glm::qn
