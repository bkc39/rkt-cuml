#pragma once

#include <cstdint>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <string>
#include <vector>

#include "cumlrkt/support.hpp"

namespace cumlrkt::glm {

enum class exponents { per_column, uniform };

struct staging {
  std::vector<int> e;
  int f;
  int g;
  std::vector<int64_t> kept;
};

int even_floor(int x);

int64_t ols_workspace(int algorithm, int64_t n, int64_t d,
                      int64_t cusolver_lwork);

std::string algorithm_name(int algorithm);

template <typename T>
staging plan_staging(const raft::handle_t& handle,
                     const matrix<T, int64_t>& x, const T* y, const T* w,
                     bool fit_intercept, exponents mode);

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
void stage_columns(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                   const T* w, const staging& s, T* out);

template <typename T>
void stage_vector(const raft::handle_t& handle, const T* v, int64_t n,
                  const T* w, int exponent, T* out);

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
                       const T* y, const T* w, const T* coef, T intercept,
                       bool fit_intercept);

template <typename T>
bool all_zero(const T* v, int64_t n, rmm::cuda_stream_view stream);

}  // namespace cumlrkt::glm
