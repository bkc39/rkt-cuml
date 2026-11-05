#pragma once

#include <cstdint>
#include <type_traits>

#include "glm_staging.hpp"

namespace cumlrkt::glm {

template <typename T>
constexpr int ridge_headroom = std::is_same_v<T, float> ? 16 : 64;

template <typename T>
struct ridge_stage {
  staging s;
  T alpha;
};

struct ridge_penalty {
  double alpha;
  int weight_power;
};

struct ridge_codes {
  int32_t solver;
  int32_t fit_intercept;
  double alpha;
};

template <typename T>
ridge_stage<T> ridge_staging(staging s, int64_t n, ridge_penalty penalty);

int ridge_algorithm(int solver, int64_t rows, int64_t kept);

int64_t ridge_workspace(int algorithm, ols_shape shape, int64_t lwork);

void require_ridge_codes(ridge_codes codes);

void require_ridge_workspace(int algorithm, ols_shape shape, int64_t lwork);

template <typename T>
void require_ridge_solution(double cosine, int algorithm);

}  // namespace cumlrkt::glm
