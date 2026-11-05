#pragma once

#include <cstdint>
#include <type_traits>

#include "fit_checks.hpp"
#include "glm_staging.hpp"

namespace cumlrkt::glm {

template <typename T>
constexpr int ridge_headroom = std::is_same_v<T, float> ? 16 : 64;

template <typename T>
constexpr int ridge_floor = ridge_headroom<T> / 2;

struct ridge_request {
  fit_request fit;
  double alpha;
};

struct ridge_penalty {
  double alpha;
  int weight_power;
};

template <typename T>
struct ridge_scale {
  int shared;
  T alpha;
  bool negligible;
};

template <typename T>
ridge_scale<T> ridge_scale_of(const staging& s, int64_t n,
                              ridge_penalty penalty);

staging shared_at(staging s, int shared);

int ridge_algorithm(int solver, int64_t rows, int64_t kept);

int lstsq_algorithm(int solver);

void require_ridge_codes(const ridge_request& request);

}  // namespace cumlrkt::glm
