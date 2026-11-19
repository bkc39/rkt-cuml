#pragma once

#include <cstdint>
#include <raft/core/handle.hpp>

#include "glm_staging.hpp"
#include "ritz.hpp"

namespace cumlrkt::glm {

struct condition_estimate {
  double kappa;
  ritz::gram_path path;
  int64_t window;
};

template <typename T>
condition_estimate equilibrated_condition(const raft::handle_t& handle,
                                          const T* x, const T* w, int64_t n,
                                          int64_t d, T alpha,
                                          bool fit_intercept, double limit,
                                          int64_t chunk = ritz::pass_chunk);

template <typename T>
double kappa_limit(glm_solver solver);

template <typename T>
bool checks_conditioning(glm_solver solver, bool alpha_window);

template <typename T>
void require_conditioned(double kappa, glm_solver solver);

}  // namespace cumlrkt::glm
