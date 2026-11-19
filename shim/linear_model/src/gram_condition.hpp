#pragma once

#include <cstdint>
#include <raft/core/handle.hpp>

#include "glm_staging.hpp"

namespace cumlrkt::glm {

template <typename T>
double equilibrated_condition(const raft::handle_t& handle, const T* x,
                              const T* w, int64_t n, int64_t d, T alpha,
                              bool fit_intercept);

template <typename T>
bool checks_conditioning(glm_solver solver, bool alpha_window);

template <typename T>
void require_conditioned(double kappa, glm_solver solver);

}  // namespace cumlrkt::glm
