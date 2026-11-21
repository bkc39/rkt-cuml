#pragma once

#include <cuml/linear_model/qn.h>

#include <cstdint>
#include <raft/core/handle.hpp>

namespace cumlrkt::glm::logistic {

struct softmax_problem {
  int64_t m;
  int64_t p;
  int64_t k;
  bool fit_intercept;
};

template <typename T>
void weighted_softmax_fit(const raft::handle_t& handle,
                          const ML::GLM::qn_params& params, T* x, T* y, T* w,
                          softmax_problem shape, T* coef, T* objective,
                          int* n_iter);

template <typename T>
T softmax_loss_grad(const raft::handle_t& handle, T* x, T* y, T* w, T* params,
                    softmax_problem shape, T* grad);

}  // namespace cumlrkt::glm::logistic
