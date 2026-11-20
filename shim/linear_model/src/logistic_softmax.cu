#include <cuml/linear_model/qn.h>

#include <cstddef>
#include <glm/qn/qn.cuh>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include "logistic_softmax.hpp"
#include "weighted_softmax.cuh"

namespace cumlrkt::glm::logistic {

namespace {

namespace qn = ML::GLM::detail;

template <typename T>
ML::SimpleDenseMat<T> rows_of(T* x, softmax_problem s) {
  return {x, static_cast<int>(s.m), static_cast<int>(s.p), ML::ROW_MAJOR};
}

template <typename T>
rmm::device_uvector<T> scores_for(const raft::handle_t& handle,
                                  softmax_problem s) {
  return rmm::device_uvector<T>(static_cast<std::size_t>(s.k * s.m),
                                handle.get_stream());
}

template <typename T, typename Loss>
T loss_grad_of(const raft::handle_t& handle, Loss& loss, T* x, T* y, T* params,
               softmax_problem s, T* grad) {
  const cudaStream_t stream = handle.get_stream();
  const int dims = static_cast<int>(s.p) + (s.fit_intercept ? 1 : 0);
  const auto k = static_cast<int>(s.k);
  rmm::device_uvector<T> z = scores_for<T>(handle, s);
  rmm::device_uvector<T> scalar(1, stream);
  ML::SimpleDenseMat<T> scores(z.data(), k, static_cast<int>(s.m));
  ML::SimpleDenseMat<T> w(params, k, dims);
  ML::SimpleDenseMat<T> g(grad, k, dims);
  const ML::SimpleDenseMat<T> rows = rows_of(x, s);
  const ML::SimpleVec<T> labels(y, static_cast<int>(s.m));
  loss.loss_grad(scalar.data(), g, w, rows, labels, scores, stream);
  T out{0};
  raft::update_host(&out, scalar.data(), 1, stream);
  raft::interruptible::synchronize(stream);
  return out;
}

}  // namespace

template <typename T>
void weighted_softmax_fit(const raft::handle_t& handle,
                          const ML::GLM::qn_params& params, T* x, T* y, T* w,
                          softmax_problem shape, T* coef, T* objective,
                          int* n_iter) {
  weighted_softmax<T> loss(handle, static_cast<int>(shape.p),
                           static_cast<int>(shape.k), shape.fit_intercept, w);
  rmm::device_uvector<T> z = scores_for<T>(handle, shape);
  ML::SimpleDenseMat<T> scores(z.data(), static_cast<int>(shape.k),
                               static_cast<int>(shape.m));
  const ML::SimpleDenseMat<T> rows = rows_of(x, shape);
  const ML::SimpleVec<T> labels(y, static_cast<int>(shape.m));
  qn::qn_fit<T, weighted_softmax<T>>(handle, params, loss, rows, labels, scores,
                                     coef, objective, n_iter);
}

template <typename T>
T softmax_loss_grad(const raft::handle_t& handle, T* x, T* y, T* w, T* params,
                    softmax_problem shape, T* grad) {
  const auto p = static_cast<int>(shape.p);
  const auto k = static_cast<int>(shape.k);
  if (w == nullptr) {
    qn::Softmax<T> loss(handle, p, k, shape.fit_intercept);
    return loss_grad_of<T>(handle, loss, x, y, params, shape, grad);
  }
  weighted_softmax<T> loss(handle, p, k, shape.fit_intercept, w);
  return loss_grad_of<T>(handle, loss, x, y, params, shape, grad);
}

#define CUMLRKT_LOGISTIC_SOFTMAX(T)                                            \
  template void weighted_softmax_fit<T>(const raft::handle_t&,                 \
                                        const ML::GLM::qn_params&, T*, T*, T*, \
                                        softmax_problem, T*, T*, int*);        \
  template T softmax_loss_grad<T>(const raft::handle_t&, T*, T*, T*, T*,       \
                                  softmax_problem, T*);

CUMLRKT_LOGISTIC_SOFTMAX(float)
CUMLRKT_LOGISTIC_SOFTMAX(double)

}  // namespace cumlrkt::glm::logistic
