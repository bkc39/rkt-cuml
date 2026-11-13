#include <cuml/linear_model/qn.h>

#include <cstdint>
#include <cuml/linear_model/glm.hpp>
#include <memory>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <utility>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "logistic_device.hpp"
#include "logistic_fit.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt::glm::qn {

namespace {

template <typename T>
buffer<T> decision_of(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                      const matrix<T, int64_t>& coef, const T* intercept) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const int64_t kk = coef.n;
  const int64_t classes = kk == 1 ? 2 : kk;
  buffer<T> params = scratch<T>((x.d + 1) * kk, stream);
  coef_to_params<T>(handle, coef.data, intercept, x.d, kk, params.data());
  buffer<T> scores = scratch<T>(x.n * kk, stream);
  ML::GLM::qn_params q;
  q.loss = classes == 2 ? ML::GLM::QN_LOSS_LOGISTIC : ML::GLM::QN_LOSS_SOFTMAX;
  q.fit_intercept = true;
  q.verbose = quiet_level;
  if (x.n > 0) {
    ML::GLM::qnDecisionFunction<T, int>(
        handle, q, x.data, false, static_cast<int>(x.n), static_cast<int>(x.d),
        static_cast<int>(classes), params.data(), scores.data());
  }
  settle(handle);
  return scores;
}

template <typename V>
void deposit_one(const raft::handle_t& handle, rr_deposit* out,
                 std::unique_ptr<buffer<V>> b, rr_view region) {
  settle(handle);
  raftrkt::deposit(out, "out", handle, std::move(b), {region});
}

struct apply_views {
  const rr_view* x;
  const rr_view* coef;
  const rr_view* intercept;
  const rr_view* classes;
};

template <typename T>
void apply_into(const raft::handle_t& handle, const apply_views& views,
                int32_t mode, rr_deposit* out) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const matrix<T, int64_t> coef = rows_of<T, int64_t>(
      views.coef, "coef", raftrkt::any_extent, raftrkt::any_extent);
  const matrix<T, int64_t> x =
      rows_of<T, int64_t>(views.x, "X", raftrkt::any_extent, coef.d);
  const T* intercept =
      raftrkt::vector_data<T, int32_t>(views.intercept, "intercept", coef.n);
  require_finite("X", x, stream);
  buffer<T> scores = decision_of<T>(handle, x, coef, intercept);
  const int64_t kk = coef.n;
  const int64_t classes = kk == 1 ? 2 : kk;
  if (mode == CUML_LINEAR_MODEL_APPLY_DECISION) {
    auto owned = std::make_unique<buffer<T>>(std::move(scores));
    const rr_view region =
        raftrkt::region(owned->data(), x.n, kk, raftrkt::layout::row_major);
    deposit_one(handle, out, std::move(owned), region);
    return;
  }
  if (mode == CUML_LINEAR_MODEL_APPLY_LABELS && views.classes == nullptr) {
    auto codes =
        std::make_unique<buffer<int32_t>>(scratch<int32_t>(x.n, stream));
    labels_of<T, int32_t>(handle, scores.data(), x.n, kk, nullptr,
                          codes->data());
    const rr_view region = raftrkt::region(*codes);
    deposit_one(handle, out, std::move(codes), region);
    return;
  }
  if (mode == CUML_LINEAR_MODEL_APPLY_LABELS) {
    on_class_labels(views.classes, [&](auto tag) {
      using L = decltype(tag);
      const L* values =
          raftrkt::vector_data<L, int32_t>(views.classes, "classes", classes);
      auto labels = std::make_unique<buffer<L>>(scratch<L>(x.n, stream));
      labels_of<T, L>(handle, scores.data(), x.n, kk, values, labels->data());
      const rr_view region = raftrkt::region(*labels);
      deposit_one(handle, out, std::move(labels), region);
    });
    return;
  }
  auto probs = std::make_unique<buffer<T>>(scratch<T>(x.n * classes, stream));
  scores_to<T>(handle, scores.data(), x.n, kk, mode, probs->data());
  const rr_view region =
      raftrkt::region(probs->data(), x.n, classes, raftrkt::layout::row_major);
  deposit_one(handle, out, std::move(probs), region);
}

template <typename L>
void encode_into(const raft::handle_t& handle, const rr_view* yv,
                 const rr_view* classesv, rr_deposit* out) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const L* y = raftrkt::vector_data<L, int32_t>(yv, "y", raftrkt::any_extent);
  const int64_t n = raftrkt::require(yv, "y")->shape[0];
  const L* classes = raftrkt::vector_data<L, int32_t>(classesv, "classes",
                                                      raftrkt::any_extent);
  const int64_t k = raftrkt::require(classesv, "classes")->shape[0];
  auto codes = std::make_unique<buffer<int32_t>>(scratch<int32_t>(n, stream));
  encode_against<L>(handle, y, n, classes, k, codes->data());
  const rr_view region = raftrkt::region(*codes);
  deposit_one(handle, out, std::move(codes), region);
}

}  // namespace

}  // namespace cumlrkt::glm::qn

extern "C" {

int cuml_linear_model_logistic_regression_fit(
    void* handle, const rr_view* x, const rr_view* y,
    const rr_view* sample_weight, const rr_view* class_keys,
    const rr_view* class_values, rr_deposit* out, int32_t penalty, double C,
    double l1_ratio, double tol, int32_t max_iter, int32_t linesearch_max_iter,
    int32_t lbfgs_memory, int32_t fit_intercept, int32_t penalty_normalized,
    int32_t class_weight_mode, int32_t* n_iter) {
  return cumlrkt::guarded(
      handle,
      {{"X", x},
       {"y", y},
       {"sample-weight", sample_weight},
       {"class-weight", class_keys},
       {"class-weight", class_values}},
      [&](const raft::handle_t& h) {
        rr_deposit* const into = raftrkt::require(out, "out");
        int32_t* const iterations = raftrkt::require(n_iter, "n_iter");
        const cumlrkt::glm::qn::qn_request request{x,
                                                   y,
                                                   sample_weight,
                                                   class_keys,
                                                   class_values,
                                                   penalty,
                                                   C,
                                                   l1_ratio,
                                                   tol,
                                                   max_iter,
                                                   linesearch_max_iter,
                                                   lbfgs_memory,
                                                   fit_intercept,
                                                   penalty_normalized,
                                                   class_weight_mode};
        cumlrkt::glm::qn::require_qn_codes(request);
        *iterations = cumlrkt::glm::qn::logistic_fit(h, request, into);
      });
}

CUMLRKT_JOB(cuml_linear_model_logistic_regression_fit)

int cuml_linear_model_logistic_regression_apply(void* handle, const rr_view* x,
                                                const rr_view* coef,
                                                const rr_view* intercept,
                                                const rr_view* classes,
                                                int32_t mode, rr_deposit* out) {
  return cumlrkt::guarded(
      handle,
      {{"X", x},
       {"coef", coef},
       {"intercept", intercept},
       {"classes", classes}},
      [&](const raft::handle_t& h) {
        rr_deposit* const into = raftrkt::require(out, "out");
        if (mode < CUML_LINEAR_MODEL_APPLY_DECISION ||
            mode > CUML_LINEAR_MODEL_APPLY_LOG_PROBA) {
          throw cumlrkt::logic_error("mode: expected 0 to 3");
        }
        cumlrkt::on_floats(x, "X", [&](auto tag) {
          cumlrkt::glm::qn::apply_into<decltype(tag)>(
              h, {x, coef, intercept, classes}, mode, into);
        });
      });
}

CUMLRKT_JOB(cuml_linear_model_logistic_regression_apply)

int cuml_linear_model_label_encode(void* handle, const rr_view* y,
                                   const rr_view* classes, rr_deposit* out) {
  return cumlrkt::guarded(
      handle, {{"y", y}, {"classes", classes}}, [&](const raft::handle_t& h) {
        rr_deposit* const into = raftrkt::require(out, "out");
        cumlrkt::glm::qn::on_class_labels(y, [&](auto tag) {
          cumlrkt::glm::qn::encode_into<decltype(tag)>(h, y, classes, into);
        });
      });
}

CUMLRKT_JOB(cuml_linear_model_label_encode)
}
