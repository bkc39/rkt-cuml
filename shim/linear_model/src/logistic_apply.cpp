#include <cuml/linear_model/qn.h>

#include <cstdint>
#include <cuml/linear_model/glm.hpp>
#include <memory>
#include <optional>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <utility>

#include "cumlrkt/common.hpp"
#include "cumlrkt/labels.hpp"
#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "logistic_checks.hpp"
#include "logistic_device.hpp"
#include "logistic_fit.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt::glm::logistic {

namespace {

template <typename T>
buffer<T> decision_of(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                      const matrix<T, int64_t>& coef, const T* intercept) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const int64_t outputs = coef.n;
  const int64_t classes = classes_of(outputs);
  buffer<T> params = scratch<T>((x.d + 1) * outputs, stream);
  coef_to_params<T>(handle, coef.data, intercept, x.d, outputs, params.data());
  buffer<T> scores = scratch<T>(x.n * outputs, stream);
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
void deposit_rows(const raft::handle_t& handle, rr_deposit* out, buffer<V> b,
                  int64_t rows, std::optional<int64_t> columns) {
  auto owned = std::make_unique<buffer<V>>(std::move(b));
  const rr_view region = columns
                             ? raftrkt::region(owned->data(), rows, *columns,
                                               raftrkt::layout::row_major)
                             : raftrkt::region(*owned);
  settle(handle);
  raftrkt::deposit(out, "out", handle, std::move(owned), {region});
}

struct apply_views {
  const rr_view* x;
  const rr_view* coef;
  const rr_view* intercept;
  const rr_view* classes;
};

template <typename T>
void codes_into(const raft::handle_t& handle, const buffer<T>& scores,
                int64_t n, int64_t outputs, rr_deposit* out) {
  buffer<int32_t> codes = scratch<int32_t>(n, stream_of(handle));
  codes_of<T>(handle, scores.data(), n, outputs, codes.data());
  deposit_rows(handle, out, std::move(codes), n, std::nullopt);
}

template <typename T>
void labels_into(const raft::handle_t& handle, const rr_view* classes_view,
                 const buffer<T>& scores, int64_t n, int64_t outputs,
                 rr_deposit* out) {
  on_class_labels(classes_view, "classes", [&](auto tag) {
    using L = decltype(tag);
    const L* classes = raftrkt::vector_data<L, int32_t>(classes_view, "classes",
                                                        classes_of(outputs));
    buffer<L> labels = scratch<L>(n, stream_of(handle));
    labels_of<T, L>(handle, scores.data(), n, outputs, classes, labels.data());
    deposit_rows(handle, out, std::move(labels), n, std::nullopt);
  });
}

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
  const int64_t outputs = coef.n;
  const int64_t classes = classes_of(outputs);
  switch (mode) {
    case CUML_LINEAR_MODEL_APPLY_DECISION:
      deposit_rows(handle, out, std::move(scores), x.n, outputs);
      break;
    case CUML_LINEAR_MODEL_APPLY_LABELS:
      if (views.classes == nullptr) {
        codes_into<T>(handle, scores, x.n, outputs, out);
      } else {
        labels_into<T>(handle, views.classes, scores, x.n, outputs, out);
      }
      break;
    case CUML_LINEAR_MODEL_APPLY_PROBA:
    case CUML_LINEAR_MODEL_APPLY_LOG_PROBA:
    default: {
      buffer<T> probs = scratch<T>(x.n * classes, stream);
      scores_to<T>(handle, scores.data(), x.n, outputs,
                   mode == CUML_LINEAR_MODEL_APPLY_LOG_PROBA, probs.data());
      deposit_rows(handle, out, std::move(probs), x.n, classes);
    }
  }
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
  buffer<int32_t> codes = scratch<int32_t>(n, stream);
  encode_against<L>(y, n, classes, k, codes.data(), stream);
  deposit_rows(handle, out, std::move(codes), n, std::nullopt);
}

}  // namespace

}  // namespace cumlrkt::glm::logistic

extern "C" {

int cuml_linear_model_logistic_regression_fit(
    void* handle, const rr_view* x, const rr_view* y,
    const rr_view* sample_weight, const rr_view* class_weight, rr_deposit* out,
    int32_t penalty, double C, double l1_ratio, double tol, int32_t max_iter,
    int32_t linesearch_max_iter, int32_t lbfgs_memory, int32_t fit_intercept,
    int32_t penalty_normalized, int32_t class_weight_mode, int32_t* n_iter) {
  return cumlrkt::guarded(
      handle,
      {{"X", x},
       {"y", y},
       {"sample-weight", sample_weight},
       {"class-weight", class_weight}},
      [&](const raft::handle_t& h) {
        rr_deposit* const into = raftrkt::require(out, "out");
        int32_t* const iterations = raftrkt::require(n_iter, "n_iter");
        namespace logistic = cumlrkt::glm::logistic;
        logistic::require_fit_codes({.penalty = penalty,
                                     .fit_intercept = fit_intercept,
                                     .normalized = penalty_normalized,
                                     .class_mode = class_weight_mode});
        const logistic::fit_options options{
            .penalty = penalty,
            .C = C,
            .l1_ratio = l1_ratio,
            .tol = tol,
            .max_iter = max_iter,
            .linesearch_max_iter = linesearch_max_iter,
            .lbfgs_memory = lbfgs_memory,
            .fit_intercept = fit_intercept != 0,
            .normalized = penalty_normalized != 0,
            .class_mode = class_weight_mode};
        logistic::require_fit_bounds(options);
        *iterations = logistic::fit(h,
                                    {.x = x,
                                     .y = y,
                                     .sample_weight = sample_weight,
                                     .class_weight = class_weight},
                                    options, into);
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
          cumlrkt::glm::logistic::apply_into<decltype(tag)>(
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
        cumlrkt::on_class_labels(y, "y", [&](auto tag) {
          cumlrkt::glm::logistic::encode_into<decltype(tag)>(h, y, classes,
                                                             into);
        });
      });
}

CUMLRKT_JOB(cuml_linear_model_label_encode)
}
