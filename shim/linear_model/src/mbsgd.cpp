#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuml/solvers/solver.hpp>
#include <memory>
#include <optional>
#include <raft/core/handle.hpp>
#include <random>
#include <rmm/cuda_stream_view.hpp>
#include <string>
#include <utility>
#include <vector>

#include "cumlrkt/common.hpp"
#include "cumlrkt/labels.hpp"
#include "cumlrkt/linear_model.h"
#include "cumlrkt/rows.hpp"
#include "cumlrkt/support.hpp"
#include "cumlrkt/vectors.hpp"
#include "glm_staging.hpp"
#include "glm_staging_device.hpp"
#include "label_checks.hpp"
#include "mbsgd_checks.hpp"
#include "mbsgd_device.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt::glm::mbsgd {

namespace {

template <typename T>
struct sgd_fit {
  buffer<T> coef;
  buffer<T> intercept;
  int64_t d;
};

template <typename T, typename L>
struct classifier_outputs {
  buffer<T> coef;
  buffer<T> intercept;
  rmm::device_uvector<L> classes;
};

std::optional<buffer<int64_t>> row_order(const sgd_options& o, int64_t n,
                                         rmm::cuda_stream_view stream) {
  if (o.shuffle == 0 || o.seeded == 0) {
    return std::nullopt;
  }
  std::vector<int64_t> order(n);
  for (int64_t i = 0; i < n; ++i) {
    order[i] = i;
  }
  std::mt19937 g(o.seed);
  for (int64_t i = n - 1; i > 0; --i) {
    const auto j = static_cast<int64_t>(g() % static_cast<uint64_t>(i + 1));
    std::swap(order[i], order[j]);
  }
  return to_device(order, stream);
}

double penalty_of(const std::vector<double>& w, const sgd_options& o) {
  double l1 = 0;
  double l2 = 0;
  for (const double v : w) {
    l1 += std::abs(v);
    l2 += v * v;
  }
  switch (o.penalty) {
    case CUML_LINEAR_MODEL_PENALTY_L1:
      return o.alpha * l1;
    case CUML_LINEAR_MODEL_PENALTY_L2:
      return o.alpha * l2;
    case CUML_LINEAR_MODEL_PENALTY_ELASTICNET:
      return o.alpha * (((1 - o.l1_ratio) * l2) + (o.l1_ratio * l1));
    default:
      return 0;
  }
}

template <typename T>
void require_improved(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                      const T* y, const T* coef, T intercept,
                      const sgd_options& o, bool ones, bool regressor) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const int64_t cols = x.d + (ones ? 1 : 0);
  const std::vector<T> w = to_host(coef, cols, stream);
  const bool finite =
      std::ranges::all_of(w, [](T v) { return std::isfinite(v); }) &&
      std::isfinite(intercept);
  const objective_shape shape{x.n, x.d, ones, o.loss};
  double start = 0;
  double end = 0;
  double floor = 0;
  if (finite) {
    const bool centred =
        o.fit_intercept != 0 && o.loss == CUML_LINEAR_MODEL_SGD_SQUARED;
    const double b0 = centred ? mean_of<T>(handle, y, x.n) : 0.0;
    const buffer<T> zero = zeroed<T>(cols, stream);
    start = mean_loss<T>(handle, x.data, y, zero.data(), b0, shape);
    const double scale =
        regressor ? mean_loss<T>(handle, x.data, y, zero.data(), 0.0, shape)
                  : start;
    floor = gate_floor_share * scale;
    std::vector<double> wd(w.begin(), w.end());
    end = mean_loss<T>(handle, x.data, y, coef, static_cast<double>(intercept),
                       shape) +
          penalty_of(wd, o);
  }
  if (const auto refusal = gate_refusal(finite, start, end, floor, o.eta0)) {
    throw logic_error(*refusal);
  }
}

template <typename T>
sgd_fit<T> solved(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                  const T* targets, const sgd_options& o, bool ones,
                  bool regressor) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const int64_t n = x.n;
  const int64_t cols = x.d + (ones ? 1 : 0);
  const std::optional<buffer<int64_t>> order = row_order(o, n, stream);
  const int64_t* by = order ? order->data() : nullptr;
  buffer<T> xc = scratch<T>(n * cols, stream);
  columns_of<T>(handle, x.data, n, x.d, by, ones, xc.data());
  buffer<T> yc = scratch<T>(n, stream);
  reordered<T>(handle, targets, n, by, yc.data());
  buffer<T> coef = zeroed<T>(cols, stream);
  T intercept{0};
  const bool upstream_intercept = o.fit_intercept != 0 && !ones;
  ML::Solver::sgdFit(const_cast<raft::handle_t&>(handle), xc.data(),
                     static_cast<int>(n), static_cast<int>(cols), yc.data(),
                     coef.data(), &intercept, upstream_intercept,
                     static_cast<int>(std::min<int64_t>(o.batch_size, n)),
                     o.epochs, o.learning_rate, static_cast<T>(o.eta0),
                     static_cast<T>(o.power_t), o.loss, o.penalty,
                     static_cast<T>(o.alpha), static_cast<T>(o.l1_ratio),
                     o.shuffle != 0, static_cast<T>(o.tol), o.n_iter_no_change);
  settle(handle);
  require_improved<T>(handle, x, targets, coef.data(), intercept, o, ones,
                      regressor);
  buffer<T> b = scratch<T>(1, stream);
  if (ones) {
    raftrkt::cuda_check(
        cudaMemcpyAsync(b.data(), coef.data() + x.d, sizeof(T),
                        cudaMemcpyDeviceToDevice, stream.value()),
        "cudaMemcpyAsync");
  } else {
    const T host = upstream_intercept ? intercept : T{0};
    raftrkt::cuda_check(cudaMemcpyAsync(b.data(), &host, sizeof(T),
                                        cudaMemcpyHostToDevice, stream.value()),
                        "cudaMemcpyAsync");
    raftrkt::cuda_check(cudaStreamSynchronize(stream.value()),
                        "cudaStreamSynchronize");
  }
  settle(handle);
  return {std::move(coef), std::move(b), x.d};
}

template <typename T>
matrix<T, int64_t> checked_x(const rr_view* xv, const rr_view* yv,
                             const sgd_options& o, bool ones,
                             rmm::cuda_stream_view stream) {
  const matrix<T, int64_t> x =
      rows_of<T, int64_t>(xv, "X", raftrkt::any_extent, raftrkt::any_extent);
  require_sgd_options(o);
  require_sgd_shape(x.n, x.d, raftrkt::require(yv, "y")->shape[0]);
  require_elements(x.n, x.d + (ones ? 1 : 0));
  require_finite("X", x, stream);
  return x;
}

template <typename T>
void regress_into(const raft::handle_t& handle, const rr_view* xv,
                  const rr_view* yv, const sgd_options& o, rr_deposit* out) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const matrix<T, int64_t> x = checked_x<T>(xv, yv, o, false, stream);
  const T* y = raftrkt::vector_data<T, int32_t>(yv, "y", x.n);
  require_finite_elements("y", y, x.n, stream);
  auto owned =
      std::make_unique<sgd_fit<T>>(solved<T>(handle, x, y, o, false, true));
  const rr_view coef = raftrkt::region(owned->coef.data(), owned->d);
  const rr_view intercept = raftrkt::region(owned->intercept);
  settle(handle);
  raftrkt::deposit(out, "out", handle, std::move(owned), {coef, intercept});
}

template <typename T, typename L>
void classify_into(const raft::handle_t& handle, const rr_view* xv,
                   const rr_view* yv, bool y_is_codes, const sgd_options& o,
                   rr_deposit* out) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const bool ones =
      o.fit_intercept != 0 && o.loss != CUML_LINEAR_MODEL_SGD_SQUARED;
  const matrix<T, int64_t> x = checked_x<T>(xv, yv, o, ones, stream);
  const L* y = raftrkt::vector_data<L, int32_t>(yv, "y", x.n);
  require_integral(y, x.n, stream);
  dense_labels<L> labels = dense_codes(y, x.n, stream);
  require_two_classes(labels.k);
  buffer<T> targets = scratch<T>(x.n, stream);
  code_targets<T>(handle, labels.codes.data(), x.n,
                  o.loss == CUML_LINEAR_MODEL_SGD_HINGE, targets.data());
  sgd_fit<T> fit = solved<T>(handle, x, targets.data(), o, ones, false);
  auto owned = std::make_unique<classifier_outputs<T, L>>(
      classifier_outputs<T, L>{std::move(fit.coef), std::move(fit.intercept),
                               std::move(labels.values)});
  const rr_view coef = raftrkt::region(owned->coef.data(), x.d);
  const rr_view intercept = raftrkt::region(owned->intercept);
  const rr_view classes = raftrkt::region(owned->classes);
  settle(handle);
  if (y_is_codes) {
    raftrkt::deposit(out, "out", handle, std::move(owned), {coef, intercept});
  } else {
    raftrkt::deposit(out, "out", handle, std::move(owned),
                     {coef, intercept, classes});
  }
}

template <typename T>
void threshold_into(const raft::handle_t& handle, const rr_view* dv,
                    double threshold, const rr_view* cv, rr_deposit* out) {
  const T* decision =
      raftrkt::vector_data<T, int32_t>(dv, "decision", raftrkt::any_extent);
  const int64_t n = raftrkt::require(dv, "decision")->shape[0];
  const auto deposit_labels = [&](auto tag, const auto* classes) {
    using L = decltype(tag);
    auto labels = std::make_unique<buffer<L>>(scratch<L>(n, stream_of(handle)));
    threshold_labels<T, L>(handle, decision, n, threshold, classes,
                           labels->data());
    const rr_view region = raftrkt::region(*labels);
    settle(handle);
    raftrkt::deposit(out, "out", handle, std::move(labels), {region});
  };
  if (cv == nullptr) {
    deposit_labels(int32_t{}, static_cast<const int32_t*>(nullptr));
    return;
  }
  on_class_labels(cv, "classes", [&](auto tag) {
    using L = decltype(tag);
    deposit_labels(tag, raftrkt::vector_data<L, int32_t>(cv, "classes", 2));
  });
}

}  // namespace

}  // namespace cumlrkt::glm::mbsgd

extern "C" {

int cuml_linear_model_mbsgd_regressor_fit(
    void* handle, const rr_view* x, const rr_view* y, rr_deposit* out,
    int32_t penalty, double alpha, double l1_ratio, int32_t fit_intercept,
    int32_t epochs, double tol, int32_t shuffle, int32_t learning_rate,
    double eta0, double power_t, int32_t batch_size, int32_t n_iter_no_change,
    int32_t seeded, uint32_t seed) {
  return cumlrkt::guarded(
      handle, {{"X", x}, {"y", y}}, [&](const raft::handle_t& h) {
        rr_deposit* const into = raftrkt::require(out, "out");
        namespace mbsgd = cumlrkt::glm::mbsgd;
        const mbsgd::sgd_options o{.loss = CUML_LINEAR_MODEL_SGD_SQUARED,
                                   .penalty = penalty,
                                   .alpha = alpha,
                                   .l1_ratio = l1_ratio,
                                   .fit_intercept = fit_intercept,
                                   .epochs = epochs,
                                   .tol = tol,
                                   .shuffle = shuffle,
                                   .learning_rate = learning_rate,
                                   .eta0 = eta0,
                                   .power_t = power_t,
                                   .batch_size = batch_size,
                                   .n_iter_no_change = n_iter_no_change,
                                   .seeded = seeded,
                                   .seed = seed};
        cumlrkt::on_floats(x, "X", [&](auto tag) {
          mbsgd::regress_into<decltype(tag)>(h, x, y, o, into);
        });
      });
}

CUMLRKT_JOB(cuml_linear_model_mbsgd_regressor_fit)

int cuml_linear_model_mbsgd_classifier_fit(
    void* handle, const rr_view* x, const rr_view* y, int32_t y_is_codes,
    rr_deposit* out, int32_t loss, int32_t penalty, double alpha,
    double l1_ratio, int32_t fit_intercept, int32_t epochs, double tol,
    int32_t shuffle, int32_t learning_rate, double eta0, double power_t,
    int32_t batch_size, int32_t n_iter_no_change, int32_t seeded,
    uint32_t seed) {
  return cumlrkt::guarded(
      handle, {{"X", x}, {"y", y}}, [&](const raft::handle_t& h) {
        rr_deposit* const into = raftrkt::require(out, "out");
        namespace mbsgd = cumlrkt::glm::mbsgd;
        mbsgd::require_flag("y-is-codes", y_is_codes);
        const mbsgd::sgd_options o{.loss = loss,
                                   .penalty = penalty,
                                   .alpha = alpha,
                                   .l1_ratio = l1_ratio,
                                   .fit_intercept = fit_intercept,
                                   .epochs = epochs,
                                   .tol = tol,
                                   .shuffle = shuffle,
                                   .learning_rate = learning_rate,
                                   .eta0 = eta0,
                                   .power_t = power_t,
                                   .batch_size = batch_size,
                                   .n_iter_no_change = n_iter_no_change,
                                   .seeded = seeded,
                                   .seed = seed};
        cumlrkt::on_floats(x, "X", [&](auto t) {
          cumlrkt::on_class_labels(y, "y", [&](auto l) {
            mbsgd::classify_into<decltype(t), decltype(l)>(
                h, x, y, y_is_codes != 0, o, into);
          });
        });
      });
}

CUMLRKT_JOB(cuml_linear_model_mbsgd_classifier_fit)

int cuml_linear_model_threshold_labels(void* handle, const rr_view* decision,
                                       double threshold, const rr_view* classes,
                                       rr_deposit* out) {
  return cumlrkt::guarded(
      handle, {{"decision", decision}, {"classes", classes}},
      [&](const raft::handle_t& h) {
        rr_deposit* const into = raftrkt::require(out, "out");
        cumlrkt::on_floats(decision, "decision", [&](auto tag) {
          cumlrkt::glm::mbsgd::threshold_into<decltype(tag)>(
              h, decision, threshold, classes, into);
        });
      });
}

CUMLRKT_JOB(cuml_linear_model_threshold_labels)
}
