#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuml/linear_model/glm.hpp>
#include <cuml/linear_model/qn.h>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cumlrkt/common.hpp"
#include "cumlrkt/labels.hpp"
#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "cumlrkt/vectors.hpp"
#include "glm_staging.hpp"
#include "glm_staging_device.hpp"
#include "logistic_device.hpp"
#include "logistic_fit.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt::glm::qn {

namespace {

struct penalties {
  double l1;
  double l2;
};

template <typename T, typename L>
struct qn_outputs {
  buffer<T> coef;
  buffer<T> intercept;
  rmm::device_uvector<L> classes;
};

template <typename T>
struct solved_fit {
  buffer<T> coef;
  buffer<T> intercept;
  int32_t n_iter;
};

struct weighting {
  buffer<double> weights;
  buffer<int64_t> index;
  row_selection rows;
  bool weighted;
  bool uniform;
  double scale;
};

penalties strengths(const qn_request& r) {
  const double strength = 1.0 / r.C;
  switch (r.penalty) {
    case CUML_LINEAR_MODEL_PENALTY_L1:
      return {strength, 0.0};
    case CUML_LINEAR_MODEL_PENALTY_L2:
      return {0.0, strength};
    case CUML_LINEAR_MODEL_PENALTY_ELASTICNET:
      return {r.l1_ratio * strength, (1.0 - r.l1_ratio) * strength};
    default:
      return {0.0, 0.0};
  }
}

template <typename L>
std::string label_text(L v) {
  if constexpr (std::is_integral_v<L>) {
    return std::format("{}", v);
  } else {
    return printed(static_cast<double>(v));
  }
}

template <typename L>
void require_integral(const L* y, int64_t n, rmm::cuda_stream_view stream) {
  if constexpr (!std::is_integral_v<L>) {
    const int64_t i = first_non_integral(y, n, stream);
    if (i >= 0) {
      throw logic_error(
          std::format("y: expected integral class labels, got {}",
                      label_text(to_host(y + i, 1, stream)[0])));
    }
  }
}

template <typename L>
std::vector<double> mapped_weights(const qn_request& r,
                                   const std::vector<L>& classes,
                                   rmm::cuda_stream_view stream) {
  const auto* keys = raftrkt::vector_data<double, int32_t>(
      r.keys, "class-weight", raftrkt::any_extent);
  const int64_t m = raftrkt::require(r.keys, "class-weight")->shape[0];
  const std::vector<double> key = to_host(keys, m, stream);
  const std::vector<double> value = to_host(
      raftrkt::vector_data<double, int32_t>(r.values, "class-weight", m), m,
      stream);
  std::vector<double> cw(classes.size(), 1.0);
  std::vector<std::string> missing;
  for (size_t c = 0; c < classes.size(); ++c) {
    const auto found = std::ranges::find(key, static_cast<double>(classes[c]));
    if (found == key.end()) {
      missing.push_back(label_text(classes[c]));
    } else {
      cw[c] = value[found - key.begin()];
    }
  }
  if (!missing.empty() &&
      static_cast<int64_t>(classes.size() - missing.size()) != m) {
    std::string listed;
    for (const std::string& s : missing) {
      listed += (listed.empty() ? "" : " ") + s;
    }
    throw logic_error(
        std::format("class-weight: classes ({}) are not in class-weight",
                    listed));
  }
  return cw;
}

template <typename L>
void require_weighted_classes(const std::vector<double>& totals,
                              const std::vector<L>& classes) {
  for (size_t c = 0; c < totals.size(); ++c) {
    if (!(totals[c] > 0)) {
      throw logic_error(std::format("y: class {} has no weight",
                                    label_text(classes[c])));
    }
  }
}

template <typename T, typename L>
std::optional<std::vector<double>> class_weights(
    const raft::handle_t& handle, const qn_request& r, const T* w,
    const dense_labels<L>& labels, const std::vector<L>& classes) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  if (r.class_mode == CUML_LINEAR_MODEL_CLASS_WEIGHT_GIVEN) {
    return mapped_weights(r, classes, stream);
  }
  if (r.class_mode != CUML_LINEAR_MODEL_CLASS_WEIGHT_BALANCED) {
    return std::nullopt;
  }
  const buffer<double> base =
      effective_weights<T>(handle, w, labels.codes.data(), nullptr,
                           static_cast<int64_t>(labels.codes.size()));
  const std::vector<double> totals =
      class_totals(stream, base.data(), labels.codes.data(),
                   static_cast<int64_t>(labels.codes.size()), labels.k);
  require_weighted_classes(totals, classes);
  double total = 0;
  for (const double t : totals) {
    total += t;
  }
  std::vector<double> cw(totals.size());
  std::ranges::transform(totals, cw.begin(), [&](double t) {
    return total / (static_cast<double>(totals.size()) * t);
  });
  return cw;
}

template <typename T, typename L>
weighting weighting_of(const raft::handle_t& handle, const T* w,
                       const dense_labels<L>& labels,
                       const std::optional<std::vector<double>>& cw,
                       const std::vector<L>& classes) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const auto n = static_cast<int64_t>(labels.codes.size());
  const buffer<double> device_cw =
      to_device(cw.value_or(std::vector<double>{}), stream);
  buffer<double> eff = effective_weights<T>(
      handle, w, labels.codes.data(), cw ? device_cw.data() : nullptr, n);
  require_weighted_classes(
      class_totals(stream, eff.data(), labels.codes.data(), n, labels.k),
      classes);
  const bool weighted = w != nullptr || cw.has_value();
  buffer<int64_t> index = scratch<int64_t>(weighted ? n : 0, stream);
  const int64_t m =
      weighted ? positive_weights(eff.data(), n, index.data(), stream) : n;
  const row_selection rows{m < n ? index.data() : nullptr, m};
  const bool uniform = !weighted || equal_weights(eff.data(), rows, stream);
  if (labels.k > 2 && !uniform) {
    throw logic_error(
        "sample-weight: weights with 3 or more classes must be equal on every "
        "row (cuML's softmax ignores them); got unequal weights");
  }
  const double scale =
      weighted ? static_cast<double>(m) / weight_sum(eff.data(), rows, stream)
               : 1.0;
  return {std::move(eff), std::move(index), rows, weighted, uniform, scale};
}

template <typename T>
int staging_exponent(const column_stats& s, int64_t m) {
  std::vector<double> sig(s.kept.size());
  std::ranges::transform(s.kept, sig.begin(),
                         [&](int64_t j) { return s.sigma[j]; });
  std::ranges::sort(sig);
  const size_t h = sig.size() / 2;
  const double median =
      sig.size() % 2 == 1 ? sig[h] : (sig[h - 1] / 2) + (sig[h] / 2);
  if (!(median > 0) || !std::isfinite(median) ||
      (median >= std::ldexp(1.0, -8) && median < std::ldexp(1.0, 8))) {
    return 0;
  }
  const int cap = column_reach<T>(m) - std::ilogb(s.largest);
  return std::min(-std::ilogb(median), cap);
}

penalties staged_penalties(penalties p, int s) {
  const penalties staged{std::ldexp(p.l1, s), std::ldexp(p.l2, 2 * s)};
  const auto lost = [](double before, double after) {
    return !std::isfinite(after) || (before != 0 && after == 0);
  };
  if (lost(p.l1, staged.l1) || lost(p.l2, staged.l2)) {
    throw logic_error(std::format(
        "C: the penalty at X's scale 2^{} is outside the float64 range", s));
  }
  return staged;
}

template <typename T>
void require_resolved(const residual_sums& sums, const std::vector<T>& params,
                      const qn_request& r, const qn_check& c) {
  const auto m = static_cast<double>(c.shape.m);
  const double l1 = r.normalized != 0 ? c.pen_l1 / m : c.pen_l1;
  const double l2 = r.normalized != 0 ? c.pen_l2 / m : c.pen_l2;
  const int64_t k = c.shape.k;
  double f = sums.loss / m;
  double worst = 0;
  for (int64_t q = 0; q < c.shape.p * k; ++q) {
    const auto v = static_cast<double>(params[q]);
    f += (l1 * std::abs(v)) + (l2 / 2 * v * v);
    const double g = (sums.gradient[q] / m) + (l2 * v);
    const double pg = v != 0 ? g + std::copysign(l1, v)
                             : std::copysign(std::max(std::abs(g) - l1, 0.0), g);
    worst = std::max(worst, std::abs(pg) / c.sigma[q / k]);
  }
  for (int64_t q = 0; c.shape.fit_intercept && q < k; ++q) {
    worst = std::max(worst, std::abs(sums.intercept[q] / m));
  }
  const double tau = std::max(r.tol, 1e-4);
  const double bound = 64 * tau * std::max(f, tau);
  if (!(worst <= bound)) {
    throw logic_error(std::format(
        "the solver stopped at first-order residual {} above the bound {} "
        "after {} iterations (max-iter {}, tol {}); raise max-iter or tol",
        printed(worst), printed(bound), c.n_iter, r.max_iter, printed(r.tol)));
  }
}

template <typename T>
solved_fit<T> closed_form(const raft::handle_t& handle, const weighting& wt,
                          const int32_t* codes, int64_t k, int64_t d,
                          bool fit_intercept) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const int64_t kk = k == 2 ? 1 : k;
  std::vector<T> b(kk, T{0});
  if (fit_intercept) {
    const std::vector<double> totals = class_totals(
        stream, wt.weights.data(), codes,
        static_cast<int64_t>(wt.weights.size()), k);
    std::vector<double> logs(k);
    std::ranges::transform(totals, logs.begin(),
                           [](double t) { return std::log(t); });
    double mean = 0;
    for (const double l : logs) {
      mean += l / static_cast<double>(k);
    }
    for (int64_t c = 0; c < kk; ++c) {
      b[c] = static_cast<T>(k == 2 ? logs[1] - logs[0] : logs[c] - mean);
    }
  }
  buffer<T> coef = scratch<T>(kk * d, stream);
  raftrkt::cuda_check(cudaMemsetAsync(coef.data(), 0, coef.size() * sizeof(T),
                                      stream.value()),
                      "cudaMemsetAsync");
  return {std::move(coef), to_device(b, stream), 0};
}

ML::GLM::qn_params params_of(const qn_request& r, int64_t k, penalties p) {
  ML::GLM::qn_params q;
  q.loss = k == 2 ? ML::GLM::QN_LOSS_LOGISTIC : ML::GLM::QN_LOSS_SOFTMAX;
  q.penalty_l1 = p.l1;
  q.penalty_l2 = p.l2;
  q.grad_tol = r.tol;
  q.change_tol = r.tol * 0.01;
  q.max_iter = r.max_iter;
  q.linesearch_max_iter = r.linesearch_max_iter;
  q.lbfgs_memory = r.lbfgs_memory;
  q.verbose = quiet_level;
  q.fit_intercept = r.fit_intercept != 0;
  q.penalty_normalized = r.normalized != 0;
  return q;
}

template <typename T>
solved_fit<T> solved(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                     const qn_request& r, const weighting& wt,
                     const int32_t* codes, int64_t k) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const bool fi = r.fit_intercept != 0;
  const column_stats stats = column_stats_of<T>(
      x, wt.weighted ? wt.weights.data() : nullptr, wt.rows, fi, stream);
  if (stats.kept.empty()) {
    return closed_form<T>(handle, wt, codes, k, x.d, fi);
  }
  const int s = staging_exponent<T>(stats, wt.rows.count);
  penalties pen = strengths(r);
  pen = {pen.l1 * wt.scale, pen.l2 * wt.scale};
  pen = staged_penalties(pen, s);
  const int64_t m = wt.rows.count;
  const auto p = static_cast<int64_t>(stats.kept.size());
  const int64_t kk = k == 2 ? 1 : k;
  const bool copy = s != 0 || m < x.n || p < x.d;
  buffer<T> staged = scratch<T>(copy ? m * p : 0, stream);
  if (copy) {
    const buffer<int64_t> kept = to_device(stats.kept, stream);
    stage_rows<T>(handle, x, wt.rows, kept.data(), p, s, staged.data());
  }
  T* const xs = copy ? staged.data() : x.data;
  buffer<T> y = scratch<T>(m, stream);
  buffer<T> w = scratch<T>(wt.uniform ? 0 : m, stream);
  T* const ws = wt.uniform ? nullptr : w.data();
  staged_targets<T>(handle, codes, wt.weights.data(), wt.scale, wt.rows,
                    y.data(), ws);
  buffer<T> params = scratch<T>((p + (fi ? 1 : 0)) * kk, stream);
  raftrkt::cuda_check(cudaMemsetAsync(params.data(), 0,
                                      params.size() * sizeof(T), stream.value()),
                      "cudaMemsetAsync");
  T objective{0};
  int n_iter = 0;
  ML::GLM::qnFit<T, int>(handle, params_of(r, k, pen), xs, false, y.data(),
                         static_cast<int>(m), static_cast<int>(p),
                         static_cast<int>(k), params.data(), &objective,
                         &n_iter, ws);
  settle(handle);
  if (first_non_finite(params.data(), static_cast<int64_t>(params.size()),
                       stream) >= 0) {
    throw logic_error("X: the qn solve gave a non-finite coefficient");
  }
  const qn_shape shape{m, p, kk, k, fi};
  std::vector<double> sigma(p);
  std::ranges::transform(stats.kept, sigma.begin(), [&](int64_t j) {
    return std::ldexp(stats.sigma[j], s);
  });
  require_resolved<T>(residuals_of<T>(handle, xs, y.data(), ws, params.data(), shape),
                      to_host(params, stream), r,
                      {shape, sigma, pen.l1, pen.l2, n_iter});
  std::vector<int64_t> slot(x.d, -1);
  for (int64_t q = 0; q < p; ++q) {
    slot[stats.kept[q]] = q;
  }
  const buffer<int64_t> slots = to_device(slot, stream);
  buffer<T> coef = scratch<T>(kk * x.d, stream);
  buffer<T> intercept = scratch<T>(kk, stream);
  params_to_coef<T>(handle, params.data(), slots.data(), x.d, kk, s, fi, p,
                    coef.data(), intercept.data());
  if (first_non_finite(coef.data(), kk * x.d, stream) >= 0) {
    throw logic_error(std::format("X: a coefficient exceeds the {} range",
                                  raftrkt::dtype_name(raftrkt::dtype_code<T>)));
  }
  settle(handle);
  return {std::move(coef), std::move(intercept), n_iter};
}

template <typename T, typename L>
int32_t fit_into(const raft::handle_t& handle, const qn_request& r,
                 rr_deposit* out) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const matrix<T, int64_t> x = rows_of<T, int64_t>(
      r.x, "X", raftrkt::any_extent, raftrkt::any_extent);
  const L* y = raftrkt::vector_data<L, int32_t>(r.y, "y", x.n);
  const T* w = r.w == nullptr ? nullptr
                              : raftrkt::vector_data<T, int32_t>(
                                    r.w, "sample-weight", x.n);
  require_some("X", "row", x.n, 2);
  require_some("X", "column", x.d);
  require_index("X", x.n, x.d);
  require_integral(y, x.n, stream);
  require_finite("X", x, stream);
  if (w != nullptr) {
    require_sample_weights(w, x.n, stream);
  }
  dense_labels<L> labels = dense_codes(y, x.n, stream);
  const std::vector<L> classes = to_host(labels.values.data(), labels.k, stream);
  const std::optional<std::vector<double>> cw =
      class_weights<T, L>(handle, r, w, labels, classes);
  if (labels.k < 2) {
    throw logic_error(
        std::format("y: expected at least 2 classes, got {}", labels.k));
  }
  const weighting wt = weighting_of<T, L>(handle, w, labels, cw, classes);
  solved_fit<T> fit = solved<T>(handle, x, r, wt, labels.codes.data(), labels.k);
  const int64_t kk = labels.k == 2 ? 1 : labels.k;
  auto outputs = std::make_unique<qn_outputs<T, L>>(qn_outputs<T, L>{
      std::move(fit.coef), std::move(fit.intercept), std::move(labels.values)});
  const rr_view coef_region = raftrkt::region(outputs->coef.data(), kk, x.d,
                                              raftrkt::layout::row_major);
  const rr_view intercept_region = raftrkt::region(outputs->intercept);
  const rr_view classes_region = raftrkt::region(outputs->classes);
  settle(handle);
  raftrkt::deposit(out, "out", handle, std::move(outputs),
                   {coef_region, intercept_region, classes_region});
  return fit.n_iter;
}

}  // namespace

void require_qn_codes(const qn_request& r) {
  if (r.penalty < CUML_LINEAR_MODEL_PENALTY_NONE ||
      r.penalty > CUML_LINEAR_MODEL_PENALTY_ELASTICNET) {
    throw logic_error("penalty: expected 'l1, 'l2, 'elasticnet or 'none");
  }
  require_positive("C", r.C);
  if (!std::isfinite(r.C)) {
    throw logic_error(std::format("C: expected a positive finite real, got {}",
                                  printed(r.C)));
  }
  if (!(r.l1_ratio >= 0 && r.l1_ratio <= 1)) {
    throw logic_error(std::format("l1-ratio: expected a real from 0 to 1, got {}",
                                  printed(r.l1_ratio)));
  }
  require_finite_non_negative("tol", r.tol);
  require_at_least("max-iter", r.max_iter, 1);
  require_at_least("linesearch-max-iter", r.linesearch_max_iter, 1);
  require_at_least("lbfgs-memory", r.lbfgs_memory, 1);
  if ((r.fit_intercept & ~1) != 0 || (r.normalized & ~1) != 0) {
    throw logic_error("fit-intercept: expected #t or #f");
  }
  if (r.class_mode < CUML_LINEAR_MODEL_CLASS_WEIGHT_NONE ||
      r.class_mode > CUML_LINEAR_MODEL_CLASS_WEIGHT_GIVEN) {
    throw logic_error("class-weight: expected #f, 'balanced or a hash");
  }
}

int32_t logistic_fit(const raft::handle_t& handle, const qn_request& r,
                     rr_deposit* out) {
  int32_t n_iter = 0;
  on_floats(r.x, "X", [&](auto t) {
    on_class_labels(r.y, [&](auto l) {
      n_iter = fit_into<decltype(t), decltype(l)>(handle, r, out);
    });
  });
  return n_iter;
}

}  // namespace cumlrkt::glm::qn
