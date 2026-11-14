#include <cuml/linear_model/qn.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuml/linear_model/glm.hpp>
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
#include "label_checks.hpp"
#include "logistic_checks.hpp"
#include "logistic_device.hpp"
#include "logistic_fit.hpp"
#include "logistic_weights.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt::glm::logistic {

namespace {

constexpr double change_tol_per_tol = 0.01;
constexpr double unstaged_spread_least = 0x1p-8;
constexpr double unstaged_spread_most = 0x1p8;
constexpr double tolerance_floor = 1e-4;
constexpr double residual_factor = 64;
constexpr double float32_l2_most = 0x1p21;
constexpr double float64_l2_most = 0x1p21;
constexpr double float64_l2_most_without_intercept = 0x1p46;

struct penalties {
  double l1;
  double l2;
};

template <typename T, typename L>
struct fit_outputs {
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

struct solve_check {
  qn_shape shape;
  std::vector<double> sigma;
  std::vector<double> mean;
  penalties pen;
  int n_iter;
};

template <typename T>
struct staged_fit {
  std::optional<buffer<T>> copied;
  T* source;
  buffer<T> y;
  std::optional<buffer<T>> w;
  std::vector<int64_t> kept;
  int shift;
  penalties pen;
  qn_shape shape;
  std::vector<double> sigma;
  std::vector<double> mean;

  T* x() {
    return copied ? copied->data() : source;
  }
  T* weights() {
    return w ? w->data() : nullptr;
  }
};

template <typename T>
struct qn_solution {
  buffer<T> params;
  int n_iter;
};

penalties penalties_of(const fit_options& o) {
  const double strength = 1.0 / o.C;
  switch (o.penalty) {
    case CUML_LINEAR_MODEL_PENALTY_L1:
      return {strength, 0.0};
    case CUML_LINEAR_MODEL_PENALTY_L2:
      return {0.0, strength};
    case CUML_LINEAR_MODEL_PENALTY_ELASTICNET:
      return {o.l1_ratio * strength, (1.0 - o.l1_ratio) * strength};
    default:
      return {0.0, 0.0};
  }
}

template <typename T>
int staging_exponent(const column_stats& s, int64_t m) {
  std::vector<double> sig(s.kept.size());
  std::ranges::transform(s.kept, sig.begin(),
                         [&](int64_t j) { return s.sigma[j]; });
  const auto middle = sig.begin() + static_cast<std::ptrdiff_t>(sig.size() / 2);
  std::ranges::nth_element(sig, middle);
  const double median =
      sig.size() % 2 == 1
          ? *middle
          : (*std::ranges::max_element(sig.begin(), middle) / 2) +
                (*middle / 2);
  if (!(median > 0) || !std::isfinite(median) ||
      (median >= unstaged_spread_least && median < unstaged_spread_most)) {
    return 0;
  }
  const int cap = column_reach<T>(m) - std::ilogb(s.largest);
  return std::min(-std::ilogb(median), cap);
}

template <typename T>
double resolved_l2(bool fit_intercept) {
  if constexpr (std::is_same_v<T, float>) {
    return float32_l2_most;
  }
  return fit_intercept ? float64_l2_most : float64_l2_most_without_intercept;
}

template <typename T>
penalties staged_penalties(penalties p, int s, const fit_options& o, int64_t m,
                           bool weighted) {
  const penalties staged{std::ldexp(p.l1, s), std::ldexp(p.l2, 2 * s)};
  const auto lost = [](double before, double after) {
    const auto stored = static_cast<T>(after);
    return !std::isfinite(stored) || (before != 0 && stored == 0);
  };
  const auto dtype = raftrkt::dtype_name(raftrkt::dtype_code<T>);
  if (lost(p.l1, staged.l1) || lost(p.l2, staged.l2)) {
    throw logic_error(std::format(
        "C: the penalty at X's scale 2^{} is outside the {} range", -s, dtype));
  }
  const double solved_l2 =
      o.normalized ? staged.l2 / static_cast<double>(m) : staged.l2;
  if (solved_l2 > resolved_l2<T>(o.fit_intercept)) {
    throw logic_error(std::format(
        "C: the penalty is too strong for the {} solver to resolve at X's "
        "scale 2^{}{}",
        dtype, -s,
        weighted ? " or the sample weights; raise C or the sample weights, or "
                   "standardise X"
                 : "; raise C or standardise X"));
  }
  return staged;
}

template <typename T>
void require_spread(const column_stats& s) {
  for (const int64_t j : s.kept) {
    if (!std::isfinite(s.sigma[j])) {
      throw logic_error(std::format(
          "X: column {} has values up to {} in magnitude; its spread is "
          "outside the float64 range",
          j, printed(s.magnitude[j])));
    }
    if (s.sigma[j] <= constant_spread * unit_roundoff<T> * s.magnitude[j]) {
      throw logic_error(std::format(
          "X: column {} varies only in its last bits (spread {} of {}); "
          "centre it",
          j, printed(s.sigma[j]), printed(s.magnitude[j])));
    }
  }
}

template <typename T>
double residual_spread(const column_stats& s, int64_t j) {
  const bool resolved =
      s.centred[j] > constant_spread * unit_roundoff<T> * s.magnitude[j];
  return resolved ? std::min(s.centred[j], s.sigma[j]) : s.sigma[j];
}

template <typename T>
void require_staged_columns(const column_stats& s, int shift) {
  const auto widest =
      std::ranges::max(s.kept, {}, [&](int64_t j) { return s.magnitude[j]; });
  for (const int64_t j : s.kept) {
    if (shift == 0 && s.magnitude[j] < std::numeric_limits<T>::min()) {
      throw logic_error(std::format(
          "X: column {} has values only up to {} in magnitude, subnormal in "
          "{}; scale it",
          j, printed(s.magnitude[j]),
          raftrkt::dtype_name(raftrkt::dtype_code<T>)));
    }
    if (std::ldexp(s.magnitude[j], shift) < std::numeric_limits<T>::min()) {
      throw logic_error(std::format(
          "X: columns {} and {} span too wide a range; standardise them", j,
          widest));
    }
  }
}

const char* cutoff_advice(const solve_check& c, const fit_options& o) {
  if (c.n_iter >= o.max_iter) {
    return "raise max-iter";
  }
  return c.shape.fit_intercept ? "lower tol, or centre X"
                               : "lower tol, centre X, or fit an intercept";
}

template <typename T>
void require_resolved(const residual_sums& sums, const std::vector<T>& params,
                      const fit_options& options, const solve_check& c) {
  const auto m = static_cast<double>(c.shape.m);
  const auto per_row = [m](double v) { return v / m; };
  const penalties pen = options.normalized
                            ? penalties{per_row(c.pen.l1), per_row(c.pen.l2)}
                            : c.pen;
  const int64_t k = c.shape.k;
  double f = per_row(sums.loss);
  double worst = 0;
  for (int64_t q = 0; q < c.shape.p * k; ++q) {
    const auto v = static_cast<double>(params[q]);
    f += (pen.l1 * std::abs(v)) + (pen.l2 / 2 * v * v);
    const double centred = c.shape.fit_intercept
                               ? c.mean[q / k] * per_row(sums.intercept[q % k])
                               : 0.0;
    const double g = per_row(sums.gradient[q]) - centred + (pen.l2 * v);
    const double pg =
        v != 0 ? g + std::copysign(pen.l1, v)
               : std::copysign(std::max(std::abs(g) - pen.l1, 0.0), g);
    worst = std::max(worst, std::abs(pg) / c.sigma[q / k]);
  }
  for (int64_t q = 0; c.shape.fit_intercept && q < k; ++q) {
    worst = std::max(worst, std::abs(per_row(sums.intercept[q])));
  }
  const double tau = std::max(options.tol, tolerance_floor);
  const double bound = residual_factor * tau * std::max(f, tau);
  if (!(worst <= bound)) {
    throw logic_error(std::format(
        "the solver stopped at first-order residual {} above the bound {} "
        "after {} iterations (max-iter {}, tol {}); {}",
        printed(worst), printed(bound), c.n_iter, options.max_iter,
        printed(options.tol), cutoff_advice(c, options)));
  }
}

template <typename T>
solved_fit<T> closed_form(const raft::handle_t& handle,
                          const std::optional<weighted_rows>& wt,
                          const int32_t* codes, int64_t k,
                          const matrix<T, int64_t>& x, bool fit_intercept) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const int64_t outputs = outputs_of(k);
  std::vector<T> b(outputs, T{0});
  if (fit_intercept) {
    const std::vector<double> totals = class_totals(
        stream, weights_in(wt), codes, row_selection{nullptr, x.n}, k);
    std::vector<double> logs(k);
    std::ranges::transform(totals, logs.begin(),
                           [](double t) { return std::log(t); });
    double mean = 0;
    for (const double l : logs) {
      mean += l / static_cast<double>(k);
    }
    for (int64_t c = 0; c < outputs; ++c) {
      b[c] = static_cast<T>(k == 2 ? logs[1] - logs[0] : logs[c] - mean);
    }
  }
  return {zeroed<T>(outputs * x.d, stream), to_device(b, stream), 0};
}

ML::GLM::qn_params qn_params_of(const fit_options& o, int64_t k, penalties p) {
  ML::GLM::qn_params q;
  q.loss = k == 2 ? ML::GLM::QN_LOSS_LOGISTIC : ML::GLM::QN_LOSS_SOFTMAX;
  q.penalty_l1 = p.l1;
  q.penalty_l2 = p.l2;
  q.grad_tol = o.tol;
  q.change_tol = o.tol * change_tol_per_tol;
  q.max_iter = o.max_iter;
  q.linesearch_max_iter = o.linesearch_max_iter;
  q.lbfgs_memory = o.lbfgs_memory;
  q.verbose = quiet_level;
  q.fit_intercept = o.fit_intercept;
  q.penalty_normalized = o.normalized;
  return q;
}

template <typename T>
staged_fit<T> staged_of(const raft::handle_t& handle,
                        const matrix<T, int64_t>& x, const fit_options& options,
                        const std::optional<weighted_rows>& wt,
                        const int32_t* codes, int64_t k,
                        const column_stats& stats) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const row_selection rows = rows_in(wt, x.n);
  require_spread<T>(stats);
  const int s = staging_exponent<T>(stats, rows.count);
  require_staged_columns<T>(stats, s);
  penalties pen = penalties_of(options);
  if (wt) {
    pen = {pen.l1 * wt->scale, pen.l2 * wt->scale};
  }
  const int64_t m = rows.count;
  pen = staged_penalties<T>(pen, s, options, m,
                            wt && wt->sampled && wt->scale != 1.0);
  const auto p = static_cast<int64_t>(stats.kept.size());
  std::optional<buffer<T>> copied;
  if (s != 0 || m < x.n || p < x.d) {
    copied.emplace(scratch<T>(m * p, stream));
    const buffer<int64_t> kept = to_device(stats.kept, stream);
    stage_rows<T>(handle, x, rows, kept.data(), p, s, copied->data());
  }
  buffer<T> y = scratch<T>(m, stream);
  std::optional<buffer<T>> w;
  if (wt && !wt->uniform) {
    w.emplace(scratch<T>(m, stream));
  }
  staged_targets<T>(handle, codes, weights_in(wt), wt ? wt->scale : 1.0, rows,
                    y.data(), w ? w->data() : nullptr);
  std::vector<double> sigma(p);
  std::vector<double> mean(p);
  std::ranges::transform(stats.kept, sigma.begin(), [&](int64_t j) {
    return std::ldexp(residual_spread<T>(stats, j), s);
  });
  std::ranges::transform(stats.kept, mean.begin(), [&](int64_t j) {
    return std::ldexp(stats.mean[j], s);
  });
  return {std::move(copied),
          x.data,
          std::move(y),
          std::move(w),
          stats.kept,
          s,
          pen,
          {m, p, outputs_of(k), k, options.fit_intercept},
          std::move(sigma),
          std::move(mean)};
}

template <typename T>
qn_solution<T> solve(const raft::handle_t& handle, staged_fit<T>& st,
                     const fit_options& options) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const qn_shape& shape = st.shape;
  buffer<T> params =
      zeroed<T>((shape.p + (shape.fit_intercept ? 1 : 0)) * shape.k, stream);
  T objective{0};
  int n_iter = 0;
  ML::GLM::qnFit<T, int>(handle, qn_params_of(options, shape.classes, st.pen),
                         st.x(), false, st.y.data(), static_cast<int>(shape.m),
                         static_cast<int>(shape.p),
                         static_cast<int>(shape.classes), params.data(),
                         &objective, &n_iter, st.weights());
  settle(handle);
  require_finite_solution(params.data(), static_cast<int64_t>(params.size()),
                          stream);
  require_resolved<T>(residuals_of<T>(handle, st.x(), st.y.data(), st.weights(),
                                      params.data(), shape),
                      to_host(params, stream), options,
                      {shape, st.sigma, st.mean, st.pen, n_iter});
  return {std::move(params), n_iter};
}

template <typename T>
solved_fit<T> unstaged(const raft::handle_t& handle, const staged_fit<T>& st,
                       const qn_solution<T>& solution, int64_t d) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const qn_shape& shape = st.shape;
  std::vector<int64_t> slot(d, -1);
  for (int64_t q = 0; q < shape.p; ++q) {
    slot[st.kept[q]] = q;
  }
  const buffer<int64_t> slots = to_device(slot, stream);
  buffer<T> coef = scratch<T>(shape.k * d, stream);
  buffer<T> intercept = scratch<T>(shape.k, stream);
  params_to_coef<T>(handle, solution.params.data(), slots.data(), d, shape.k,
                    st.shift, shape.fit_intercept, shape.p, coef.data(),
                    intercept.data());
  require_coef_in_range(coef.data(), shape.k * d, stream);
  settle(handle);
  return {std::move(coef), std::move(intercept), solution.n_iter};
}

template <typename T>
solved_fit<T> solved(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                     const fit_options& options,
                     const std::optional<weighted_rows>& wt,
                     const int32_t* codes, int64_t k) {
  const column_stats stats =
      column_stats_of<T>(x, weights_in(wt), rows_in(wt, x.n),
                         options.fit_intercept, stream_of(handle));
  if (stats.kept.empty()) {
    return closed_form<T>(handle, wt, codes, k, x, options.fit_intercept);
  }
  staged_fit<T> st = staged_of<T>(handle, x, options, wt, codes, k, stats);
  const qn_solution<T> solution = solve<T>(handle, st, options);
  return unstaged<T>(handle, st, solution, x.d);
}

template <typename T, typename L>
int32_t deposit_fit(const raft::handle_t& handle, solved_fit<T> fit,
                    rmm::device_uvector<L> classes, int64_t d,
                    rr_deposit* out) {
  const int64_t outputs = outputs_of(static_cast<int64_t>(classes.size()));
  auto owned = std::make_unique<fit_outputs<T, L>>(fit_outputs<T, L>{
      std::move(fit.coef), std::move(fit.intercept), std::move(classes)});
  const rr_view coef_region = raftrkt::region(owned->coef.data(), outputs, d,
                                              raftrkt::layout::row_major);
  const rr_view intercept_region = raftrkt::region(owned->intercept);
  const rr_view classes_region = raftrkt::region(owned->classes);
  settle(handle);
  raftrkt::deposit(out, "out", handle, std::move(owned),
                   {coef_region, intercept_region, classes_region});
  return fit.n_iter;
}

template <typename T, typename L>
int32_t fit_into(const raft::handle_t& handle, const fit_views& views,
                 const fit_options& options, rr_deposit* out) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const matrix<T, int64_t> x = rows_of<T, int64_t>(
      views.x, "X", raftrkt::any_extent, raftrkt::any_extent);
  const L* y = raftrkt::vector_data<L, int32_t>(views.y, "y", x.n);
  const T* w = views.sample_weight == nullptr
                   ? nullptr
                   : raftrkt::vector_data<T, int32_t>(views.sample_weight,
                                                      "sample-weight", x.n);
  require_some("X", "row", x.n, 2);
  require_some("X", "column", x.d);
  require_index("X", x.n, x.d);
  require_integral(y, x.n, stream);
  require_finite("X", x, stream);
  if (w != nullptr) {
    require_sample_weights(w, x.n, stream);
  }
  dense_labels<L> labels = dense_codes(y, x.n, stream);
  const std::vector<L> classes =
      to_host(labels.values.data(), labels.k, stream);
  const std::optional<std::vector<double>> cw =
      class_weights_of<T, L>(handle, views, options, w, labels, classes);
  if (labels.k < 2) {
    throw logic_error(
        std::format("y: expected at least 2 classes, got {}", labels.k));
  }
  const std::optional<weighted_rows> wt =
      weighted_rows_of<T, L>(handle, w, labels, cw, classes);
  solved_fit<T> fitted =
      solved<T>(handle, x, options, wt, labels.codes.data(), labels.k);
  return deposit_fit<T, L>(handle, std::move(fitted), std::move(labels.values),
                           x.d, out);
}

}  // namespace

int32_t fit(const raft::handle_t& handle, const fit_views& views,
            const fit_options& options, rr_deposit* out) {
  int32_t n_iter = 0;
  on_floats(views.x, "X", [&](auto t) {
    on_class_labels(views.y, "y", [&](auto l) {
      n_iter = fit_into<decltype(t), decltype(l)>(handle, views, options, out);
    });
  });
  return n_iter;
}

}  // namespace cumlrkt::glm::logistic
