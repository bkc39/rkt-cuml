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
#include "logistic_resolve.hpp"
#include "logistic_solve.hpp"
#include "logistic_weights.hpp"
#include "min_rows.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt::glm::logistic {

namespace {

constexpr double unstaged_spread_least = 0x1p-8;
constexpr double unstaged_spread_most = 0x1p8;
constexpr double float32_l2_most = 0x1p21;
constexpr double float64_l2_most = 0x1p21;
constexpr double float64_l2_most_without_intercept = 0x1p46;

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
qn_solution<T> solve(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                     row_selection rows, const column_stats& stats,
                     staged_fit<T>& st, const fit_options& options) {
  attempt<T> plain =
      attempted<T>(handle, st.x(), st.y.data(), st.weights(),
                   {st.shape, st.sigma, st.mean, st.pen, 0}, options);
  if (plain.r.resolved()) {
    return std::move(plain.solution);
  }
  rescue<T> second = rescued<T>(handle, x, rows, stats, st, plain, options);
  if (!second.solution) {
    solve_check spent = plain.check;
    spent.n_iter = second.n_iter;
    refuse_cutoff(second.last.value_or(plain.r), options, spent);
  }
  return std::move(*second.solution);
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
  const qn_solution<T> solution =
      solve<T>(handle, x, rows_in(wt, x.n), stats, st, options);
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
  require_some("X", "row", x.n, min_rows);
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
