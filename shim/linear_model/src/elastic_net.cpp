#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuml/solvers/solver.hpp>
#include <limits>
#include <memory>
#include <raft/core/handle.hpp>
#include <ranges>
#include <rmm/cuda_stream_view.hpp>
#include <utility>
#include <vector>

#include "cd_device.hpp"
#include "cd_staging.hpp"
#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "fit_checks.hpp"
#include "glm_fit.hpp"
#include "glm_staging.hpp"
#include "glm_staging_device.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"
#include "residual_check.hpp"

namespace cumlrkt::glm {

namespace {

template <typename T>
struct cd_result {
  fit_result<T> fit;
  int n_iter;
};

struct cd_outputs {
  double intercept;
  int32_t n_iter;
};

template <typename T>
constexpr double roundoff =
    1.0 / static_cast<double>(uint64_t{1} << std::numeric_limits<T>::digits);

template <typename T>
double largest_of(const std::vector<T>& b) {
  double largest = 0;
  for (const T bk : b) {
    largest = std::max(largest, std::abs(static_cast<double>(bk)));
  }
  return largest;
}

template <typename T>
void check_stop(const raft::handle_t& handle, const fit_inputs<T>& in,
                const fit_plan& plan, const cd_request& request,
                const std::vector<T>& b, cd_scale<T> scale, int n_iter) {
  const T tol = static_cast<T>(request.tol);
  if (!absolute_stop(n_iter, request.max_iter, largest_of(b),
                     static_cast<double>(tol))) {
    return;
  }
  const row_selection rows = plan.counted.rows;
  const residual_terms terms = residual_moments(
      handle, in.x, in.y, in.w, rows, plan.s,
      to_device(b, stream_of(handle)).data(), plan.fit_intercept);
  const double weight = in.w == nullptr
                            ? static_cast<double>(rows.count)
                            : weight_total(in.w, rows, stream_of(handle));
  const auto alpha = static_cast<double>(scale.alpha);
  const auto ratio = static_cast<double>(static_cast<T>(request.l1_ratio));
  const std::vector<double> coef(b.begin(), b.end());
  const cd_epochs epochs{n_iter, rows.count, static_cast<double>(tol),
                         roundoff<T>};
  require_kkt(
      kkt_of(terms, coef, weight, {alpha * ratio, alpha * (1 - ratio)}, epochs),
      {n_iter, rows.count, request.tol, roundoff<T>});
}

template <typename T>
double weighted_intercept(const cd_centres& c, const std::vector<T>& b,
                          int shared) {
  double fitted = 0;
  for (size_t k = 0; k < b.size(); ++k) {
    fitted += (c.mean[k] + c.remainder[k]) * static_cast<double>(b[k]);
  }
  return std::ldexp((c.y.mean + c.y.remainder) - fitted, -shared);
}

template <typename T>
T unscaled_intercept(bool weighted, const cd_centres& c,
                     const std::vector<T>& b, T staged, int shared) {
  if (weighted) {
    return static_cast<T>(weighted_intercept(c, b, shared));
  }
  return std::ldexp(staged, -shared);
}

template <typename T>
cd_result<T> cd_solved(const raft::handle_t& handle, const fit_inputs<T>& in,
                       const fit_plan& plan, const cd_request& request,
                       cd_scale<T> scale) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const auto& [x, y, w] = in;
  const staging& s = plan.s;
  const row_selection rows = plan.counted.rows;
  const auto kept = static_cast<int64_t>(s.kept.size());
  const bool weighted = w != nullptr;
  const cd_centres centres =
      centres_of(y, w, rows, s, weighted && plan.fit_intercept, stream);
  staged_problem<T> p = weighted
                            ? stage_weighted(handle, x, y, w, rows, s, centres,
                                             weight_total(w, rows, stream))
                            : stage_problem(handle, x, y, w, s, rows);
  buffer<T> cs = scratch<T>(kept, stream);
  raftrkt::cuda_check(
      cudaMemsetAsync(cs.data(), 0, cs.size() * sizeof(T), stream.value()),
      "cudaMemsetAsync");
  T intercept{0};
  const int n_iter = ML::Solver::cdFit(
      const_cast<raft::handle_t&>(handle), p.x.data(),
      static_cast<int>(rows.count), static_cast<int>(kept), p.y.data(),
      cs.data(), &intercept, plan.fit_intercept && !weighted, request.max_iter,
      0, scale.alpha, static_cast<T>(request.l1_ratio),
      request.fit.algorithm == CUML_LINEAR_MODEL_CD_RANDOM,
      static_cast<T>(request.tol), nullptr);
  settle(handle);
  require_finite_solution(cs.data(), kept, intercept, cd_algorithm, stream);
  const std::vector<T> b = to_host(cs, stream);
  check_stop(handle, in, plan, request, b, scale, n_iter);
  auto coef = std::make_unique<buffer<T>>(scratch<T>(x.d, stream));
  const T unscaled =
      plan.fit_intercept
          ? unscaled_intercept(weighted, centres, b, intercept, scale.shared)
          : T{0};
  require_in_range(unstage_coef(handle, cs.data(), s, x.d, coef->data()),
                   unscaled);
  return {{std::move(coef), static_cast<double>(unscaled)}, n_iter};
}

template <typename T>
cd_outputs cd_fit_into(const raft::handle_t& handle, const cd_request& request,
                       rr_deposit* out) {
  const checked_fit<T> checked = checked_inputs<T>(
      handle, request.fit, [&] { require_cd_codes(request); });
  prepared_fit<T> prepared =
      prepare_fit(handle, checked, request.fit.fit_intercept);
  require_cd_rows(prepared.counted.rows.count);
  const fit_inputs<T> in = prepared.in;
  const bool constant_y =
      constant_vector(handle, in.y, in.x.n, in.w, prepared.fit_intercept);
  if (constant_y || prepared.planned.kept.empty()) {
    const fit_plan plan{std::move(prepared.planned),
                        prepared.fit_intercept,
                        {glm_family::cd, cd_algorithm},
                        std::move(prepared.counted)};
    return {deposit_fit(handle, nothing_to_solve(handle, in, plan, constant_y),
                        out),
            0};
  }
  const cd_scale<T> scale = cd_scale_of<T>(
      prepared.planned, prepared.counted.rows.count, request.alpha);
  staging s = std::move(prepared.planned);
  std::ranges::fill(s.column_exponent, scale.shared);
  s.y_exponent = scale.shared;
  const fit_plan plan{std::move(s),
                      prepared.fit_intercept,
                      {glm_family::cd, cd_algorithm},
                      std::move(prepared.counted)};
  cd_result<T> r = cd_solved(handle, in, plan, request, scale);
  return {deposit_fit(handle, std::move(r.fit), out), r.n_iter};
}

}  // namespace

}  // namespace cumlrkt::glm

extern "C" {

int cuml_linear_model_elastic_net_fit(
    void* handle, const rr_view* x, const rr_view* y,
    const rr_view* sample_weight, rr_deposit* out, double alpha,
    double l1_ratio, int32_t fit_intercept, int32_t max_iter, double tol,
    int32_t selection, double* intercept, int32_t* n_iter) {
  return cumlrkt::guarded(
      handle, {{"X", x}, {"y", y}, {"sample-weight", sample_weight}},
      [&](const raft::handle_t& h) {
        rr_deposit* const into = raftrkt::require(out, "out");
        double* const icpt = raftrkt::require(intercept, "intercept");
        int32_t* const epochs = raftrkt::require(n_iter, "n_iter");
        const cumlrkt::glm::cd_request request{
            {x, y, sample_weight, fit_intercept, selection},
            alpha,
            l1_ratio,
            max_iter,
            tol};
        cumlrkt::on_floats(x, "X", [&](auto tag) {
          const auto r =
              cumlrkt::glm::cd_fit_into<decltype(tag)>(h, request, into);
          *icpt = r.intercept;
          *epochs = r.n_iter;
        });
      });
}

CUMLRKT_JOB(cuml_linear_model_elastic_net_fit)
}
