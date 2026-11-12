#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuml/solvers/params.hpp>
#include <cuml/solvers/solver.hpp>
#include <memory>
#include <optional>
#include <raft/core/handle.hpp>
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

struct cd_outputs {
  double intercept;
  int32_t n_iter;
};

template <typename T>
struct cd_solution {
  fit_result<T> fit;
  int32_t n_iter;
};

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
                const fit_plan& plan, const staging& s,
                const cd_request& request, const std::vector<T>& b,
                cd_scale<T> scale, int n_iter) {
  const T tol = static_cast<T>(request.tol);
  const double largest = largest_of(b);
  if (!absolute_stop(n_iter, request.max_iter, largest,
                     static_cast<double>(tol))) {
    return;
  }
  const row_selection rows = plan.counted.rows;
  const residual_terms terms = residual_moments(
      handle, in.x, in.y, in.w, rows, s, to_device(b, stream_of(handle)).data(),
      plan.fit_intercept);
  const double weight = in.w == nullptr
                            ? static_cast<double>(rows.count)
                            : weight_total(in.w, rows, stream_of(handle));
  const auto alpha = static_cast<double>(scale.alpha);
  const auto ratio = static_cast<double>(static_cast<T>(request.l1_ratio));
  const std::vector<double> coef(b.begin(), b.end());
  const cd_epochs epochs{n_iter, rows.count, static_cast<double>(tol),
                         unit_roundoff<T>};
  const kkt_terms k =
      kkt_of(terms, coef, weight, {alpha * ratio, alpha * (1 - ratio)}, epochs);
  require_kkt(k, n_iter, request.tol);
  require_resolved<T>(k, n_iter, request.tol, largest);
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
T unscaled_intercept(const std::optional<cd_centres>& centres,
                     const std::vector<T>& b, T staged, int shared) {
  if (centres) {
    return static_cast<T>(weighted_intercept(*centres, b, shared));
  }
  return std::ldexp(staged, -shared);
}

template <typename T>
cd_solution<T> cd_solved(const raft::handle_t& handle, const fit_inputs<T>& in,
                         const fit_plan& plan, const cd_request& request) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const auto& [x, y, w] = in;
  const row_selection rows = plan.counted.rows;
  const cd_scale<T> scale = cd_scale_of<T>(plan.s, rows.count, request.alpha);
  const staging s = shared_at(plan.s, scale.shared, scale.shared);
  const auto kept = static_cast<int64_t>(s.kept.size());
  const bool weighted = w != nullptr;
  const std::optional<cd_centres> centres =
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
  T* const unweighted = nullptr;
  const int32_t n_iter = ML::Solver::cdFit(
      const_cast<raft::handle_t&>(handle), p.x.data(),
      static_cast<int>(rows.count), static_cast<int>(kept), p.y.data(),
      cs.data(), &intercept, plan.fit_intercept && !weighted, request.max_iter,
      ML::loss_funct::SQRD_LOSS, scale.alpha, static_cast<T>(request.l1_ratio),
      plan.solver.algorithm == CUML_LINEAR_MODEL_CD_RANDOM,
      static_cast<T>(request.tol), unweighted);
  settle(handle);
  require_finite_solution(cs.data(), kept, intercept, plan.solver, stream);
  const std::vector<T> b = to_host(cs, stream);
  check_stop(handle, in, plan, s, request, b, scale, n_iter);
  auto coef = std::make_unique<buffer<T>>(scratch<T>(x.d, stream));
  const T unscaled =
      plan.fit_intercept
          ? unscaled_intercept(centres, b, intercept, scale.shared)
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
  const fit_plan plan{std::move(prepared.planned),
                      prepared.fit_intercept,
                      {glm_family::cd, request.fit.algorithm},
                      std::move(prepared.counted)};
  int32_t n_iter = 0;
  const double intercept = finish_fit<T>(
      handle, in, plan,
      [&] {
        cd_solution<T> solution = cd_solved(handle, in, plan, request);
        n_iter = solution.n_iter;
        return std::move(solution.fit);
      },
      out);
  return {intercept, n_iter};
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
