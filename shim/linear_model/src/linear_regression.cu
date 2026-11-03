#include <cmath>
#include <cstdint>
#include <cuml/linear_model/glm.hpp>
#include <format>
#include <memory>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <utility>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "cumlrkt/transpose.hpp"
#include "cumlrkt/vectors.hpp"
#include "fit_checks.hpp"
#include "glm_staging.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt::glm {

namespace {

constexpr double single_tolerance = 0x1p-8;
constexpr double double_tolerance = 0x1p-18;

template <typename T>
struct fit_result {
  std::unique_ptr<buffer<T>> coef;
  double intercept;
};

template <typename T>
fit_result<T> nothing_to_solve(const raft::handle_t& handle,
                               const fit_inputs<T>& in, const staging& s,
                               bool fit_intercept, bool constant_y) {
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  auto coef = std::make_unique<buffer<T>>(scratch<T>(in.x.d, stream));
  raftrkt::cuda_check(cudaMemsetAsync(coef->data(), 0, coef->size() * sizeof(T),
                                      stream.value()),
                      "cudaMemsetAsync");
  double intercept = 0;
  if (fit_intercept && constant_y) {
    intercept = static_cast<double>(
        read_one(in.y, first_positive(in.w, in.x.n, stream), stream));
  } else if (fit_intercept) {
    buffer<T> ys = scratch<T>(in.x.n, stream);
    buffer<T> ws = scratch<T>(in.w == nullptr ? 0 : in.x.n, stream);
    stage_vector(handle, in.y, in.x.n, in.w, s.f, ys.data());
    if (in.w != nullptr) {
      scaled_copy(handle, in.w, in.x.n, ws.data(), s.g);
    }
    const double mean = staged_mean(
        handle, ys.data(), in.w == nullptr ? nullptr : ws.data(), in.x.n);
    intercept = static_cast<double>(
        static_cast<T>(std::ldexp(static_cast<T>(mean), -s.f)));
  }
  settle(handle);
  return {std::move(coef), intercept};
}

template <typename T>
fit_result<T> solved(const raft::handle_t& handle, const fit_inputs<T>& in,
                     const staging& s, bool fit_intercept, int32_t algorithm) {
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  const auto& [x, y, w] = in;
  const auto kept = static_cast<int64_t>(s.kept.size());
  const int effective =
      kept > x.n || kept == 1 ? CUML_LINEAR_MODEL_OLS_SVD : algorithm;
  require_workspace(effective, x.n, kept,
                    cusolver_lwork_of<T>(handle, effective, x.n, kept));
  auto coef = std::make_unique<buffer<T>>(scratch<T>(x.d, stream));
  buffer<T> xs = scratch<T>(x.n * kept, stream);
  buffer<T> ys = scratch<T>(x.n, stream);
  buffer<T> ws = scratch<T>(w == nullptr ? 0 : x.n, stream);
  buffer<T> cs = scratch<T>(kept, stream);
  T* staged_w = w == nullptr ? nullptr : ws.data();
  const auto stage_targets = [&] {
    stage_vector(handle, y, x.n, w, s.f, ys.data());
    if (w != nullptr) {
      scaled_copy(handle, w, x.n, ws.data(), s.g);
    }
  };
  stage_columns(handle, x, w, s, xs.data());
  stage_targets();
  T intercept{0};
  ML::GLM::olsFit(handle, xs.data(), static_cast<size_t>(x.n),
                  static_cast<size_t>(kept), ys.data(), cs.data(), &intercept,
                  fit_intercept, algorithm, staged_w);
  settle(handle);
  require_finite_solution(cs.data(), kept, intercept, effective, stream);
  stage_targets();
  const double cosine = residual_cosine(handle, x, s, ys.data(), staged_w,
                                        cs.data(), intercept, fit_intercept);
  const bool single = sizeof(T) == sizeof(float);
  const double tolerance = single ? single_tolerance : double_tolerance;
  if (!(cosine <= tolerance)) {
    throw logic_error(std::format(
        "X: the {} solve is not a least-squares solution (residual cosine {} "
        "> 2^-{}); X may have collinear columns",
        algorithm_name(effective), printed(cosine), single ? 8 : 18));
  }
  const int64_t j = unstage_coef(handle, cs.data(), s, x.d, coef->data());
  const char* dtype = single ? "float32" : "float64";
  if (j >= 0) {
    throw logic_error(std::format(
        "X: column {}: the coefficient exceeds the {} range", j, dtype));
  }
  const T unscaled = std::ldexp(intercept, -s.f);
  if (!std::isfinite(unscaled)) {
    throw logic_error(
        std::format("y: the intercept exceeds the {} range", dtype));
  }
  return {std::move(coef), fit_intercept ? static_cast<double>(unscaled) : 0.0};
}

template <typename T>
double fit_into(const raft::handle_t& handle, const rr_view* x_view,
                const rr_view* y_view, const rr_view* w_view, rr_deposit* out,
                int32_t fit_intercept, int32_t algorithm) {
  const fit_inputs<T> in = checked_inputs<T>(handle, x_view, y_view, w_view,
                                             fit_intercept, algorithm);
  const bool intercepted = fit_intercept == 1;
  const staging planned = plan_staging(handle, in.x, in.y, in.w, intercepted,
                                       exponents::per_column);
  const staging s = std::cmp_greater(planned.kept.size(), in.x.n)
                        ? plan_staging(handle, in.x, in.y, in.w, intercepted,
                                       exponents::uniform)
                        : planned;
  const bool constant_y =
      constant_vector(handle, in.y, in.x.n, in.w, intercepted);
  fit_result<T> r =
      constant_y || s.kept.empty()
          ? nothing_to_solve(handle, in, s, intercepted, constant_y)
          : solved(handle, in, s, intercepted, algorithm);
  settle(handle);
  const rr_view region = raftrkt::region(*r.coef);
  raftrkt::deposit(out, "out", handle, std::move(r.coef), {region});
  return r.intercept;
}

}  // namespace

}  // namespace cumlrkt::glm

extern "C" {

int cuml_linear_model_linear_regression_fit(
    void* handle, const rr_view* x, const rr_view* y,
    const rr_view* sample_weight, rr_deposit* out, int32_t fit_intercept,
    int32_t algorithm, double* intercept) {
  return cumlrkt::guarded(
      handle, {{"X", x}, {"y", y}, {"sample-weight", sample_weight}},
      [&](const raft::handle_t& h) {
        rr_deposit* const into = raftrkt::require(out, "out");
        double* const icpt = raftrkt::require(intercept, "intercept");
        cumlrkt::on_floats(x, "X", [&](auto tag) {
          *icpt = cumlrkt::glm::fit_into<decltype(tag)>(
              h, x, y, sample_weight, into, fit_intercept, algorithm);
        });
      });
}

CUMLRKT_JOB(cuml_linear_model_linear_regression_fit)
}
