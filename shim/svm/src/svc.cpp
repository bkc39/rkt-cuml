#include <cuml/svm/svm_model.h>
#include <cuml/svm/svm_parameter.h>

#include <cstdint>
#include <cuml/common/logger.hpp>
#include <cuml/matrix/kernel_params.hpp>
#include <cuml/svm/svc.hpp>
#include <format>
#include <memory>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <utility>

#include "cumlrkt/common.hpp"
#include "cumlrkt/support.hpp"
#include "cumlrkt/svm.h"
#include "cumlrkt/transpose.hpp"
#include "gamma.hpp"
#include "kernel_checks.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"
#include "svm_device.hpp"
#include "svm_model.hpp"

namespace cumlrkt::svm {

namespace {

struct fit_settings {
  int32_t kernel;
  int32_t degree;
  int32_t gamma_mode;
  double gamma;
  double coef0;
  double C;
  double tol;
  double cache_size;
  int32_t max_iter;
  int32_t nochange_steps;
};

struct fit_views {
  const rr_view* x;
  const rr_view* codes;
  const rr_view* weight;
};

struct fit_report {
  double gamma = 0.0;
  int32_t n_iter = 0;
  int64_t n_negative = 0;
  int64_t n_positive = 0;
  int32_t empty_class = -1;
};

template <typename T>
const T* checked_weights(const rr_view* view, int64_t n,
                         rmm::cuda_stream_view stream) {
  if (view == nullptr) {
    return nullptr;
  }
  T* w = raftrkt::vector_data<T, int32_t>(view, "sample-weight", n);
  const int64_t non_finite = first_non_finite(w, n, stream);
  if (non_finite >= 0) {
    throw logic_error(std::format(
        "sample-weight: row {}: expected a finite value, got {}", non_finite,
        printed(static_cast<double>(read_one(w, non_finite, stream)))));
  }
  const int64_t negative = first_negative(w, n, stream);
  if (negative >= 0) {
    throw logic_error(std::format(
        "sample-weight: negative weight {} at row {}",
        printed(static_cast<double>(read_one(w, negative, stream))), negative));
  }
  return w;
}

template <typename T>
int32_t empty_class(const T* w, const int32_t* codes, int64_t n, double C,
                    rmm::cuda_stream_view stream) {
  if (w == nullptr) {
    return -1;
  }
  const auto [first, second] = class_penalties(codes, w, n, C, stream);
  if (!(first > 0.0)) {
    return 0;
  }
  return second > 0.0 ? -1 : 1;
}

template <typename T>
double resolved_gamma(const fit_settings& s, const matrix<T, int64_t>& x,
                      rmm::cuda_stream_view stream) {
  switch (s.gamma_mode) {
    case CUML_SVM_GAMMA_SCALE:
      return scale_gamma(x.data, x.n, x.d, stream);
    case CUML_SVM_GAMMA_AUTO:
      return 1.0 / static_cast<double>(x.d);
    default:
      return s.gamma;
  }
}

constexpr double svr_epsilon = 0.1;

ML::SVM::SvmParameter solver_of(const fit_settings& s) {
  ML::SVM::SvmParameter p{};
  p.C = s.C;
  p.cache_size = s.cache_size;
  p.max_outer_iter = -1;
  p.max_iter = s.max_iter;
  p.nochange_steps = s.nochange_steps;
  p.tol = s.tol;
  p.verbosity = rapids_logger::level_enum::off;
  p.epsilon = svr_epsilon;
  p.svmType = ML::SVM::C_SVC;
  return p;
}

template <typename T>
void deposit_parts(const raft::handle_t& handle, svc_parts<T> parts, int64_t d,
                   rr_deposit* out) {
  auto owned = std::make_unique<svc_parts<T>>(std::move(parts));
  const int64_t ns = owned->n_support;
  const rr_view sv = raftrkt::region(owned->support_vectors.data(), ns, d,
                                     raftrkt::layout::row_major);
  const rr_view support = raftrkt::region(owned->support);
  const rr_view dual =
      raftrkt::region(owned->dual.data(), 1, ns, raftrkt::layout::row_major);
  const rr_view intercept = raftrkt::region(owned->intercept);
  settle(handle);
  raftrkt::deposit(out, "out", handle, std::move(owned),
                   {sv, support, dual, intercept});
}

template <typename T>
fit_report fit_into(const raft::handle_t& handle, const fit_views& views,
                    const fit_settings& s, rr_deposit* out) {
  require_narrowed<T>("C", s.C, true);
  require_narrowed<T>("tol", s.tol, true);
  require_narrowed<T>("coef0", s.coef0, false);
  if (s.gamma_mode == CUML_SVM_GAMMA_NUMBER) {
    require_narrowed<T>("gamma", s.gamma, false);
  }
  const rmm::cuda_stream_view stream = stream_of(handle);
  const matrix<T, int64_t> x = rows_of<T, int64_t>(
      views.x, "X", raftrkt::any_extent, raftrkt::any_extent);
  require_index("X", x.n, x.d);
  require_some("X", "row", x.n);
  require_some("X", "column", x.d);
  const int32_t* codes =
      raftrkt::vector_data<int32_t, int32_t>(views.codes, "y", x.n);
  const int64_t bad = first_bad_code(codes, x.n, stream);
  if (bad >= 0) {
    throw logic_error(
        std::format("y: row {}: expected a class code 0 or 1", bad));
  }
  const T* w = checked_weights<T>(views.weight, x.n, stream);
  const int32_t empty = empty_class(w, codes, x.n, s.C, stream);
  if (empty >= 0) {
    return {.empty_class = empty};
  }
  require_finite("X", x, stream);
  const double largest = largest_squared_norm(x.data, x.n, x.d, stream);
  require_row_norms("X", largest, x.data, x.n, x.d, stream);
  const double gamma = resolved_gamma(s, x, stream);
  require_narrowed<T>("gamma", gamma, false);
  const double weight =
      w == nullptr
          ? static_cast<double>(x.n) * static_cast<double>(static_cast<T>(s.C))
          : penalty_total(w, x.n, s.C, stream);
  require_kernel<T>({s.kernel, s.degree, gamma, s.coef0}, largest, weight);
  require_cache<T>(s.cache_size, x.n);

  buffer<T> columns = scratch<T>(x.n * x.d, stream);
  to_columns(handle, x.data, x.n, x.d, static_cast<const T*>(nullptr),
             columns.data(), 0);
  buffer<T> labels = scratch<T>(x.n, stream);
  codes_to(codes, x.n, labels.data(), stream);

  const ML::SVM::SvmParameter param = solver_of(s);
  ML::matrix::KernelParams kernel{static_cast<ML::matrix::KernelType>(s.kernel),
                                  s.degree, gamma, s.coef0};
  ML::SVM::SvmModel<T> model{};
  model.support_matrix.nnz = -1;
  const model_guard<T> guard{handle, model};
  const int n_iter = stall_translated([&] {
    return ML::SVM::svcFit(handle, columns.data(), static_cast<int>(x.n),
                           static_cast<int>(x.d), labels.data(), param, kernel,
                           model, w);
  });
  settle(handle);
  if (model.n_support == 0) {
    throw logic_error(
        "the fit has no support vectors; lower #:tol or raise #:max-iter");
  }
  svc_parts<T> parts = copy_out(handle, model);
  const fit_report report{gamma, n_iter, parts.n_negative, parts.n_positive};
  deposit_parts(handle, std::move(parts), x.d, out);
  return report;
}

}  // namespace

}  // namespace cumlrkt::svm

extern "C" {

int cuml_svm_svc_fit(void* handle, const rr_view* x, const rr_view* codes,
                     const rr_view* sample_weight, rr_deposit* out,
                     int32_t kernel, int32_t degree, int32_t gamma_mode,
                     double gamma, double coef0, double C, double tol,
                     double cache_size, int32_t max_iter,
                     int32_t nochange_steps, double* gamma_used,
                     int32_t* n_iter, int32_t* n_negative, int32_t* n_positive,
                     int32_t* empty_class) {
  return cumlrkt::guarded(
      handle, {{"X", x}, {"y", codes}, {"sample-weight", sample_weight}},
      [&](const raft::handle_t& h) {
        rr_deposit* const into = raftrkt::require(out, "out");
        double* const gamma_out = raftrkt::require(gamma_used, "gamma_used");
        int32_t* const iter_out = raftrkt::require(n_iter, "n_iter");
        int32_t* const negative_out =
            raftrkt::require(n_negative, "n_negative");
        int32_t* const positive_out =
            raftrkt::require(n_positive, "n_positive");
        int32_t* const empty_out = raftrkt::require(empty_class, "empty_class");
        if (kernel < CUML_SVM_KERNEL_LINEAR ||
            kernel > CUML_SVM_KERNEL_SIGMOID) {
          throw cumlrkt::logic_error("kernel: expected a code from 0 to 3");
        }
        if (gamma_mode < CUML_SVM_GAMMA_NUMBER ||
            gamma_mode > CUML_SVM_GAMMA_AUTO) {
          throw cumlrkt::logic_error("gamma-mode: expected a code from 0 to 2");
        }
        const cumlrkt::svm::fit_settings s{
            kernel, degree, gamma_mode, gamma,    coef0,
            C,      tol,    cache_size, max_iter, nochange_steps};
        cumlrkt::on_floats(x, "X", [&](auto tag) {
          const cumlrkt::svm::fit_report r =
              cumlrkt::svm::fit_into<decltype(tag)>(
                  h, {x, codes, sample_weight}, s, into);
          *gamma_out = r.gamma;
          *iter_out = r.n_iter;
          *negative_out = static_cast<int32_t>(r.n_negative);
          *positive_out = static_cast<int32_t>(r.n_positive);
          *empty_out = r.empty_class;
        });
      });
}

CUMLRKT_JOB(cuml_svm_svc_fit)
}
