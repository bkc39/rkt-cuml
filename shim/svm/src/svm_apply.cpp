#include <cuml/svm/svm_model.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuml/matrix/kernel_params.hpp>
#include <cuml/svm/svc.hpp>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <utility>

#include "cumlrkt/common.hpp"
#include "cumlrkt/label_entries.hpp"
#include "cumlrkt/support.hpp"
#include "cumlrkt/svm.h"
#include "cumlrkt/transpose.hpp"
#include "kernel_checks.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"
#include "svm_device.hpp"
#include "svm_model.hpp"

namespace cumlrkt::svm {

namespace {

struct apply_views {
  const rr_view* x;
  const rr_view* support_vectors;
  const rr_view* dual;
  const rr_view* intercept;
  const rr_view* classes;
};

struct apply_settings {
  int32_t kernel;
  int32_t degree;
  double gamma;
  double coef0;
  double cache_size;
  int32_t mode;
};

template <typename T>
buffer<T> decision_of(const raft::handle_t& handle, const apply_views& views,
                      const apply_settings& s) {
  require_narrowed<T>("gamma", s.gamma, false);
  require_narrowed<T>("coef0", s.coef0, false);
  require_positive_cache(s.cache_size);
  require_narrowed<T>("cache-size", s.cache_size, false);
  const rmm::cuda_stream_view stream = stream_of(handle);
  const matrix<T, int64_t> x = rows_of<T, int64_t>(
      views.x, "X", raftrkt::any_extent, raftrkt::any_extent);
  const matrix<T, int64_t> sv = rows_of<T, int64_t>(
      views.support_vectors, "support-vectors", raftrkt::any_extent, x.d);
  T* dual = raftrkt::matrix_data<T, int32_t>(
      views.dual, "dual-coef", raftrkt::layout::row_major, 1, sv.n);
  const T* intercept =
      raftrkt::vector_data<T, int32_t>(views.intercept, "intercept", 1);
  require_index("X", x.n, x.d);
  require_index("support-vectors", sv.n, sv.d);
  require_some("support-vectors", "row", sv.n);
  require_predict_tile<T>(s.cache_size, x.n, sv.n);
  require_finite("X", x, stream);
  const double x_norm = largest_squared_norm(x.data, x.n, x.d, stream);
  require_row_norms("X", x_norm, x.data, x.n, x.d, stream);
  const double sv_norm = largest_squared_norm(sv.data, sv.n, sv.d, stream);
  const kernel_shape k{s.kernel, s.degree, s.gamma, s.coef0};
  const bool inner =
      s.kernel == CUML_SVM_KERNEL_LINEAR || s.kernel == CUML_SVM_KERNEL_POLY;
  require_kernel<T>(k,
                    inner ? std::sqrt(x_norm) * std::sqrt(sv_norm)
                          : std::max(x_norm, sv_norm),
                    absolute_total(dual, sv.n, stream));

  buffer<T> x_columns = scratch<T>(x.n * x.d, stream);
  to_columns(handle, x.data, x.n, x.d, static_cast<const T*>(nullptr),
             x_columns.data(), 0);
  buffer<T> sv_columns = scratch<T>(sv.n * sv.d, stream);
  to_columns(handle, sv.data, sv.n, sv.d, static_cast<const T*>(nullptr),
             sv_columns.data(), 0);
  const T b = read_one(intercept, 0, stream);
  const ML::SVM::SvmModel<T> model =
      rebuilt<T>({.n_support = static_cast<int>(sv.n),
                  .n_cols = static_cast<int>(sv.d),
                  .b = b,
                  .dual = dual,
                  .support_columns = sv_columns.data()});
  ML::matrix::KernelParams kernel{static_cast<ML::matrix::KernelType>(s.kernel),
                                  s.degree, s.gamma, s.coef0};
  buffer<T> out = scratch<T>(x.n, stream);
  if (x.n > 0) {
    ML::SVM::svcPredict(handle, x_columns.data(), static_cast<int>(x.n),
                        static_cast<int>(x.d), kernel, model, out.data(),
                        static_cast<T>(s.cache_size), false);
  }
  return out;
}

template <typename T>
void apply_into(const raft::handle_t& handle, const apply_views& views,
                const apply_settings& s, rr_deposit* out) {
  buffer<T> decision = decision_of<T>(handle, views, s);
  const auto n = static_cast<int64_t>(decision.size());
  const rmm::cuda_stream_view stream = stream_of(handle);
  if (s.mode == CUML_SVM_APPLY_DECISION) {
    deposit_vector(handle, out, std::move(decision));
  } else if (views.classes == nullptr) {
    buffer<int32_t> labels = scratch<int32_t>(n, stream);
    threshold_labels<T, int32_t>(decision.data(), n, nullptr, labels.data(),
                                 stream);
    deposit_vector(handle, out, std::move(labels));
  } else {
    on_class_labels(views.classes, "classes", [&](auto tag) {
      using L = decltype(tag);
      const L* classes =
          raftrkt::vector_data<L, int32_t>(views.classes, "classes", 2);
      buffer<L> labels = scratch<L>(n, stream);
      threshold_labels<T, L>(decision.data(), n, classes, labels.data(),
                             stream);
      deposit_vector(handle, out, std::move(labels));
    });
  }
}

}  // namespace

}  // namespace cumlrkt::svm

extern "C" {

int cuml_svm_svc_apply(void* handle, const rr_view* x,
                       const rr_view* support_vectors, const rr_view* dual_coef,
                       const rr_view* intercept, const rr_view* classes,
                       rr_deposit* out, int32_t kernel, int32_t degree,
                       double gamma, double coef0, double cache_size,
                       int32_t mode) {
  return cumlrkt::guarded(
      handle,
      {{"X", x},
       {"support-vectors", support_vectors},
       {"dual-coef", dual_coef},
       {"intercept", intercept},
       {"classes", classes}},
      [&](const raft::handle_t& h) {
        rr_deposit* const into = raftrkt::require(out, "out");
        if (mode < CUML_SVM_APPLY_DECISION || mode > CUML_SVM_APPLY_LABELS) {
          throw cumlrkt::logic_error("mode: expected 0 or 1");
        }
        if (kernel < CUML_SVM_KERNEL_LINEAR ||
            kernel > CUML_SVM_KERNEL_SIGMOID) {
          throw cumlrkt::logic_error("kernel: expected a code from 0 to 3");
        }
        const cumlrkt::svm::apply_settings s{kernel, degree,     gamma,
                                             coef0,  cache_size, mode};
        cumlrkt::on_floats(x, "X", [&](auto tag) {
          cumlrkt::svm::apply_into<decltype(tag)>(
              h, {x, support_vectors, dual_coef, intercept, classes}, s, into);
        });
      });
}

CUMLRKT_JOB(cuml_svm_svc_apply)
}
