#include <cmath>
#include <cstdint>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <type_traits>

#include "cumlrkt/common.hpp"
#include "cumlrkt/metrics.h"
#include "cumlrkt/rows.hpp"
#include "cumlrkt/support.hpp"
#include "cumlrkt/vectors.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"
#include "scores_device.hpp"

namespace {

using cumlrkt::guarded;
using cumlrkt::largest_magnitude;
using cumlrkt::logic_error;
using cumlrkt::matrix;
using cumlrkt::on_floats;
using cumlrkt::on_labels;
using cumlrkt::require_finite;
using cumlrkt::require_finite_elements;
using cumlrkt::require_sample_weights;
using cumlrkt::require_some;
using cumlrkt::rows_of;
using cumlrkt::metrics::matched;
using cumlrkt::metrics::matches_of;
using cumlrkt::metrics::r2_of;
using cumlrkt::metrics::sample_weights;
using cumlrkt::metrics::squared_error_of;
using raftrkt::any_extent;
using raftrkt::vector_data;

using stream_view = rmm::cuda_stream_view;

stream_view stream_of(const raft::handle_t& handle) {
  return raft::resource::get_cuda_stream(handle);
}

sample_weights weight_data(const rr_view* view, int64_t n) {
  sample_weights w{nullptr, nullptr, 0};
  if (view != nullptr) {
    on_floats(view, "sample-weight", [&](auto tag) {
      using W = decltype(tag);
      const W* data = vector_data<W, int32_t>(view, "sample-weight", n);
      if constexpr (std::is_same_v<W, float>) {
        w.single = data;
      } else {
        w.wide = data;
      }
    });
  }
  return w;
}

template <typename W>
int checked_exponent(const W* w, int64_t n, stream_view stream) {
  return std::ilogb(require_sample_weights(w, n, stream));
}

sample_weights checked(sample_weights w, int64_t n, stream_view stream) {
  if (w.single != nullptr) {
    w.exponent = checked_exponent(w.single, n, stream);
  } else if (w.wide != nullptr) {
    w.exponent = checked_exponent(w.wide, n, stream);
  }
  return w;
}

template <typename T, typename P>
int value_exponent(const T* y, const P* p, int64_t count, stream_view stream) {
  const double largest = std::fmax(largest_magnitude(y, count, stream),
                                   largest_magnitude(p, count, stream));
  return largest > 0 ? std::ilogb(largest) : 0;
}

struct score_views {
  const rr_view* truth;
  const rr_view* pred;
  const rr_view* weights;
};

template <typename Use>
void on_value_pair(const score_views& views, Use&& use) {
  on_floats(views.truth, "y-true", [&](auto t) {
    using T = decltype(t);
    const T* y = vector_data<T, int32_t>(views.truth, "y-true");
    const int64_t n = views.truth->shape[0];
    on_floats(views.pred, "y-pred", [&](auto q) {
      using P = decltype(q);
      use(y, vector_data<P, int32_t>(views.pred, "y-pred", n), n);
    });
  });
}

template <typename Use>
void on_value_rows(const score_views& views, Use&& use) {
  on_floats(views.truth, "y-true", [&](auto t) {
    using T = decltype(t);
    const matrix<T, int32_t> y =
        rows_of<T, int32_t>(views.truth, "y-true", any_extent, any_extent);
    on_floats(views.pred, "y-pred", [&](auto q) {
      using P = decltype(q);
      use(y, rows_of<P, int32_t>(views.pred, "y-pred", y.n, y.d));
    });
  });
}

template <typename Use>
void on_label_pair(const score_views& views, Use&& use) {
  on_labels(views.truth, "y-true", [&](auto t) {
    using T = decltype(t);
    const T* a = vector_data<T, int32_t>(views.truth, "y-true");
    const int64_t n = views.truth->shape[0];
    on_labels(views.pred, "y-pred", [&](auto q) {
      using P = decltype(q);
      use(a, vector_data<P, int32_t>(views.pred, "y-pred", n), n);
    });
  });
}

double r2(const raft::handle_t& handle, const score_views& views) {
  const stream_view stream = stream_of(handle);
  double score = 0;
  on_value_pair(views, [&](const auto* y, const auto* p, int64_t n) {
    require_some("y-true", "element", n);
    const sample_weights w = weight_data(views.weights, n);
    require_finite_elements("y-true", y, n, stream);
    require_finite_elements("y-pred", p, n, stream);
    score = r2_of(y, p, n, value_exponent(y, p, n, stream),
                  checked(w, n, stream), stream);
  });
  return score;
}

template <typename T>
matrix<T, int64_t> wide(const matrix<T, int32_t>& m) {
  return {m.data, m.n, m.d};
}

double squared_error(const raft::handle_t& handle, const score_views& views) {
  const stream_view stream = stream_of(handle);
  double error = 0;
  const auto score = [&](const auto* y, const auto* p, int64_t n, int64_t d,
                         sample_weights w) {
    error = squared_error_of(y, p, n, d, value_exponent(y, p, n * d, stream),
                             checked(w, n, stream), stream);
  };
  if (raftrkt::require(views.truth, "y-true")->rank == 2) {
    on_value_rows(views, [&](const auto& y, const auto& p) {
      require_some("y-true", "row", y.n);
      require_some("y-true", "column", y.d);
      const sample_weights w = weight_data(views.weights, y.n);
      require_finite("y-true", wide(y), stream);
      require_finite("y-pred", wide(p), stream);
      score(y.data, p.data, y.n, y.d, w);
    });
  } else {
    on_value_pair(views, [&](const auto* y, const auto* p, int64_t n) {
      require_some("y-true", "element", n);
      const sample_weights w = weight_data(views.weights, n);
      require_finite_elements("y-true", y, n, stream);
      require_finite_elements("y-pred", p, n, stream);
      score(y, p, n, 1, w);
    });
  }
  return error;
}

matched accuracy(const raft::handle_t& handle, const score_views& views) {
  const stream_view stream = stream_of(handle);
  matched found{0, 0};
  on_label_pair(views, [&](const auto* a, const auto* b, int64_t n) {
    require_some("y-true", "element", n);
    const sample_weights w = checked(weight_data(views.weights, n), n, stream);
    found = matches_of(a, b, n, w, stream);
  });
  return found;
}

}  // namespace

extern "C" {

int cuml_metrics_r2_score(void* handle, const rr_view* y_true,
                          const rr_view* y_pred, const rr_view* sample_weight,
                          double* score) {
  return guarded(handle,
                 {{"y-true", y_true},
                  {"y-pred", y_pred},
                  {"sample-weight", sample_weight}},
                 [&](const raft::handle_t& h) {
                   double* const into = raftrkt::require(score, "score");
                   *into = r2(h, {y_true, y_pred, sample_weight});
                 });
}

int cuml_metrics_mean_squared_error(void* handle, const rr_view* y_true,
                                    const rr_view* y_pred,
                                    const rr_view* sample_weight,
                                    double* error) {
  return guarded(handle,
                 {{"y-true", y_true},
                  {"y-pred", y_pred},
                  {"sample-weight", sample_weight}},
                 [&](const raft::handle_t& h) {
                   double* const into = raftrkt::require(error, "error");
                   *into = squared_error(h, {y_true, y_pred, sample_weight});
                 });
}

int cuml_metrics_accuracy_score(void* handle, const rr_view* y_true,
                                const rr_view* y_pred,
                                const rr_view* sample_weight, double* correct,
                                double* total) {
  return guarded(
      handle,
      {{"y-true", y_true},
       {"y-pred", y_pred},
       {"sample-weight", sample_weight}},
      [&](const raft::handle_t& h) {
        double* const matched_into = raftrkt::require(correct, "correct");
        double* const total_into = raftrkt::require(total, "total");
        const matched found = accuracy(h, {y_true, y_pred, sample_weight});
        *matched_into = found.correct;
        *total_into = found.total;
      });
}

CUMLRKT_JOB(cuml_metrics_r2_score)
CUMLRKT_JOB(cuml_metrics_mean_squared_error)
CUMLRKT_JOB(cuml_metrics_accuracy_score)
}
