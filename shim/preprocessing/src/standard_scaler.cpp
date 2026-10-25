#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <memory>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <string>
#include <utility>

#include "cumlrkt/preprocessing.h"
#include "cumlrkt/support.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"
#include "standard_scaler_device.hpp"

namespace {

using cumlrkt::guarded;
using cumlrkt::logic_error;
using cumlrkt::matrix;
using cumlrkt::most_index;
using cumlrkt::on_floats;
using cumlrkt::printed;
using cumlrkt::read_one;
using cumlrkt::require_at_least;
using cumlrkt::require_some;
using cumlrkt::rows_of;
using cumlrkt::settle;
using cumlrkt::preprocessing::column_statistics;
using cumlrkt::preprocessing::first_non_finite;
using cumlrkt::preprocessing::scales_of;
using cumlrkt::preprocessing::standardized;
using cumlrkt::preprocessing::unstandardized;
using raftrkt::layout;

template <typename T>
matrix<T, int64_t> samples(const rr_view* x) {
  const matrix<T, int64_t> m =
      rows_of<T, int64_t>(x, "X", raftrkt::any_extent, raftrkt::any_extent);
  require_some("X", "row", m.n);
  return m;
}

template <typename T>
void require_finite(const raft::handle_t& handle, const matrix<T, int64_t>& x) {
  const int64_t i = first_non_finite(x.data, x.n * x.d, handle);
  if (i >= 0) {
    const T got = read_one(x.data, i, raft::resource::get_cuda_stream(handle));
    throw logic_error(
        std::format("X: row {}, column {}: expected a finite value, got {}",
                    i / x.d, i % x.d, printed(static_cast<double>(got))));
  }
}

struct statistics {
  explicit statistics(int64_t d, bool with_std, rmm::cuda_stream_view stream)
      : mean(static_cast<std::size_t>(d), stream),
        var(static_cast<std::size_t>(d), stream),
        scale(static_cast<std::size_t>(with_std ? d : 0), stream) {}
  rmm::device_uvector<double> mean;
  rmm::device_uvector<double> var;
  rmm::device_uvector<double> scale;
};

struct fit_request {
  rr_deposit* out;
  bool with_std;
  int64_t scratch_elements;
  int64_t* n_samples;
};

template <typename T>
void fit_into(const raft::handle_t& handle, const rr_view* view,
              const fit_request& r) {
  const matrix<T, int64_t> x = samples<T>(view);
  require_some("X", "column", x.d);
  if (x.d > most_index) {
    throw logic_error(
        std::format("X: extent {} does not fit the index type", x.d));
  }
  require_at_least("scratch-elements", r.scratch_elements, 1);
  require_finite(handle, x);
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  auto stats = std::make_unique<statistics>(x.d, r.with_std, stream);
  {
    const int64_t chunk = std::clamp<int64_t>(r.scratch_elements / x.n, 1, x.d);
    rmm::device_uvector<double> scratch(
        static_cast<std::size_t>((x.n + 1) * chunk), stream);
    column_statistics(handle, x.data, x.n, x.d, chunk, scratch.data(),
                      scratch.data() + (x.n * chunk), stats->mean.data(),
                      stats->var.data());
    if (r.with_std) {
      scales_of(handle, stats->mean.data(), stats->var.data(),
                stats->scale.data(), x.n, x.d);
    }
    settle(handle);
  }
  const std::initializer_list<rr_view> all{raftrkt::region(stats->mean),
                                           raftrkt::region(stats->var),
                                           raftrkt::region(stats->scale)};
  const std::initializer_list<rr_view> mean_only{raftrkt::region(stats->mean)};
  raftrkt::deposit(r.out, "out", handle, std::move(stats),
                   r.with_std ? all : mean_only);
  *r.n_samples = x.n;
}

const double* statistic(const rr_view* v, const char* name, int64_t d) {
  return v == nullptr ? nullptr : raftrkt::vector_data<double>(v, name, d);
}

struct statistics_views {
  const rr_view* mean;
  const rr_view* scale;
};

template <typename T, bool Inverse>
void apply_into(const raft::handle_t& handle, const rr_view* view,
                statistics_views stats, rr_deposit* out, int64_t n_features) {
  const matrix<T, int64_t> x = samples<T>(view);
  if (x.d != n_features) {
    throw logic_error(
        std::format("X: expected {} columns, got {}", n_features, x.d));
  }
  const double* mean = statistic(stats.mean, "mean", n_features);
  const double* scale = statistic(stats.scale, "scale", n_features);
  require_finite(handle, x);
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  auto result = std::make_unique<rmm::device_uvector<T>>(
      static_cast<std::size_t>(x.n * x.d), stream);
  if constexpr (Inverse) {
    unstandardized(handle, x.data, mean, scale, result->data(), x.n, x.d);
  } else {
    standardized(handle, x.data, mean, scale, result->data(), x.n, x.d);
  }
  const std::initializer_list<rr_view> regions{
      raftrkt::region(result->data(), x.n, x.d, layout::row_major)};
  settle(handle);
  raftrkt::deposit(out, "out", handle, std::move(result), regions);
}

template <bool Inverse>
int applied(void* handle, const rr_view* x, const rr_view* mean,
            const rr_view* scale, rr_deposit* out, int64_t n_features) {
  return guarded(handle, {{"X", x}, {"mean", mean}, {"scale", scale}},
                 [&](const raft::handle_t& h) {
                   rr_deposit* const into = raftrkt::require(out, "out");
                   on_floats(x, "X", [&](auto tag) {
                     apply_into<decltype(tag), Inverse>(h, x, {mean, scale},
                                                        into, n_features);
                   });
                 });
}

}  // namespace

extern "C" {

int cuml_preprocessing_standard_scaler_fit(
    void* handle, const rr_view* x, rr_deposit* out, int32_t /*with_mean*/,
    int32_t with_std, int64_t scratch_elements, int64_t* n_samples) {
  return guarded(handle, {{"X", x}}, [&](const raft::handle_t& h) {
    const fit_request r{raftrkt::require(out, "out"), with_std != 0,
                        scratch_elements,
                        raftrkt::require(n_samples, "n-samples")};
    on_floats(x, "X", [&](auto tag) { fit_into<decltype(tag)>(h, x, r); });
  });
}

int cuml_preprocessing_standard_scaler_transform(void* handle, const rr_view* x,
                                                 const rr_view* mean,
                                                 const rr_view* scale,
                                                 rr_deposit* out,
                                                 int64_t n_features) {
  return applied<false>(handle, x, mean, scale, out, n_features);
}

int cuml_preprocessing_standard_scaler_inverse_transform(
    void* handle, const rr_view* x, const rr_view* mean, const rr_view* scale,
    rr_deposit* out, int64_t n_features) {
  return applied<true>(handle, x, mean, scale, out, n_features);
}

CUMLRKT_JOB(cuml_preprocessing_standard_scaler_fit)
CUMLRKT_JOB(cuml_preprocessing_standard_scaler_transform)
CUMLRKT_JOB(cuml_preprocessing_standard_scaler_inverse_transform)
}
