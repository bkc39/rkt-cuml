#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuml/cluster/dbscan.hpp>
#include <cuml/common/distance_type.hpp>
#include <format>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <string>
#include <utility>

#include "cumlrkt/cluster.h"
#include "cumlrkt/common.hpp"
#include "dbscan_device.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "support.hpp"

namespace {

using cumlrkt::logic_error;
using cumlrkt::cluster::core_count;
using cumlrkt::cluster::first_unnormable_row;
using cumlrkt::cluster::matrix;
using cumlrkt::cluster::on_device;
using cumlrkt::cluster::printed;
using cumlrkt::cluster::require_at_least;
using cumlrkt::cluster::require_positive;
using cumlrkt::cluster::require_some;
using cumlrkt::cluster::rows_of;
using cumlrkt::cluster::settle;
using raftrkt::any_extent;
using Distance = ML::distance::DistanceType;

constexpr std::size_t megabyte = 1000000;
constexpr std::size_t percent = 100;
constexpr std::size_t usable_percent = 80;

struct fit_arrays {
  const rr_view* x;
  const rr_view* sample_weight;
  rr_deposit* out;
};

struct fit_options {
  double eps;
  int32_t min_samples;
  int32_t metric;
  int32_t out_dtype;
  int64_t max_mbytes_per_batch;
};

Distance distance_of(int32_t metric) {
  switch (metric) {
    case CUML_CLUSTER_DBSCAN_METRIC_EUCLIDEAN:
      return Distance::L2SqrtExpanded;
    case CUML_CLUSTER_DBSCAN_METRIC_COSINE:
      return Distance::CosineExpanded;
    case CUML_CLUSTER_DBSCAN_METRIC_PRECOMPUTED:
      return Distance::Precomputed;
    default:
      throw logic_error("metric: expected euclidean, cosine or precomputed");
  }
}

template <typename Index>
struct batch_bytes {
  std::size_t per_row;
  std::size_t fixed;

  explicit batch_bytes(std::size_t n)
      : per_row((n * sizeof(bool)) + ((n + 2) * sizeof(Index))),
        fixed(n * (sizeof(Index) + sizeof(bool))) {}

  [[nodiscard]] std::size_t rows(std::size_t mbytes) const {
    return ((mbytes * megabyte) - fixed) / per_row;
  }

  [[nodiscard]] std::size_t least_mbytes() const {
    return (fixed + per_row + megabyte - 1) / megabyte;
  }
};

template <typename T>
std::size_t automatic_mbytes(std::size_t n, std::size_t d) {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  raftrkt::cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes),
                      "cudaMemGetInfo");
  const std::size_t dataset = n * d * sizeof(T);
  const std::size_t usable =
      ((usable_percent * total_bytes) / percent) - dataset;
  return static_cast<std::size_t>(static_cast<double>(usable) /
                                  static_cast<double>(megabyte));
}

template <typename T, typename Index>
void require_batch(int64_t mbytes, std::size_t n, std::size_t d) {
  require_at_least("max-mbytes-per-batch", mbytes, 0);
  const std::size_t most = std::numeric_limits<std::size_t>::max() / megabyte;
  if (static_cast<uint64_t>(mbytes) > most) {
    throw logic_error("max-mbytes-per-batch: expected at most " +
                      std::to_string(most) + ", got " + std::to_string(mbytes));
  }
  const batch_bytes<Index> batch(n);
  const std::size_t least = batch.least_mbytes();
  if (mbytes > 0 && static_cast<std::size_t>(mbytes) < least) {
    throw logic_error("max-mbytes-per-batch: expected at least " +
                      std::to_string(least) + " for " + std::to_string(n) +
                      " rows, got " + std::to_string(mbytes));
  }
  if (mbytes == 0 && batch.rows(automatic_mbytes<T>(n, d)) == 0) {
    throw logic_error(
        "max-mbytes-per-batch: the automatic size holds no batch of " +
        std::to_string(n) + " rows; pass at least " + std::to_string(least));
  }
}

template <typename Index>
constexpr int64_t most_rows() {
  constexpr std::size_t label = sizeof(Index);
  constexpr std::size_t batch_rows =
      (std::numeric_limits<std::size_t>::max() - (2 * label)) /
      ((2 * label) + 2);
  return static_cast<int64_t>(std::min<std::size_t>(
      static_cast<std::size_t>(std::numeric_limits<Index>::max()) - 2,
      batch_rows));
}

template <typename Index>
void require_rows(int64_t n) {
  if (n <= most_rows<Index>()) {
    return;
  }
  if (sizeof(Index) == sizeof(int32_t)) {
    throw logic_error("X: " + std::to_string(n) +
                      " rows do not fit int32 labels; use int64");
  }
  throw logic_error("X: " + std::to_string(n) +
                    " rows overflow cuML's batch size arithmetic");
}

template <typename T, typename Index>
matrix<T, Index> checked_x(const rr_view* view, Distance metric) {
  const matrix<T, int64_t> x =
      rows_of<T, int64_t>(view, "X", any_extent, any_extent);
  require_some("X", "row", x.n);
  require_some("X", "column", x.d);
  if (metric == Distance::Precomputed && x.n != x.d) {
    throw logic_error("X: expected a square distance matrix, got " +
                      std::to_string(x.n) + "x" + std::to_string(x.d));
  }
  if (sizeof(Index) == sizeof(int32_t) &&
      x.n * x.d > std::numeric_limits<int32_t>::max()) {
    throw logic_error("X: " + std::to_string(x.n) + "x" + std::to_string(x.d) +
                      " elements do not fit int32 labels; use int64");
  }
  require_rows<Index>(x.n);
  return {x.data, static_cast<Index>(x.n), static_cast<Index>(x.d)};
}

template <typename T>
T* weights_of(const rr_view* sample_weight, int64_t n) {
  if (sample_weight == nullptr) {
    return nullptr;
  }
  return raftrkt::vector_data<T, int64_t>(sample_weight, "sample-weight", n);
}

template <typename T>
T eps_of(double eps) {
  require_positive("eps", eps);
  const double most = std::numeric_limits<T>::max();
  if (eps > most) {
    throw logic_error(std::format("eps: expected at most {}, got {}",
                                  printed(most), printed(eps)));
  }
  const auto narrowed = static_cast<T>(eps);
  require_positive("eps", eps, narrowed > 0);
  return narrowed;
}

template <typename T>
rmm::device_uvector<T> cosine_copy(const T* x, int64_t n, int64_t d,
                                   rmm::cuda_stream_view stream) {
  rmm::device_uvector<T> copy(static_cast<std::size_t>(n * d), stream);
  raftrkt::cuda_check(cudaMemcpyAsync(copy.data(), x, sizeof(T) * copy.size(),
                                      cudaMemcpyDeviceToDevice, stream.value()),
                      "cudaMemcpyAsync");
  const int64_t row = first_unnormable_row(copy.data(), n, d, stream);
  if (row >= 0) {
    throw logic_error("X: row " + std::to_string(row) +
                      " has a norm outside float range, so no cosine distance");
  }
  return copy;
}

template <typename Index>
struct dbscan_outputs {
  rmm::device_uvector<Index> labels;
  rmm::device_uvector<Index> core;
};

template <typename T, typename Index>
void fit(const raft::handle_t& handle, const fit_arrays& arrays,
         const fit_options& options) {
  const Distance metric = distance_of(options.metric);
  const matrix<T, Index> x = checked_x<T, Index>(arrays.x, metric);
  T* w = weights_of<T>(arrays.sample_weight, x.n);
  const T eps = eps_of<T>(options.eps);
  require_at_least("min-samples", options.min_samples, 1);
  const auto n = static_cast<std::size_t>(x.n);
  require_batch<T, Index>(options.max_mbytes_per_batch, n,
                          static_cast<std::size_t>(x.d));
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  std::optional<rmm::device_uvector<T>> copy;
  if (metric == Distance::CosineExpanded) {
    copy = cosine_copy(x.data, x.n, x.d, stream);
  }
  T* input = copy ? copy->data() : x.data;
  auto outputs = std::make_unique<dbscan_outputs<Index>>(
      dbscan_outputs<Index>{rmm::device_uvector<Index>(n, stream),
                            rmm::device_uvector<Index>(n, stream)});
  ML::Dbscan::fit(handle, input, x.n, x.d, eps, options.min_samples, metric,
                  outputs->labels.data(), outputs->core.data(), w,
                  static_cast<std::size_t>(options.max_mbytes_per_batch),
                  ML::Dbscan::BRUTE_FORCE, rapids_logger::level_enum::warn,
                  false);
  const int64_t cores = core_count(outputs->core.data(), x.n, stream);
  const std::initializer_list<rr_view> regions{
      raftrkt::region(outputs->labels),
      raftrkt::region(outputs->core.data(), cores)};
  settle(handle);
  copy.reset();
  raftrkt::deposit(arrays.out, "out", handle, std::move(outputs), regions);
}

template <typename T>
void fit_labels(const raft::handle_t& handle, const fit_arrays& arrays,
                const fit_options& options) {
  if (options.out_dtype == RR_DTYPE_INT32) {
    fit<T, int32_t>(handle, arrays, options);
  } else if (options.out_dtype == RR_DTYPE_INT64) {
    fit<T, int64_t>(handle, arrays, options);
  } else {
    throw logic_error("out-dtype: expected int32 or int64");
  }
}

}  // namespace

extern "C" {

int cuml_cluster_dbscan_fit(void* handle, const rr_view* x,
                            const rr_view* sample_weight, rr_deposit* out,
                            double eps, int32_t min_samples, int32_t metric,
                            int32_t out_dtype, int64_t max_mbytes_per_batch) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [&] {
    const fit_arrays arrays{x, sample_weight, raftrkt::require(out, "out")};
    const fit_options options{eps, min_samples, metric, out_dtype,
                              max_mbytes_per_batch};
    on_device(handle, {{"X", x}, {"sample-weight", sample_weight}}, x, "X",
              [&](const raft::handle_t& h, auto tag) {
                fit_labels<decltype(tag)>(h, arrays, options);
              });
  });
}

void cuml_cluster_dbscan_fit_job(void* args, size_t args_len,
                                 rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_dbscan_fit>(args, args_len, outcome,
                                             cumlrkt::last_error_slot());
}
}
