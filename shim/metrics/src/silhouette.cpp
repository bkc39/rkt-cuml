#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuml/common/distance_type.hpp>
#include <cuml/metrics/metrics.hpp>
#include <format>
#include <initializer_list>
#include <memory>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <string>
#include <utility>

#include "checks.hpp"
#include "cumlrkt/metrics.h"
#include "cumlrkt/support.hpp"
#include "labels.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"

namespace {

using cumlrkt::first_zero_row;
using cumlrkt::guarded;
using cumlrkt::logic_error;
using cumlrkt::matrix;
using cumlrkt::on_floats;
using cumlrkt::require_at_least;
using cumlrkt::require_bounded_rows;
using cumlrkt::require_index;
using cumlrkt::require_some;
using cumlrkt::rows_of;
using cumlrkt::settle;
using cumlrkt::metrics::dense_codes;
using cumlrkt::metrics::dense_labels;
using cumlrkt::metrics::labels_of;
using cumlrkt::metrics::most_distinct_labels;
using cumlrkt::metrics::on_labels;
using cumlrkt::metrics::require_entries;
using ML::distance::DistanceType;

constexpr int64_t least_rows = 3;
constexpr int64_t least_labels = 2;

DistanceType distance_of(int32_t metric) {
  switch (metric) {
    case CUML_METRICS_METRIC_EUCLIDEAN:
      return DistanceType::L2SqrtUnexpanded;
    case CUML_METRICS_METRIC_SQEUCLIDEAN:
      return DistanceType::L2Expanded;
    case CUML_METRICS_METRIC_MANHATTAN:
      return DistanceType::L1;
    case CUML_METRICS_METRIC_COSINE:
      return DistanceType::CosineExpanded;
    case CUML_METRICS_METRIC_CHEBYSHEV:
      return DistanceType::Linf;
    default:
      throw logic_error(
          "metric: expected euclidean, sqeuclidean, manhattan, cosine or "
          "chebyshev");
  }
}

struct request {
  DistanceType distance;
  int64_t chunk_size;
};

template <typename T>
struct checked {
  matrix<T, int64_t> x;
  rmm::device_uvector<int32_t> codes;
  int64_t k;
  int64_t chunk;
};

void require_chunk(int64_t chunk_size, int64_t n) {
  if (std::min(chunk_size, n) > most_distinct_labels) {
    throw logic_error(std::format("chunk-size: expected at most {}, got {}",
                                  most_distinct_labels, chunk_size));
  }
}

void require_shape(int64_t n, int64_t d) {
  require_some("X", "column", d);
  require_index("X", n, d);
  if (n < least_rows) {
    throw logic_error(
        std::format("X: expected at least {} rows, got {}", least_rows, n));
  }
}

void require_label_count(int64_t k, int64_t n) {
  if (k < least_labels) {
    throw logic_error(std::format(
        "labels: expected at least {} distinct labels, got {}", least_labels,
        k));
  }
  if (k > n - 1) {
    throw logic_error(std::format(
        "labels: expected at most {} distinct labels, got {}", n - 1, k));
  }
  require_entries(
      std::format("labels: {} rows with {} distinct labels", n, k), n, k);
}

template <typename T>
void require_rows(const matrix<T, int64_t>& x, DistanceType distance,
                  rmm::cuda_stream_view stream) {
  require_bounded_rows("X", x, stream);
  if (distance == DistanceType::CosineExpanded) {
    const int64_t row = first_zero_row(x.data, x.n, x.d, stream);
    if (row >= 0) {
      throw logic_error(
          std::format("X: row {} is zero, so no cosine distance", row));
    }
  }
}

template <typename T>
checked<T> check(const raft::handle_t& handle, const rr_view* x_view,
                 const rr_view* labels, const request& r) {
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  checked<T> c{rows_of<T, int64_t>(x_view, "X", raftrkt::any_extent,
                                   raftrkt::any_extent),
               rmm::device_uvector<int32_t>(0, stream), 0, 0};
  on_labels(labels, "labels", [&](auto l) {
    using L = decltype(l);
    const L* y = labels_of<L>(labels, "labels", c.x.n);
    require_shape(c.x.n, c.x.d);
    require_chunk(r.chunk_size, c.x.n);
    dense_labels<L> found = dense_codes(y, c.x.n, stream);
    c.codes = std::move(found.codes);
    c.k = found.k;
  });
  require_label_count(c.k, c.x.n);
  require_rows(c.x, r.distance, stream);
  c.chunk = std::min(r.chunk_size, c.x.n);
  return c;
}

template <typename T>
double score_of(const raft::handle_t& handle, checked<T>& c, T* scores,
                DistanceType distance) {
  const double score = ML::Metrics::Batched::silhouette_score(
      handle, c.x.data, static_cast<int>(c.x.n), static_cast<int>(c.x.d),
      c.codes.data(), static_cast<int>(c.k), scores,
      static_cast<int>(c.chunk), distance);
  settle(handle);
  return score;
}

template <typename Run>
void on_request(const rr_view* x, int32_t metric, int64_t chunk_size,
                Run&& run) {
  const request r{distance_of(metric), chunk_size};
  require_at_least("chunk-size", chunk_size, 1);
  on_floats(x, "X", [&](auto tag) { run(r, tag); });
}

template <typename T>
void samples_into(const raft::handle_t& handle, const rr_view* x,
                  const rr_view* labels, rr_deposit* out, const request& r,
                  double* score) {
  checked<T> c = check<T>(handle, x, labels, r);
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  auto scores = std::make_unique<rmm::device_uvector<T>>(
      static_cast<std::size_t>(c.x.n), stream);
  *score = score_of(handle, c, scores->data(), r.distance);
  const std::initializer_list<rr_view> regions{raftrkt::region(*scores)};
  raftrkt::deposit(out, "out", handle, std::move(scores), regions);
}

}  // namespace

extern "C" {

int cuml_metrics_silhouette_score(void* handle, const rr_view* x,
                                  const rr_view* labels, int32_t metric,
                                  int64_t chunk_size, double* score) {
  return guarded(
      handle, {{"X", x}, {"labels", labels}}, [&](const raft::handle_t& h) {
        double* const into = raftrkt::require(score, "score");
        on_request(x, metric, chunk_size, [&](const request& r, auto tag) {
          using T = decltype(tag);
          checked<T> c = check<T>(h, x, labels, r);
          *into = score_of<T>(h, c, nullptr, r.distance);
        });
      });
}

int cuml_metrics_silhouette_samples(void* handle, const rr_view* x,
                                    const rr_view* labels, rr_deposit* out,
                                    int32_t metric, int64_t chunk_size,
                                    double* score) {
  return guarded(
      handle, {{"X", x}, {"labels", labels}}, [&](const raft::handle_t& h) {
        rr_deposit* const into = raftrkt::require(out, "out");
        double* const mean = raftrkt::require(score, "score");
        on_request(x, metric, chunk_size, [&](const request& r, auto tag) {
          samples_into<decltype(tag)>(h, x, labels, into, r, mean);
        });
      });
}

CUMLRKT_JOB(cuml_metrics_silhouette_score)
CUMLRKT_JOB(cuml_metrics_silhouette_samples)
}
