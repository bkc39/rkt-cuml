#include <algorithm>
#include <cstdint>
#include <cuml/common/distance_type.hpp>
#include <cuml/metrics/metrics.hpp>
#include <format>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <string>

#include "checks.hpp"
#include "cumlrkt/metrics.h"
#include "cumlrkt/support.hpp"
#include "raftrkt/job.hpp"

namespace {

using cumlrkt::guarded;
using cumlrkt::logic_error;
using cumlrkt::matrix;
using cumlrkt::require_at_least;
using cumlrkt::require_bounded_rows;
using cumlrkt::require_index;
using cumlrkt::require_some;
using cumlrkt::rows_of;
using cumlrkt::settle;
using cumlrkt::metrics::require_entries;
using ML::distance::DistanceType;

struct options {
  int64_t n_neighbors;
  int32_t metric;
  int64_t batch_size;
};

void require_options(const options& o) {
  if (o.metric != CUML_METRICS_METRIC_EUCLIDEAN) {
    throw logic_error("metric: expected euclidean");
  }
  require_at_least("batch-size", o.batch_size, 1);
  require_at_least("n-neighbors", o.n_neighbors, 1);
}

void require_float32(const rr_view* view, const char* name) {
  const rr_view& v = *raftrkt::require(view, name);
  if (v.dtype != RR_DTYPE_FLOAT32) {
    throw logic_error(std::string(name) + ": expected float32, got " +
                      raftrkt::dtype_name(v.dtype));
  }
}

void require_neighbours(int64_t n, int64_t k) {
  if (k >= (n + 1) / 2) {
    throw logic_error(std::format(
        "n-neighbors: expected less than half the {} rows, got {}", n, k));
  }
  require_entries(
      std::format("n-neighbors: {} rows with {} neighbours each", n, k + 1), n,
      k + 1);
}

void require_batch(int64_t n, int64_t batch) {
  require_entries(
      std::format("batch-size: {} rows of {} distances", batch, n), batch, n);
}

double score(const raft::handle_t& handle, const rr_view* x_view,
             const rr_view* embedded_view, const options& o) {
  require_float32(x_view, "X");
  require_float32(embedded_view, "X-embedded");
  const matrix<float, int64_t> x = rows_of<float, int64_t>(
      x_view, "X", raftrkt::any_extent, raftrkt::any_extent);
  const matrix<float, int64_t> e = rows_of<float, int64_t>(
      embedded_view, "X-embedded", x.n, raftrkt::any_extent);
  require_some("X", "column", x.d);
  require_some("X-embedded", "column", e.d);
  require_neighbours(x.n, o.n_neighbors);
  const int64_t batch = std::min(o.batch_size, x.n);
  require_batch(x.n, batch);
  require_index("X", x.n, x.d);
  require_index("X-embedded", e.n, e.d);
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  require_bounded_rows("X", x, stream);
  require_bounded_rows("X-embedded", e, stream);
  const double found =
      ML::Metrics::trustworthiness_score<float,
                                         DistanceType::L2SqrtUnexpanded>(
          handle, x.data, e.data, static_cast<int>(x.n),
          static_cast<int>(x.d), static_cast<int>(e.d),
          static_cast<int>(o.n_neighbors), static_cast<int>(batch));
  settle(handle);
  return found;
}

}  // namespace

extern "C" {

int cuml_metrics_trustworthiness(void* handle, const rr_view* x,
                                 const rr_view* x_embedded,
                                 int64_t n_neighbors, int32_t metric,
                                 int64_t batch_size, double* score_out) {
  return guarded(handle, {{"X", x}, {"X-embedded", x_embedded}},
                 [&](const raft::handle_t& h) {
                   double* const into = raftrkt::require(score_out, "score");
                   const options o{n_neighbors, metric, batch_size};
                   require_options(o);
                   *into = score(h, x, x_embedded, o);
                 });
}

CUMLRKT_JOB(cuml_metrics_trustworthiness)
}
