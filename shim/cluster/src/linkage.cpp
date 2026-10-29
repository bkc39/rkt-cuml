#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cuml/cluster/linkage.hpp>
#include <cuml/common/distance_type.hpp>
#include <initializer_list>
#include <memory>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <string>
#include <utility>

#include "cumlrkt/cluster.h"
#include "cumlrkt/common.hpp"
#include "cumlrkt/support.hpp"
#include "linkage_device.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"

namespace {

using cumlrkt::logic_error;
using cumlrkt::matrix;
using cumlrkt::most_index;
using cumlrkt::on_float32_device;
using cumlrkt::require_at_least;
using cumlrkt::require_bounded_rows;
using cumlrkt::require_index;
using cumlrkt::require_some;
using cumlrkt::rows_of;
using cumlrkt::settle;
using cumlrkt::cluster::ascending_pairs;
using raftrkt::any_extent;

constexpr int64_t most_pairwise_rows = 46340;
constexpr int64_t most_log2_rows = 30;
constexpr int64_t most_c = most_index - most_log2_rows;
constexpr int64_t least_neighbours = 2;

struct fit_options {
  int64_t n_clusters;
  int32_t metric;
  int32_t connectivity;
  int32_t c;
};

bool knn_of(int32_t connectivity) {
  switch (connectivity) {
    case CUML_CLUSTER_LINKAGE_CONNECTIVITY_PAIRWISE:
      return false;
    case CUML_CLUSTER_LINKAGE_CONNECTIVITY_KNN:
      return true;
    default:
      throw logic_error("connectivity: expected knn or pairwise");
  }
}

void require_euclidean(int32_t metric) {
  if (metric != CUML_CLUSTER_LINKAGE_METRIC_EUCLIDEAN) {
    throw logic_error("metric: expected euclidean");
  }
}

matrix<float, int64_t> checked_x(const rr_view* view) {
  const matrix<float, int64_t> x =
      rows_of<float, int64_t>(view, "X", any_extent, any_extent);
  require_some("X", "row", x.n);
  require_some("X", "column", x.d);
  require_index("X", x.n, x.d);
  return x;
}

void require_clusters(int64_t n_clusters, int64_t n) {
  require_at_least("n-clusters", n_clusters, 1);
  if (n_clusters > n) {
    throw logic_error("X: " + std::to_string(n) +
                      " samples are fewer than the " +
                      std::to_string(n_clusters) + " clusters");
  }
}

void require_c(int32_t c) {
  if (c > most_c) {
    throw logic_error("c: expected at most " + std::to_string(most_c) +
                      ", got " + std::to_string(c));
  }
}

int64_t log2_of(int64_t n) {
  return std::bit_width(static_cast<uint64_t>(n)) - 1;
}

int32_t clamped_c(int64_t n, int32_t c) {
  return static_cast<int32_t>(std::max<int64_t>(c, -log2_of(n)));
}

int64_t neighbours(int64_t n, int32_t c) {
  return std::min(n, std::max(least_neighbours, log2_of(n) + c));
}

void require_pairwise_rows(int64_t n) {
  if (n > most_pairwise_rows) {
    throw logic_error("X: pairwise connectivity holds at most " +
                      std::to_string(most_pairwise_rows) + " rows, got " +
                      std::to_string(n) + "; use knn");
  }
}

void require_knn_graph(int64_t n, int32_t c) {
  const int64_t k = neighbours(n, c);
  const int64_t entries = 2 * n * k;
  if (entries > most_index) {
    throw logic_error(
        "X: " + std::to_string(n) + " rows with " + std::to_string(k) +
        " neighbours need " + std::to_string(entries) +
        " knn graph entries, more than " + std::to_string(most_index));
  }
}

struct linkage_outputs {
  rmm::device_uvector<int32_t> labels;
  rmm::device_uvector<int32_t> children;
};

void fit(const raft::handle_t& handle, const rr_view* view, rr_deposit* out,
         const fit_options& options) {
  const bool knn = knn_of(options.connectivity);
  require_euclidean(options.metric);
  const matrix<float, int64_t> x = checked_x(view);
  require_clusters(options.n_clusters, x.n);
  require_c(options.c);
  const int32_t c = clamped_c(x.n, options.c);
  if (knn) {
    require_knn_graph(x.n, c);
  } else {
    require_pairwise_rows(x.n);
  }
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  require_bounded_rows("X", x, stream);
  const auto n = static_cast<std::size_t>(x.n);
  auto outputs = std::make_unique<linkage_outputs>(
      linkage_outputs{rmm::device_uvector<int32_t>(n, stream),
                      rmm::device_uvector<int32_t>(2 * (n - 1), stream)});
  ML::linkage::single_linkage(
      handle, x.data, static_cast<int>(x.n), static_cast<int>(x.d),
      static_cast<std::size_t>(options.n_clusters),
      ML::distance::DistanceType::L2SqrtExpanded, outputs->children.data(),
      outputs->labels.data(), knn, c);
  ascending_pairs(outputs->children.data(), x.n - 1, stream);
  const std::initializer_list<rr_view> regions{
      raftrkt::region(outputs->labels),
      raftrkt::region(outputs->children.data(), x.n - 1, 2,
                      raftrkt::layout::row_major)};
  settle(handle);
  raftrkt::deposit(out, "out", handle, std::move(outputs), regions);
}

}  // namespace

extern "C" {

int cuml_cluster_single_linkage_fit(void* handle, const rr_view* x,
                                    rr_deposit* out, int64_t n_clusters,
                                    int32_t metric, int32_t connectivity,
                                    int32_t c) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [&] {
    rr_deposit* const into = raftrkt::require(out, "out");
    const fit_options options{n_clusters, metric, connectivity, c};
    on_float32_device(handle, {{"X", x}}, x, "X", [&](const raft::handle_t& h) {
      fit(h, x, into, options);
    });
  });
}

void cuml_cluster_single_linkage_fit_job(void* args, size_t args_len,
                                         rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_single_linkage_fit>(
      args, args_len, outcome, cumlrkt::last_error_slot());
}
}
