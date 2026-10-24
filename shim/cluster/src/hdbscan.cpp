#include <array>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cuml/cluster/hdbscan.hpp>
#include <cuml/common/distance_type.hpp>
#include <format>
#include <limits>
#include <memory>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <string>
#include <utility>

#include "cumlrkt/cluster.h"
#include "cumlrkt/common.hpp"
#include "cumlrkt/model.hpp"
#include "hdbscan_model.hpp"
#include "hdbscan_support.hpp"
#include "linkage_device.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"
#include "support.hpp"

namespace {

using cumlrkt::logic_error;
using cumlrkt::cluster::ascending_pairs;
using cumlrkt::cluster::hdbscan_model;
using cumlrkt::cluster::hdbscan_shape;
using cumlrkt::cluster::in_device_scope;
using cumlrkt::cluster::matrix;
using cumlrkt::cluster::printed;
using cumlrkt::cluster::require_at_least;
using cumlrkt::cluster::require_non_negative;
using cumlrkt::cluster::require_positive;
using cumlrkt::cluster::require_some;
using cumlrkt::cluster::rows_of;
using cumlrkt::cluster::settle;
using cumlrkt::cluster::hdbscan_limits::most_index;
using cumlrkt::cluster::hdbscan_limits::require_bounded_rows;
using cumlrkt::cluster::hdbscan_limits::require_index;
using raftrkt::any_extent;

namespace common = ML::HDBSCAN::Common;

constexpr int64_t least_rows = 2;
constexpr int64_t least_cluster_size = 2;
constexpr int64_t most_rows = int64_t{1} << 30;
constexpr int64_t most_min_samples = 1023;
constexpr double most_float = FLT_MAX;

struct options {
  int64_t min_cluster_size;
  int64_t min_samples;
  double cluster_selection_epsilon;
  int64_t max_cluster_size;
  double alpha;
  int32_t cluster_selection_method;
  int32_t allow_single_cluster;
};

void require_at_most(const char* what, int64_t got, int64_t most) {
  if (got > most) {
    throw logic_error(
        std::format("{}: expected at most {}, got {}", what, most, got));
  }
}

void require_float(const char* what, double got) {
  if (got > most_float) {
    throw logic_error(std::format("{}: expected at most {}, got {}", what,
                                  printed(most_float), printed(got)));
  }
}

matrix<float, int64_t> checked_x(const rr_view* view) {
  const matrix<float, int64_t> x =
      rows_of<float, int64_t>(view, "X", any_extent, any_extent);
  if (x.n < least_rows) {
    throw logic_error(
        std::format("X: expected at least {} rows, got {}", least_rows, x.n));
  }
  require_some("X", "column", x.d);
  require_index("X", x.n, x.d);
  if (x.n > most_rows) {
    throw logic_error(
        std::format("X: expected at most {} rows, got {}", most_rows, x.n));
  }
  return x;
}

void require_options(const options& o, int64_t n) {
  require_at_least("min-cluster-size", o.min_cluster_size, least_cluster_size);
  require_at_most("min-cluster-size", o.min_cluster_size, most_index);
  require_at_least("min-samples", o.min_samples, 1);
  require_at_most("min-samples", o.min_samples, most_min_samples);
  if (o.min_samples > n) {
    throw logic_error(std::format("X: {} samples are fewer than min-samples {}",
                                  n, o.min_samples));
  }
  require_non_negative("cluster-selection-epsilon",
                       o.cluster_selection_epsilon);
  require_float("cluster-selection-epsilon", o.cluster_selection_epsilon);
  require_at_least("max-cluster-size", o.max_cluster_size, 0);
  require_at_most("max-cluster-size", o.max_cluster_size, most_index);
  require_positive("alpha", o.alpha);
  require_float("alpha", o.alpha);
  if (o.cluster_selection_method != CUML_CLUSTER_HDBSCAN_EOM &&
      o.cluster_selection_method != CUML_CLUSTER_HDBSCAN_LEAF) {
    throw logic_error("cluster-selection-method: expected eom or leaf");
  }
}

common::HDBSCANParams params_of(const options& o) {
  common::HDBSCANParams p;
  p.min_samples = static_cast<int>(o.min_samples);
  p.min_cluster_size = static_cast<int>(o.min_cluster_size);
  p.max_cluster_size = static_cast<int>(o.max_cluster_size);
  p.cluster_selection_epsilon = static_cast<float>(o.cluster_selection_epsilon);
  p.allow_single_cluster = o.allow_single_cluster != 0;
  p.alpha = static_cast<float>(o.alpha);
  p.cluster_selection_method =
      o.cluster_selection_method == CUML_CLUSTER_HDBSCAN_LEAF
          ? common::CLUSTER_SELECTION_METHOD::LEAF
          : common::CLUSTER_SELECTION_METHOD::EOM;
  return p;
}

struct plain_outputs {
  rmm::device_uvector<float> probabilities;
  rmm::device_uvector<int64_t> children;
  rmm::device_uvector<float> distances;
  rmm::device_uvector<int64_t> sizes;
  rmm::device_uvector<int64_t> sources;
  rmm::device_uvector<int64_t> destinations;
  rmm::device_uvector<float> weights;
};

std::unique_ptr<plain_outputs> plain_for(int64_t n,
                                         rmm::cuda_stream_view stream) {
  const auto rows = static_cast<std::size_t>(n);
  return std::make_unique<plain_outputs>(
      plain_outputs{rmm::device_uvector<float>(rows, stream),
                    rmm::device_uvector<int64_t>(2 * rows, stream),
                    rmm::device_uvector<float>(rows, stream),
                    rmm::device_uvector<int64_t>(rows, stream),
                    rmm::device_uvector<int64_t>(rows - 1, stream),
                    rmm::device_uvector<int64_t>(rows - 1, stream),
                    rmm::device_uvector<float>(rows - 1, stream)});
}

void fit(const raft::handle_t& handle, const rr_view* view,
         rr_deposit* model_out, rr_deposit* out, const options& o,
         int64_t* opaque_bytes) {
  const matrix<float, int64_t> x = checked_x(view);
  require_options(o, x.n);
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  require_bounded_rows("X", x, stream);
  const auto n = static_cast<std::size_t>(x.n);
  std::unique_ptr<plain_outputs> plain = plain_for(x.n, stream);
  rmm::device_uvector<float> core_dists(n, stream);
  rmm::device_uvector<int64_t> labels(n, stream);
  common::hdbscan_output<int64_t, float> found(
      handle, static_cast<int>(x.n), labels.data(), plain->probabilities.data(),
      plain->children.data(), plain->sizes.data(), plain->distances.data(),
      plain->sources.data(), plain->destinations.data(), plain->weights.data());
  common::HDBSCANParams params = params_of(o);
  ML::hdbscan(handle, x.data, n, static_cast<std::size_t>(x.d),
              ML::distance::DistanceType::L2SqrtExpanded, params, found,
              core_dists.data());
  auto m = std::make_unique<hdbscan_model>(
      handle, hdbscan_shape{x.n, x.d, o.min_samples}, std::move(core_dists),
      std::move(labels), found);
  ascending_pairs(plain->children.data(), x.n - 1, stream);
  settle(handle);
  const int64_t edges = m->tree.get_n_edges();
  const std::array<rr_view, 7> model_regions{
      raftrkt::region(m->labels),
      raftrkt::region(m->tree.get_parents(), edges),
      raftrkt::region(m->tree.get_children(), edges),
      raftrkt::region(m->tree.get_lambdas(), edges),
      raftrkt::region(m->tree.get_sizes(), edges),
      raftrkt::region(m->stabilities),
      raftrkt::region(m->inverse_label_map)};
  const int64_t merges = x.n - 1;
  const std::initializer_list<rr_view> plain_regions{
      raftrkt::region(plain->probabilities),
      raftrkt::region(plain->children.data(), merges, 2,
                      raftrkt::layout::row_major),
      raftrkt::region(plain->distances.data(), merges),
      raftrkt::region(plain->sizes.data(), merges),
      raftrkt::region(plain->sources),
      raftrkt::region(plain->destinations),
      raftrkt::region(plain->weights)};
  const int64_t bytes = m->opaque_bytes();
  cumlrkt::publish_model(model_out, "model-out", handle, std::move(m),
                         model_regions[0], model_regions[1], model_regions[2],
                         model_regions[3], model_regions[4], model_regions[5],
                         model_regions[6]);
  raftrkt::deposit(out, "out", handle, std::move(plain), plain_regions);
  *opaque_bytes = bytes;
}

}  // namespace

extern "C" {

int cuml_cluster_hdbscan_fit(void* handle, const rr_view* x,
                             rr_deposit* model_out, rr_deposit* out,
                             int64_t min_cluster_size, int64_t min_samples,
                             double cluster_selection_epsilon,
                             int64_t max_cluster_size, double alpha,
                             int32_t cluster_selection_method,
                             int32_t allow_single_cluster,
                             int64_t* opaque_bytes) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [&] {
    const options o{min_cluster_size,    min_samples, cluster_selection_epsilon,
                    max_cluster_size,    alpha,       cluster_selection_method,
                    allow_single_cluster};
    in_device_scope(handle, {{"X", x}}, [&](const raft::handle_t& h) {
      rr_deposit* const model_into = raftrkt::require(model_out, "model-out");
      rr_deposit* const into = raftrkt::require(out, "out");
      int64_t* const bytes = raftrkt::require(opaque_bytes, "opaque-bytes");
      fit(h, x, model_into, into, o, bytes);
    });
  });
}

void cuml_cluster_hdbscan_fit_job(void* args, size_t args_len,
                                  rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_hdbscan_fit>(args, args_len, outcome,
                                              cumlrkt::last_error_slot());
}
}
