#include <cuda_runtime_api.h>
#include <cuml/manifold/umapparams.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuml/common/distance_type.hpp>
#include <cuml/manifold/umap.hpp>
#include <format>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <raft/core/handle.hpp>
#include <raft/core/host_coo_matrix.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <utility>

#include "cumlrkt/manifold.h"
#include "cumlrkt/support.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"
#include "umap_device.hpp"

namespace {

using cumlrkt::guarded;
using cumlrkt::logic_error;
using cumlrkt::matrix;
using cumlrkt::most_float;
using cumlrkt::most_index;
using cumlrkt::on_labels;
using cumlrkt::printed;
using cumlrkt::read_one;
using cumlrkt::require_at_least;
using cumlrkt::require_at_most;
using cumlrkt::require_bounded_rows;
using cumlrkt::require_finite;
using cumlrkt::require_float;
using cumlrkt::require_some;
using cumlrkt::rows_of;
using cumlrkt::settle;
using cumlrkt::manifold::float_codes;
using cumlrkt::manifold::label_codes;
using raftrkt::any_extent;
using raftrkt::layout;
using DistanceType = ML::distance::DistanceType;

constexpr int64_t least_nn_descent_rows = 150;
constexpr int64_t least_transform_epochs = 3;
constexpr float least_float = std::numeric_limits<float>::min();
constexpr int64_t most_labels = int64_t{1} << 24;
constexpr std::size_t nn_descent_degree = 64;
constexpr std::size_t nn_descent_intermediate_degree = 128;
constexpr int most_serial_components = 512;
constexpr float transform_queue_size = 4.0F;
constexpr float target_weight = 0.5F;
constexpr float euclidean_p = 2.0F;

struct umap_request {
  int64_t n_neighbors;
  int64_t n_components;
  bool epochs_given;
  int64_t n_epochs;
  double learning_rate;
  double a;
  double b;
  int32_t init;
  int64_t negative_sample_rate;
  uint64_t random_state;
  bool deterministic;
  int32_t build_algo;
  int32_t metric;
};

DistanceType distance_of(int32_t metric) {
  switch (metric) {
    case CUML_MANIFOLD_METRIC_EUCLIDEAN:
      return DistanceType::L2SqrtExpanded;
    case CUML_MANIFOLD_METRIC_SQEUCLIDEAN:
      return DistanceType::L2Expanded;
    case CUML_MANIFOLD_METRIC_MANHATTAN:
      return DistanceType::L1;
    case CUML_MANIFOLD_METRIC_CHEBYSHEV:
      return DistanceType::Linf;
    case CUML_MANIFOLD_METRIC_COSINE:
      return DistanceType::CosineExpanded;
    default:
      throw logic_error(
          "metric: expected euclidean, sqeuclidean, manhattan, chebyshev or "
          "cosine");
  }
}

const char* metric_name(int32_t metric) {
  return metric == CUML_MANIFOLD_METRIC_MANHATTAN ? "manhattan" : "chebyshev";
}

void require_curve_value(const char* what, double got) {
  if (!std::isfinite(got) || got < least_float || got > most_float) {
    throw logic_error(
        std::format("{}: expected a finite float32 value above 0, got {}", what,
                    printed(got)));
  }
}

void require_shared(const umap_request& r) {
  static_cast<void>(distance_of(r.metric));
}

void require_numbers(const umap_request& r) {
  require_at_least("n-neighbors", r.n_neighbors, 2);
  require_at_most("n-neighbors", r.n_neighbors, most_index);
  require_at_least("n-components", r.n_components, 1);
  require_at_most("n-components", r.n_components, most_index);
  if (r.epochs_given) {
    require_at_least("n-epochs", r.n_epochs, 1);
    require_at_most("n-epochs", r.n_epochs, most_index);
  }
  if (!std::isfinite(r.learning_rate) || r.learning_rate <= 0) {
    throw logic_error(
        std::format("learning-rate: expected a finite value above 0, got {}",
                    printed(r.learning_rate)));
  }
  require_float("learning-rate", r.learning_rate);
  if (static_cast<float>(r.learning_rate) < least_float) {
    throw logic_error(std::format("learning-rate: expected at least {}, got {}",
                                  printed(least_float),
                                  printed(r.learning_rate)));
  }
  require_at_least("negative-sample-rate", r.negative_sample_rate, 0);
  require_at_most("negative-sample-rate", r.negative_sample_rate, most_index);
  require_curve_value("a", r.a);
  require_curve_value("b", r.b);
}

void require_fit_parameters(const umap_request& r) {
  require_shared(r);
  if (r.init != CUML_MANIFOLD_INIT_RANDOM &&
      r.init != CUML_MANIFOLD_INIT_SPECTRAL) {
    throw logic_error("init: expected spectral or random");
  }
  if (r.build_algo != CUML_MANIFOLD_BUILD_BRUTE_FORCE &&
      r.build_algo != CUML_MANIFOLD_BUILD_NN_DESCENT) {
    throw logic_error("build-algo: expected brute-force or nn-descent");
  }
  if (r.build_algo == CUML_MANIFOLD_BUILD_NN_DESCENT &&
      (r.metric == CUML_MANIFOLD_METRIC_MANHATTAN ||
       r.metric == CUML_MANIFOLD_METRIC_CHEBYSHEV)) {
    throw logic_error(std::format(
        "metric: nn-descent supports euclidean, sqeuclidean and cosine, got {}",
        metric_name(r.metric)));
  }
  require_numbers(r);
}

ML::UMAPParams umap_params(const umap_request& r, int64_t n_neighbors,
                           bool fit) {
  ML::UMAPParams p;
  p.n_neighbors = static_cast<int>(n_neighbors);
  p.n_components = static_cast<int>(r.n_components);
  p.n_epochs = r.epochs_given ? static_cast<int>(r.n_epochs) : 0;
  p.learning_rate = static_cast<float>(r.learning_rate);
  p.initial_alpha = static_cast<float>(r.learning_rate);
  p.a = static_cast<float>(r.a);
  p.b = static_cast<float>(r.b);
  p.init = r.init;
  p.negative_sample_rate = static_cast<int>(r.negative_sample_rate);
  p.random_state = r.random_state;
  p.deterministic = r.deterministic;
  p.force_serial_epochs = fit && r.init == CUML_MANIFOLD_INIT_SPECTRAL &&
                          r.n_components <= most_serial_components;
  p.metric = distance_of(r.metric);
  p.p = euclidean_p;
  p.verbosity = rapids_logger::level_enum::warn;
  p.target_metric = ML::UMAPParams::CATEGORICAL;
  p.target_weight = target_weight;
  p.target_n_neighbors = -1;
  p.set_op_mix_ratio = 1.0F;
  p.local_connectivity = 1.0F;
  p.repulsion_strength = 1.0F;
  p.transform_queue_size = transform_queue_size;
  p.callback = nullptr;
  if (fit && r.build_algo == CUML_MANIFOLD_BUILD_NN_DESCENT) {
    p.build_algo = ML::UMAPParams::graph_build_algo::NN_DESCENT;
    auto& nnd = p.build_params.nn_descent_params;
    nnd.graph_degree =
        std::max(nn_descent_degree, static_cast<std::size_t>(n_neighbors));
    nnd.intermediate_graph_degree =
        std::max(nn_descent_intermediate_degree, nnd.graph_degree);
  } else {
    p.build_algo = ML::UMAPParams::graph_build_algo::BRUTE_FORCE_KNN;
  }
  return p;
}

matrix<float, int64_t> float_rows(const rr_view* view, const char* name) {
  const rr_view& v = *raftrkt::require(view, name);
  if (v.dtype != RR_DTYPE_FLOAT32) {
    throw logic_error(std::format("{}: expected float32, got {}", name,
                                  raftrkt::dtype_name(v.dtype)));
  }
  return rows_of<float, int64_t>(view, name, any_extent, any_extent);
}

void require_rows_fit(const char* name, const matrix<float, int64_t>& x) {
  if (x.n > most_index) {
    throw logic_error(
        std::format("{}: {} rows do not fit the index type", name, x.n));
  }
  if (x.d > most_index) {
    throw logic_error(
        std::format("{}: {} columns do not fit the index type", name, x.d));
  }
}

void require_scanned(const char* name, const matrix<float, int64_t>& x,
                     int32_t metric, rmm::cuda_stream_view stream) {
  require_finite(name, x, stream);
  require_bounded_rows(name, x, stream);
  if (metric == CUML_MANIFOLD_METRIC_COSINE) {
    const float least = static_cast<float>(x.d) * least_float;
    const auto [row, squared] =
        cumlrkt::first_normless_row(x.data, x.n, x.d, least, stream);
    if (row >= 0) {
      throw logic_error(std::format(
          "{}: row {}: expected a squared norm of at least {} under cosine, "
          "got {}",
          name, row, printed(least), printed<float>(squared)));
    }
  }
}

void require_converged(const char* name, const float* out, int64_t rows,
                       int64_t cols, rmm::cuda_stream_view stream) {
  const int64_t i = cumlrkt::first_non_finite(out, rows * cols, stream);
  if (i >= 0) {
    const float got = read_one(out, i, stream);
    throw logic_error(std::format(
        "{}: row {}, column {}: the optimisation diverged to {}; lower "
        "learning-rate",
        name, i / cols, i % cols, printed(static_cast<double>(got))));
  }
}

void require_spectral(const umap_request& r, int64_t n) {
  if (r.init != CUML_MANIFOLD_INIT_SPECTRAL) {
    return;
  }
  if (n <= r.n_components) {
    throw logic_error(std::format(
        "n-components: spectral init needs fewer components than the {} rows, "
        "got {}",
        n, r.n_components));
  }
  const int64_t entries = n * (r.n_components + 1);
  if (entries > most_index) {
    throw logic_error(std::format(
        "n-components: spectral init of {} rows with {} components needs {} "
        "entries, more than {}",
        n, r.n_components, entries, most_index));
  }
}

void require_y(const rr_view* y, int64_t n) {
  if (y == nullptr) {
    return;
  }
  on_labels(y, "y", [&](auto tag) {
    static_cast<void>(raftrkt::vector_data<decltype(tag)>(y, "y"));
  });
  raftrkt::detail::require_extent("y", "elements", y->shape[0], n);
}

std::optional<label_codes> codes_of(const rr_view* y, int64_t n,
                                    rmm::cuda_stream_view stream) {
  if (y == nullptr) {
    return std::nullopt;
  }
  std::optional<label_codes> found;
  on_labels(y, "y", [&](auto tag) {
    using L = decltype(tag);
    found.emplace(float_codes(static_cast<const L*>(y->data), n, stream));
  });
  return found;
}

void require_labels(const std::optional<label_codes>& codes) {
  if (codes && codes->labelled > most_labels) {
    throw logic_error(
        std::format("y: expected at most {} distinct labels besides -1, got {}",
                    most_labels, codes->labelled));
  }
}

void fit_into(const raft::handle_t& handle, const rr_view* x_view,
              rr_deposit* out, const rr_view* y, const umap_request& r) {
  require_fit_parameters(r);
  const matrix<float, int64_t> x = float_rows(x_view, "X");
  require_some("X", "rows", x.n, 2);
  require_some("X", "column", x.d, 1);
  require_rows_fit("X", x);
  require_spectral(r, x.n);
  if (r.build_algo == CUML_MANIFOLD_BUILD_NN_DESCENT &&
      x.n < least_nn_descent_rows) {
    throw logic_error(
        std::format("X: nn-descent needs at least {} rows, got {}",
                    least_nn_descent_rows, x.n));
  }
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  require_y(y, x.n);
  require_scanned("X", x, r.metric, stream);
  std::optional<label_codes> codes = codes_of(y, x.n, stream);
  require_labels(codes);
  const int64_t k = std::min(r.n_neighbors, x.n);
  ML::UMAPParams p = umap_params(r, k, true);
  std::unique_ptr<rmm::device_buffer> embedding;
  {
    auto graph = raft::make_host_coo_matrix<float, int, int, uint64_t>(
        handle, static_cast<int>(x.n), static_cast<int>(x.n));
    ML::UMAP::fit(handle, x.data, codes ? codes->codes.data() : nullptr,
                  static_cast<int>(x.n), static_cast<int>(x.d), nullptr,
                  nullptr, &p, embedding, graph);
  }
  auto* data = static_cast<float*>(embedding->data());
  require_converged("embedding", data, x.n, r.n_components, stream);
  settle(handle);
  const std::initializer_list<rr_view> regions{
      raftrkt::region(data, x.n, r.n_components, layout::row_major)};
  raftrkt::deposit(out, "out", handle, std::move(embedding), regions);
}

struct transform_views {
  const rr_view* x;
  const rr_view* train_x;
  const rr_view* embedding;
  rr_deposit* out;
};

void require_entries(const char* name, int64_t m, int64_t each,
                     const char* what) {
  const int64_t entries = m * each;
  if (entries > most_index) {
    throw logic_error(
        std::format("{}: {} rows with {} {} each need {} entries, more than {}",
                    name, m, each, what, entries, most_index));
  }
}

void transform_into(const raft::handle_t& handle, const transform_views& v,
                    const umap_request& r) {
  const matrix<float, int64_t> x = float_rows(v.x, "X");
  require_some("X", "row", x.n);
  require_rows_fit("X", x);
  const matrix<float, int64_t> train = float_rows(v.train_x, "result X");
  require_some("result X", "rows", train.n, 2);
  require_rows_fit("result X", train);
  const matrix<float, int64_t> embedding = float_rows(v.embedding, "embedding");
  raftrkt::detail::require_extent("embedding", "rows", embedding.n, train.n);
  raftrkt::detail::require_extent("embedding", "columns", embedding.d,
                                  r.n_components);
  raftrkt::detail::require_extent("X", "columns", x.d, train.d);
  require_shared(r);
  require_numbers(r);
  if (r.epochs_given && r.n_epochs < least_transform_epochs) {
    throw logic_error(std::format(
        "n-epochs: transform runs n-epochs/3 epochs and needs at least {}, "
        "got {}",
        least_transform_epochs, r.n_epochs));
  }
  const int64_t k = std::min(r.n_neighbors, train.n);
  require_entries("X", x.n, k, "neighbours");
  require_entries("X", x.n, r.n_components, "components");
  require_entries("result X", train.n, r.n_components, "components");
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  require_scanned("result X", train, r.metric, stream);
  require_scanned("X", x, r.metric, stream);
  ML::UMAPParams p = umap_params(r, k, false);
  auto result = std::make_unique<rmm::device_uvector<float>>(
      static_cast<std::size_t>(x.n * r.n_components), stream);
  raftrkt::cuda_check(
      cudaMemsetAsync(result->data(), 0, result->size() * sizeof(float),
                      stream.value()),
      "cudaMemsetAsync");
  ML::UMAP::transform(handle, x.data, static_cast<int>(x.n),
                      static_cast<int>(x.d), train.data,
                      static_cast<int>(train.n), embedding.data,
                      static_cast<int>(train.n), &p, result->data());
  require_converged("output", result->data(), x.n, r.n_components, stream);
  settle(handle);
  const std::initializer_list<rr_view> regions{
      raftrkt::region(result->data(), x.n, r.n_components, layout::row_major)};
  raftrkt::deposit(v.out, "out", handle, std::move(result), regions);
}

}  // namespace

extern "C" {

int cuml_manifold_umap_fit(void* handle, const rr_view* x, const rr_view* y,
                           rr_deposit* out, int64_t n_neighbors,
                           int64_t n_components, int32_t epochs_given,
                           int64_t n_epochs, double learning_rate, double a,
                           double b, int32_t init, int64_t negative_sample_rate,
                           uint64_t random_state, int32_t deterministic,
                           int32_t build_algo, int32_t metric) {
  return guarded(handle, {{"X", x}, {"y", y}}, [&](const raft::handle_t& h) {
    const umap_request r{.n_neighbors = n_neighbors,
                         .n_components = n_components,
                         .epochs_given = epochs_given != 0,
                         .n_epochs = n_epochs,
                         .learning_rate = learning_rate,
                         .a = a,
                         .b = b,
                         .init = init,
                         .negative_sample_rate = negative_sample_rate,
                         .random_state = random_state,
                         .deterministic = deterministic != 0,
                         .build_algo = build_algo,
                         .metric = metric};
    fit_into(h, x, raftrkt::require(out, "out"), y, r);
  });
}

int cuml_manifold_umap_transform(
    void* handle, const rr_view* x, const rr_view* train_x,
    const rr_view* embedding, rr_deposit* out, int64_t n_neighbors,
    int64_t n_components, int32_t epochs_given, int64_t n_epochs,
    double learning_rate, double a, double b, int64_t negative_sample_rate,
    uint64_t random_state, int32_t deterministic, int32_t metric) {
  return guarded(
      handle, {{"X", x}, {"result X", train_x}, {"embedding", embedding}},
      [&](const raft::handle_t& h) {
        const umap_request r{.n_neighbors = n_neighbors,
                             .n_components = n_components,
                             .epochs_given = epochs_given != 0,
                             .n_epochs = n_epochs,
                             .learning_rate = learning_rate,
                             .a = a,
                             .b = b,
                             .init = CUML_MANIFOLD_INIT_RANDOM,
                             .negative_sample_rate = negative_sample_rate,
                             .random_state = random_state,
                             .deterministic = deterministic != 0,
                             .build_algo = CUML_MANIFOLD_BUILD_BRUTE_FORCE,
                             .metric = metric};
        transform_into(h, {x, train_x, embedding, raftrkt::require(out, "out")},
                       r);
      });
}

CUMLRKT_JOB(cuml_manifold_umap_fit)
CUMLRKT_JOB(cuml_manifold_umap_transform)
}
