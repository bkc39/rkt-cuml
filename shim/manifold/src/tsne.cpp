#include <cuda_runtime_api.h>
#include <cuml/manifold/tsne.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <utility>

#include "cumlrkt/manifold.h"
#include "cumlrkt/rows.hpp"
#include "cumlrkt/support.hpp"
#include "cumlrkt/transpose.hpp"
#include "manifold_support.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"

namespace {

using cumlrkt::buffer;
using cumlrkt::guarded;
using cumlrkt::logic_error;
using cumlrkt::matrix;
using cumlrkt::most_float;
using cumlrkt::most_index;
using cumlrkt::printed;
using cumlrkt::require_at_least;
using cumlrkt::require_at_most;
using cumlrkt::require_float;
using cumlrkt::require_some;
using cumlrkt::scratch;
using cumlrkt::settle;
using cumlrkt::to_columns;
using cumlrkt::to_rows;
using cumlrkt::manifold::distance_of;
using cumlrkt::manifold::float_rows;
using cumlrkt::manifold::require_converged;
using cumlrkt::manifold::require_rows_fit;
using cumlrkt::manifold::require_scanned;
using raftrkt::layout;

constexpr int64_t embedding_dim = 2;
constexpr int64_t most_neighbors = 1023;
constexpr int64_t most_exact_rows = int64_t{65535} * 32;
constexpr int64_t most_exaggeration_iter = 250;
constexpr float least_float = std::numeric_limits<float>::min();
constexpr float exact_min_grad_norm = 1e-7F;
constexpr int perplexity_max_iter = 100;
constexpr float perplexity_tol = 1e-5F;
constexpr float theta = 0.5F;
constexpr float epssq = 0.0025F;
constexpr float min_gain = 0.01F;
constexpr float pre_momentum = 0.5F;
constexpr float post_momentum = 0.8F;
constexpr float euclidean_p = 2.0F;
constexpr double most_sqeuclidean_norm = 0x1p61;
constexpr double most_manhattan_norm = std::numbers::sqrt2 * 0x1p62;
constexpr double most_pca_spread = 0x1p126;
constexpr const char* divergence_advice = "lower learning-rate, or rescale X";

struct tsne_request {
  int64_t n_components;
  double perplexity;
  double early_exaggeration;
  double learning_rate;
  int64_t max_iter;
  int64_t n_neighbors;
  int32_t method;
  int32_t metric;
  int32_t init;
  int64_t random_state;
};

void require_codes(const tsne_request& r) {
  if (r.method != CUML_MANIFOLD_TSNE_FFT &&
      r.method != CUML_MANIFOLD_TSNE_EXACT) {
    throw logic_error("method: expected fft or exact");
  }
  static_cast<void>(distance_of(r.metric));
  if (r.init != CUML_MANIFOLD_TSNE_INIT_RANDOM &&
      r.init != CUML_MANIFOLD_TSNE_INIT_PCA) {
    throw logic_error("init: expected random or pca");
  }
}

void require_learning_rate(double rate) {
  if (!std::isfinite(rate) || rate <= 0) {
    throw logic_error(
        std::format("learning-rate: expected a finite value above 0, got {}",
                    printed(rate)));
  }
  if (rate > most_float / 2) {
    throw logic_error(std::format("learning-rate: expected at most {}, got {}",
                                  printed(most_float / 2), printed(rate)));
  }
  if (static_cast<float>(rate) < least_float) {
    throw logic_error(std::format("learning-rate: expected at least {}, got {}",
                                  printed(least_float), printed(rate)));
  }
}

void require_numbers(const tsne_request& r) {
  if (r.n_components != embedding_dim) {
    throw logic_error(std::format("n-components: expected {}, got {}",
                                  embedding_dim, r.n_components));
  }
  require_at_least("n-neighbors", r.n_neighbors, 2);
  require_at_most("n-neighbors", r.n_neighbors, most_neighbors);
  if (!std::isfinite(r.perplexity) || r.perplexity <= 1 ||
      (r.perplexity < 2 && static_cast<float>(r.perplexity) <= 1)) {
    throw logic_error(std::format(
        "perplexity: expected a finite value above 1 in float32, got {}",
        printed(r.perplexity)));
  }
  require_float("perplexity", r.perplexity);
  if (!std::isfinite(r.early_exaggeration) || r.early_exaggeration < 1) {
    throw logic_error(std::format(
        "early-exaggeration: expected a finite value of at least 1, got {}",
        printed(r.early_exaggeration)));
  }
  require_float("early-exaggeration", r.early_exaggeration);
  require_learning_rate(r.learning_rate);
  require_at_least("max-iter", r.max_iter, 1);
  require_at_most("max-iter", r.max_iter, most_index);
  require_at_least("random-state", r.random_state, 0);
}

void require_shape(const tsne_request& r, const matrix<float, int64_t>& x) {
  require_some("X", "row", x.n, 2);
  require_some("X", "column", x.d, 2);
  require_rows_fit("X", x);
  const int64_t entries = x.n * embedding_dim;
  if (r.init == CUML_MANIFOLD_TSNE_INIT_RANDOM && entries > most_index) {
    throw logic_error(
        std::format("X: random init of {} rows needs {} entries, more than {}",
                    x.n, entries, most_index));
  }
  const int64_t k = std::min(r.n_neighbors, x.n);
  if (r.perplexity > static_cast<double>(k)) {
    throw logic_error(
        std::format("perplexity: expected at most the {} neighbours, got {}", k,
                    printed(r.perplexity)));
  }
  if (r.method == CUML_MANIFOLD_TSNE_EXACT && x.n > most_exact_rows) {
    throw logic_error(
        std::format("method: exact supports at most {} rows, got {}",
                    most_exact_rows, x.n));
  }
}

struct norm_bound {
  const char* metric;
  const char* norm;
  double most;
  bool l1;
};

std::optional<norm_bound> norm_bound_of(int32_t metric) {
  switch (metric) {
    case CUML_MANIFOLD_METRIC_SQEUCLIDEAN:
      return norm_bound{.metric = "sqeuclidean",
                        .norm = "a squared norm",
                        .most = most_sqeuclidean_norm,
                        .l1 = false};
    case CUML_MANIFOLD_METRIC_MANHATTAN:
      return norm_bound{.metric = "manhattan",
                        .norm = "an L1 norm",
                        .most = most_manhattan_norm,
                        .l1 = true};
    default:
      return std::nullopt;
  }
}

void require_distance_bound(const matrix<float, int64_t>& x, int32_t metric,
                            rmm::cuda_stream_view stream) {
  const std::optional<norm_bound> b = norm_bound_of(metric);
  if (!b) {
    return;
  }
  const auto [row, norm] =
      b->l1 ? cumlrkt::first_large_l1_row(x.data, x.n, x.d, b->most, stream)
            : cumlrkt::first_large_row(x.data, x.n, x.d, b->most, stream);
  if (row >= 0) {
    throw logic_error(
        std::format("X: row {}: expected {} of at most {} under {}, got {}",
                    row, b->norm, printed(b->most), b->metric, printed(norm)));
  }
}

void require_pca_spread(const matrix<float, int64_t>& x, int32_t init,
                        rmm::cuda_stream_view stream) {
  const double spread = init == CUML_MANIFOLD_TSNE_INIT_PCA
                            ? cumlrkt::total_spread(x.data, x.n, x.d, stream)
                            : 0;
  if (!(spread <= most_pca_spread)) {
    throw logic_error(
        std::format("X: expected a total squared distance from the mean of at "
                    "most {} under pca init, got {}",
                    printed(most_pca_spread), printed(spread)));
  }
}

void require_distinct(const matrix<float, int64_t>& x,
                      rmm::cuda_stream_view stream) {
  if (cumlrkt::first_row_unlike_first(x.data, x.n, x.d, stream) < 0) {
    throw logic_error(
        "X: every row equals row 0; t-SNE needs two distinct rows");
  }
}

void require_spread_embedding(const float* y, int64_t n,
                              rmm::cuda_stream_view stream) {
  if (cumlrkt::first_row_unlike_first(y, n, embedding_dim, stream) < 0) {
    throw logic_error(
        "embedding: every row equals row 0; the embedding is constant, use "
        "init 'random");
  }
}

ML::TSNEParams tsne_params(const tsne_request& r, int64_t n) {
  ML::TSNEParams p;
  p.dim = static_cast<int>(embedding_dim);
  p.n_neighbors = static_cast<int>(std::min(r.n_neighbors, n));
  p.theta = theta;
  p.epssq = epssq;
  p.perplexity = static_cast<float>(r.perplexity);
  p.perplexity_max_iter = perplexity_max_iter;
  p.perplexity_tol = perplexity_tol;
  p.early_exaggeration = static_cast<float>(r.early_exaggeration);
  p.late_exaggeration = 1.0F;
  p.exaggeration_iter =
      static_cast<int>(std::min(most_exaggeration_iter, r.max_iter));
  p.min_gain = min_gain;
  p.pre_learning_rate = static_cast<float>(r.learning_rate);
  p.post_learning_rate = static_cast<float>(r.learning_rate * 2);
  p.max_iter = static_cast<int>(r.max_iter);
  p.min_grad_norm =
      r.method == CUML_MANIFOLD_TSNE_FFT ? 0.0F : exact_min_grad_norm;
  p.pre_momentum = pre_momentum;
  p.post_momentum = post_momentum;
  p.random_state = r.random_state;
  p.verbosity = rapids_logger::level_enum::warn;
  p.init = r.init == CUML_MANIFOLD_TSNE_INIT_PCA ? ML::TSNE_INIT::PCA
                                                 : ML::TSNE_INIT::RANDOM;
  p.square_distances = true;
  p.metric = distance_of(r.metric);
  p.p = euclidean_p;
  p.algorithm = r.method == CUML_MANIFOLD_TSNE_EXACT ? ML::TSNE_ALGORITHM::EXACT
                                                     : ML::TSNE_ALGORITHM::FFT;
  return p;
}

struct fit_outputs {
  rr_deposit* out;
  double* kl_divergence;
  int64_t* n_iter;
};

void fit_into(const raft::handle_t& handle, const rr_view* x_view,
              const fit_outputs& o, const tsne_request& r) {
  require_codes(r);
  require_numbers(r);
  const matrix<float, int64_t> x = float_rows(x_view, "X");
  require_shape(r, x);
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  require_scanned("X", x, r.metric, stream);
  require_distance_bound(x, r.metric, stream);
  require_pca_spread(x, r.init, stream);
  require_distinct(x, stream);
  ML::TSNEParams p = tsne_params(r, x.n);
  auto embedding = std::make_unique<buffer<float>>(
      scratch<float>(x.n * embedding_dim, stream));
  float kl = 0;
  int iterations = 0;
  {
    buffer<float> columns = scratch<float>(x.n * x.d, stream);
    to_columns<float>(handle, x.data, x.n, x.d, nullptr, columns.data(), 0);
    buffer<float> y = scratch<float>(x.n * embedding_dim, stream);
    raftrkt::cuda_check(
        cudaMemsetAsync(y.data(), 0, y.size() * sizeof(float), stream.value()),
        "cudaMemsetAsync");
    ML::TSNE_fit(handle, columns.data(), y.data(), static_cast<int>(x.n),
                 static_cast<int>(x.d), nullptr, nullptr, p, &kl, &iterations);
    to_rows<float>(handle, y.data(), x.n, embedding_dim, nullptr,
                   embedding->data());
  }
  require_converged("embedding", embedding->data(), x.n, embedding_dim,
                    divergence_advice, stream);
  require_spread_embedding(embedding->data(), x.n, stream);
  settle(handle);
  const std::initializer_list<rr_view> regions{raftrkt::region(
      embedding->data(), x.n, embedding_dim, layout::row_major)};
  raftrkt::deposit(o.out, "out", handle, std::move(embedding), regions);
  *o.kl_divergence = kl;
  *o.n_iter = iterations;
}

}  // namespace

extern "C" {

int cuml_manifold_tsne_fit(void* handle, const rr_view* x, rr_deposit* out,
                           int64_t n_components, double perplexity,
                           double early_exaggeration, double learning_rate,
                           int64_t max_iter, int64_t n_neighbors,
                           int32_t method, int32_t metric, int32_t init,
                           int64_t random_state, double* kl_divergence,
                           int64_t* n_iter) {
  return guarded(handle, {{"X", x}}, [&](const raft::handle_t& h) {
    const tsne_request r{.n_components = n_components,
                         .perplexity = perplexity,
                         .early_exaggeration = early_exaggeration,
                         .learning_rate = learning_rate,
                         .max_iter = max_iter,
                         .n_neighbors = n_neighbors,
                         .method = method,
                         .metric = metric,
                         .init = init,
                         .random_state = random_state};
    fit_into(h, x,
             {.out = raftrkt::require(out, "out"),
              .kl_divergence = raftrkt::require(kl_divergence, "kl_divergence"),
              .n_iter = raftrkt::require(n_iter, "n_iter")},
             r);
  });
}

CUMLRKT_JOB(cuml_manifold_tsne_fit)
}
