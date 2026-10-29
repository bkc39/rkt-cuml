#include <cstdint>
#include <cuml/manifold/spectral_embedding.hpp>
#include <format>
#include <initializer_list>
#include <memory>
#include <optional>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <string>
#include <utility>

#include "cumlrkt/graph.hpp"
#include "cumlrkt/largest_positive.hpp"
#include "cumlrkt/manifold.h"
#include "cumlrkt/rows.hpp"
#include "cumlrkt/scale.hpp"
#include "cumlrkt/support.hpp"
#include "cumlrkt/transpose.hpp"
#include "manifold_support.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"
#include "spectral_checks.hpp"
#include "spectral_residual.hpp"

namespace {

using cumlrkt::buffer;
using cumlrkt::checked_graph;
using cumlrkt::coo;
using cumlrkt::graph_views;
using cumlrkt::guarded;
using cumlrkt::largest_magnitude;
using cumlrkt::logic_error;
using cumlrkt::make_largest_positive;
using cumlrkt::matrix;
using cumlrkt::most_index;
using cumlrkt::require_at_least;
using cumlrkt::require_at_most;
using cumlrkt::require_basis;
using cumlrkt::require_lanczos_rows;
using cumlrkt::require_samples;
using cumlrkt::require_some;
using cumlrkt::require_sound_graph;
using cumlrkt::require_unlike_first;
using cumlrkt::scale_exponent;
using cumlrkt::scaled_copy;
using cumlrkt::scratch;
using cumlrkt::settle;
using cumlrkt::to_rows;
using cumlrkt::unit_exponent;
using cumlrkt::manifold::column_fits;
using cumlrkt::manifold::first_refused;
using cumlrkt::manifold::float_rows;
using cumlrkt::manifold::graph_shape;
using cumlrkt::manifold::refusal;
using cumlrkt::manifold::require_nonzero_columns;
using cumlrkt::manifold::require_rows_fit;
using cumlrkt::manifold::require_scanned;
using cumlrkt::manifold::require_solved;
using cumlrkt::manifold::shape_of;
using raftrkt::layout;

constexpr int64_t least_neighbors = 2;
constexpr int64_t least_columns = 2;
constexpr const char* zero_advice =
    "; deduplicate or centre X, or change n-neighbors";
constexpr const char* solver_advice =
    "; graphs whose Laplacian repeats eigenvalues (rings, grids, complete "
    "graphs, stars, identical pieces) are beyond cuML's eigensolver";
constexpr const char* spread_advice =
    "; these weights span orders of magnitude and need norm-laplacian";
constexpr const char* asymmetric_advice = "; the graph is not symmetric";
constexpr const char* edgeless_advice =
    "; samples without edges can displace the graph's eigenvectors";

constexpr double heavy_span = 0x1p8;

struct embedding_request {
  int64_t n_components;
  int32_t norm_laplacian;
  int32_t drop_first;
  uint64_t random_state;
};

embedding_request request_of(int64_t n_components, int32_t norm_laplacian,
                             int32_t drop_first, uint64_t random_state) {
  return {.n_components = n_components,
          .norm_laplacian = norm_laplacian,
          .drop_first = drop_first,
          .random_state = random_state};
}

int64_t eigenvectors(const embedding_request& r) {
  return r.n_components + (r.drop_first != 0 ? 1 : 0);
}

void require_components(const embedding_request& r) {
  require_at_least("n-components", r.n_components, 1);
  require_at_most("n-components", r.n_components, most_index);
}

void require_lanczos(const embedding_request& r, int64_t n, const char* unit) {
  const bool one = r.n_components == 1;
  require_lanczos_rows(
      n, eigenvectors(r), "n-components", unit,
      std::format("{} component{}{} need{}", r.n_components, one ? "" : "s",
                  r.drop_first != 0 ? " with drop-first" : "", one ? "s" : ""));
}

ML::SpectralEmbedding::params params_of(const embedding_request& r,
                                        int64_t n_neighbors) {
  return {.n_components = static_cast<int>(eigenvectors(r)),
          .n_neighbors = static_cast<int>(n_neighbors),
          .norm_laplacian = r.norm_laplacian != 0,
          .drop_first = r.drop_first != 0,
          .seed = r.random_state};
}

using embedding_view = raft::device_matrix_view<float, int, raft::col_major>;

template <typename Unsolved, typename Advice, typename Solve, typename Check>
void embed(const raft::handle_t& handle, rr_deposit* out, int64_t n, int64_t k,
           Unsolved&& unsolved, Advice&& advice, Solve&& solve, Check&& check) {
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  auto embedding =
      std::make_unique<buffer<float>>(scratch<float>(n * k, stream));
  {
    buffer<float> columns = scratch<float>(n * k, stream);
    solve(raft::make_device_matrix_view<float, int, raft::col_major>(
        columns.data(), static_cast<int>(n), static_cast<int>(k)));
    to_rows<float>(handle, columns.data(), n, k, nullptr, embedding->data());
    require_solved(embedding->data(), n, k, unsolved, stream);
    require_nonzero_columns(columns.data(), n, k, advice, stream);
    check(columns.data());
  }
  make_largest_positive<float>(handle, embedding->data(), n, k);
  settle(handle);
  const std::initializer_list<rr_view> regions{
      raftrkt::region(embedding->data(), n, k, layout::row_major)};
  raftrkt::deposit(out, "out", handle, std::move(embedding), regions);
}

std::string graph_advice(const raft::handle_t& handle, const coo& c, int64_t n,
                         const embedding_request& r) {
  const graph_shape s = shape_of(handle, c, n);
  if (!s.symmetric) {
    return asymmetric_advice;
  }
  const bool heavy = r.norm_laplacian == 0 && s.span >= heavy_span;
  const bool edgeless = r.norm_laplacian != 0 && s.edgeless;
  return std::string(solver_advice) + (edgeless ? edgeless_advice : "") +
         (heavy ? spread_advice : "");
}

template <typename Advice>
void require_accurate(const raft::handle_t& handle, const coo& c, int64_t n,
                      const float* columns, const embedding_request& r,
                      Advice&& advice) {
  const std::optional<refusal> refused = first_refused(
      column_fits(handle, c, n, columns, r.n_components, r.norm_laplacian != 0),
      r.drop_first != 0);
  if (refused) {
    throw logic_error(std::format(
        "embedding: column {}{}: the eigensolver's eigenvectors "
        "are inaccurate on this graph{}",
        refused->column, refused->constant ? " is constant" : "", advice()));
  }
}

template <typename Solve>
void on_scaled(const raft::handle_t& handle, float* data, int64_t count,
               int exponent, Solve&& solve) {
  if (exponent == 0) {
    solve(data);
  } else {
    buffer<float> scaled =
        scratch<float>(count, raft::resource::get_cuda_stream(handle));
    scaled_copy<float>(handle, data, count, scaled.data(), exponent);
    solve(scaled.data());
  }
}

void fit_dense(const raft::handle_t& handle, const rr_view* view,
               rr_deposit* out, const embedding_request& r,
               int64_t n_neighbors) {
  require_components(r);
  require_at_least("n-neighbors", n_neighbors, least_neighbors);
  const matrix<float, int64_t> x = float_rows(view, "X");
  require_some("X", "column", x.d, least_columns);
  require_rows_fit("X", x);
  require_lanczos(r, x.n, "rows");
  require_basis(x.n, eigenvectors(r), "n-components", "rows", "eigenvectors");
  if (n_neighbors > x.n) {
    throw logic_error(std::format(
        "n-neighbors: expected at most the {} rows, got {}", x.n, n_neighbors));
  }
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  require_scanned("X", x, CUML_MANIFOLD_METRIC_EUCLIDEAN, stream);
  require_unlike_first("X", x.data, x.n, x.d,
                       "a spectral embedding needs two distinct rows", stream);
  const int exponent =
      scale_exponent<float>(largest_magnitude(x.data, x.n * x.d, stream));
  embed(
      handle, out, x.n, r.n_components, [] { return ""; },
      [] { return zero_advice; },
      [&](embedding_view e) {
        on_scaled(handle, x.data, x.n * x.d, exponent, [&](float* data) {
          ML::SpectralEmbedding::transform(
              handle, params_of(r, n_neighbors),
              raft::make_device_matrix_view<float, int, raft::row_major>(
                  data, static_cast<int>(x.n), static_cast<int>(x.d)),
              e);
        });
      },
      [](const float*) {});
}

void fit_graph(const raft::handle_t& handle, const graph_views& g, int64_t n,
               rr_deposit* out, const embedding_request& r) {
  require_components(r);
  const coo c = checked_graph(g);
  require_lanczos(r, n, "samples");
  require_samples(n, c.nnz, 1);
  require_basis(n, eigenvectors(r), "n-components", "samples", "eigenvectors");
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  require_sound_graph(c, n, stream);
  const int exponent =
      r.norm_laplacian != 0
          ? 0
          : unit_exponent(largest_magnitude(c.vals, c.nnz, stream));
  const auto advice = [&] { return graph_advice(handle, c, n, r); };
  embed(
      handle, out, n, r.n_components, advice, advice,
      [&](embedding_view e) {
        on_scaled(handle, c.vals, c.nnz, exponent, [&](float* vals) {
          ML::SpectralEmbedding::transform(
              handle, params_of(r, 0),
              raft::make_device_vector_view<int, int64_t>(c.rows, c.nnz),
              raft::make_device_vector_view<int, int64_t>(c.cols, c.nnz),
              raft::make_device_vector_view<float, int64_t>(vals, c.nnz), e);
        });
      },
      [&](const float* columns) {
        require_accurate(handle, c, n, columns, r, advice);
      });
}

}  // namespace

extern "C" {

int cuml_manifold_spectral_embedding_fit(void* handle, const rr_view* x,
                                         int64_t n_neighbors, rr_deposit* out,
                                         int64_t n_components,
                                         int32_t norm_laplacian,
                                         int32_t drop_first,
                                         uint64_t random_state) {
  return guarded(handle, {{"X", x}}, [&](const raft::handle_t& h) {
    fit_dense(
        h, x, raftrkt::require(out, "out"),
        request_of(n_components, norm_laplacian, drop_first, random_state),
        n_neighbors);
  });
}

CUMLRKT_JOB(cuml_manifold_spectral_embedding_fit)

int cuml_manifold_spectral_embedding_fit_graph(
    void* handle, const rr_view* rows, const rr_view* cols, const rr_view* vals,
    int64_t n_samples, rr_deposit* out, int64_t n_components,
    int32_t norm_laplacian, int32_t drop_first, uint64_t random_state) {
  return guarded(
      handle, {{"rows", rows}, {"cols", cols}, {"vals", vals}},
      [&](const raft::handle_t& h) {
        fit_graph(
            h, {rows, cols, vals}, n_samples, raftrkt::require(out, "out"),
            request_of(n_components, norm_laplacian, drop_first, random_state));
      });
}

CUMLRKT_JOB(cuml_manifold_spectral_embedding_fit_graph)
}
