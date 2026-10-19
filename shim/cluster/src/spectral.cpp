#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cuml/cluster/spectral_clustering.hpp>
#include <format>
#include <initializer_list>
#include <limits>
#include <memory>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <string>
#include <utility>

#include "cumlrkt/cluster.h"
#include "cumlrkt/common.hpp"
#include "linkage_device.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"
#include "spectral_device.hpp"
#include "support.hpp"

namespace {

using cumlrkt::logic_error;
using cumlrkt::cluster::first_graph_fault;
using cumlrkt::cluster::first_large_row;
using cumlrkt::cluster::graph_fault;
using cumlrkt::cluster::graph_fault_kind;
using cumlrkt::cluster::in_device_scope;
using cumlrkt::cluster::matrix;
using cumlrkt::cluster::on_float32_device;
using cumlrkt::cluster::printed;
using cumlrkt::cluster::require_at_least;
using cumlrkt::cluster::require_non_negative;
using cumlrkt::cluster::require_some;
using cumlrkt::cluster::rows_of;
using cumlrkt::cluster::settle;
using raftrkt::any_extent;

constexpr int64_t most_index = std::numeric_limits<int32_t>::max();
constexpr int64_t least_samples = 4;
constexpr int64_t least_neighbours = 2;
constexpr int64_t components_per_sample = 3;
constexpr double most_squared_norm = 0x1p125;
constexpr double most_tol = FLT_MAX;

struct solver_options {
  int64_t n_clusters;
  int64_t n_components;
  int32_t auto_components;
  int32_t n_init;
  double eigen_tol;
  uint64_t seed;
};

void require_clusters(int64_t n_clusters, int64_t n, const char* subject) {
  require_at_least("n-clusters", n_clusters, 1);
  if (n_clusters > n) {
    throw logic_error(std::format("{}: {} samples are fewer than the {} clusters",
                                  subject, n, n_clusters));
  }
}

int64_t components(const solver_options& o, int64_t n, const char* unit) {
  const int64_t most = (n - 1) / components_per_sample;
  if (o.auto_components != 0) {
    return std::min(o.n_clusters, most);
  }
  require_at_least("n-components", o.n_components, 1);
  if (o.n_components > most) {
    throw logic_error(
        std::format("n-components: expected at most {} for {} {}, got {}", most,
                    n, unit, o.n_components));
  }
  return o.n_components;
}

void require_solver(const solver_options& o) {
  require_at_least("n-init", o.n_init, 1);
  require_non_negative("eigen-tol", o.eigen_tol);
  if (o.eigen_tol > most_tol) {
    throw logic_error(std::format("eigen-tol: expected at most {}, got {}",
                                  most_tol, printed(o.eigen_tol)));
  }
}

ML::SpectralClustering::params params_of(const solver_options& o,
                                         int64_t n_components,
                                         int64_t n_neighbors) {
  return {.n_clusters = static_cast<int>(o.n_clusters),
          .n_components = static_cast<int>(n_components),
          .n_init = o.n_init,
          .n_neighbors = static_cast<int>(n_neighbors),
          .eigen_tol = static_cast<float>(o.eigen_tol),
          .seed = o.seed};
}

struct spectral_outputs {
  rmm::device_uvector<int32_t> labels;
};

template <typename Fit>
void fit_into(const raft::handle_t& handle, rr_deposit* out, int64_t n,
              Fit&& fit) {
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  auto outputs = std::make_unique<spectral_outputs>(spectral_outputs{
      rmm::device_uvector<int32_t>(static_cast<std::size_t>(n), stream)});
  fit(raft::make_device_vector_view<int, int>(outputs->labels.data(),
                                              static_cast<int>(n)));
  const std::initializer_list<rr_view> regions{
      raftrkt::region(outputs->labels)};
  settle(handle);
  raftrkt::deposit(out, "out", handle, std::move(outputs), regions);
}

matrix<float, int64_t> checked_x(const rr_view* view) {
  const matrix<float, int64_t> x =
      rows_of<float, int64_t>(view, "X", any_extent, any_extent);
  if (x.n < least_samples) {
    throw logic_error(std::format("X: expected at least {} rows, got {}",
                                  least_samples, x.n));
  }
  require_some("X", "column", x.d);
  if (x.n > most_index / x.d) {
    throw logic_error(std::format(
        "X: {}x{} elements do not fit the index type", x.n, x.d));
  }
  return x;
}

void require_knn_graph(int64_t n, int64_t k) {
  const int64_t entries = n * ((2 * k) + 1);
  if (entries > most_index) {
    throw logic_error(std::format(
        "X: {} rows with {} neighbours need {} graph entries, more than {}", n,
        k, entries, most_index));
  }
}

void require_bounded_rows(const matrix<float, int64_t>& x,
                          rmm::cuda_stream_view stream) {
  const auto [row, squared] =
      first_large_row(x.data, x.n, x.d, most_squared_norm, stream);
  if (row >= 0) {
    throw logic_error(
        std::format("X: row {}: expected a squared norm of at most {}, got {}",
                    row, printed(most_squared_norm), printed(squared)));
  }
}

void fit_dense(const raft::handle_t& handle, const rr_view* view,
               rr_deposit* out, const solver_options& o, int64_t n_neighbors) {
  const matrix<float, int64_t> x = checked_x(view);
  require_clusters(o.n_clusters, x.n, "X");
  const int64_t nc = components(o, x.n, "rows");
  require_at_least("n-neighbors", n_neighbors, least_neighbours);
  require_solver(o);
  const int64_t k = std::min(n_neighbors, x.n - 1);
  require_knn_graph(x.n, k);
  require_bounded_rows(x, raft::resource::get_cuda_stream(handle));
  const auto dataset = raft::make_device_matrix_view<float, int, raft::row_major>(
      x.data, static_cast<int>(x.n), static_cast<int>(x.d));
  fit_into(handle, out, x.n, [&](raft::device_vector_view<int, int> labels) {
    ML::SpectralClustering::fit_predict(handle, params_of(o, nc, k), dataset,
                                        labels);
  });
}

struct graph_views {
  const rr_view* rows;
  const rr_view* cols;
  const rr_view* vals;
};

struct coo {
  int32_t* rows;
  int32_t* cols;
  float* vals;
  int64_t nnz;
};

void require_entries(const char* name, int64_t got, int64_t nnz) {
  if (got != nnz) {
    throw logic_error(std::format("{}: expected {} entries, as rows has, got {}",
                                  name, nnz, got));
  }
}

coo checked_graph(const graph_views& g) {
  coo c{raftrkt::vector_data<int32_t>(g.rows, "rows"),
        raftrkt::vector_data<int32_t>(g.cols, "cols"),
        raftrkt::vector_data<float>(g.vals, "vals"), g.rows->shape[0]};
  require_entries("cols", g.cols->shape[0], c.nnz);
  require_entries("vals", g.vals->shape[0], c.nnz);
  return c;
}

void require_samples(int64_t n, int64_t nnz) {
  require_at_least("n-samples", n, least_samples);
  if (n > most_index) {
    throw logic_error(std::format("n-samples: expected at most {}, got {}",
                                  most_index, n));
  }
  if (nnz > most_index - n) {
    throw logic_error(std::format(
        "rows: {} entries and {} samples do not fit the index type", nnz, n));
  }
}

void require_sound_graph(const coo& c, int64_t n,
                         rmm::cuda_stream_view stream) {
  const graph_fault f =
      first_graph_fault(c.rows, c.cols, c.vals, c.nnz, n, stream);
  switch (f.kind) {
    case graph_fault_kind::row_index:
    case graph_fault_kind::col_index:
      throw logic_error(std::format(
          "{}: entry {}: expected an index in [0, {}), got {}",
          f.kind == graph_fault_kind::row_index ? "rows" : "cols", f.entry, n,
          static_cast<int64_t>(f.value)));
    case graph_fault_kind::self_loop:
      throw logic_error(std::format(
          "rows: entry {}: self-loop at sample {}; leave the diagonal out",
          f.entry, static_cast<int64_t>(f.value)));
    case graph_fault_kind::weight:
      throw logic_error(std::format(
          "vals: entry {}: expected a finite weight of at least 0, got {}",
          f.entry, printed(f.value)));
    case graph_fault_kind::none:
      break;
  }
  if (!f.any_positive) {
    throw logic_error("vals: expected at least one positive weight, got none");
  }
}

void fit_graph(const raft::handle_t& handle, const graph_views& g, int64_t n,
               rr_deposit* out, const solver_options& o) {
  const coo c = checked_graph(g);
  require_samples(n, c.nnz);
  require_clusters(o.n_clusters, n, "n-samples");
  const int64_t nc = components(o, n, "samples");
  require_solver(o);
  require_sound_graph(c, n, raft::resource::get_cuda_stream(handle));
  const auto nnz = static_cast<int>(c.nnz);
  fit_into(handle, out, n, [&](raft::device_vector_view<int, int> labels) {
    ML::SpectralClustering::fit_predict(
        handle, params_of(o, nc, 0),
        raft::make_device_vector_view<int, int>(c.rows, nnz),
        raft::make_device_vector_view<int, int>(c.cols, nnz),
        raft::make_device_vector_view<float, int>(c.vals, nnz), labels);
  });
}

}  // namespace

extern "C" {

int cuml_cluster_spectral_clustering_fit(void* handle, const rr_view* x,
                                         rr_deposit* out, int64_t n_clusters,
                                         int64_t n_components,
                                         int32_t auto_components,
                                         int64_t n_neighbors, int32_t n_init,
                                         double eigen_tol, uint64_t seed) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [&] {
    rr_deposit* const into = raftrkt::require(out, "out");
    const solver_options o{n_clusters, n_components, auto_components,
                           n_init,     eigen_tol,    seed};
    on_float32_device(handle, {{"X", x}}, x, "X", [&](const raft::handle_t& h) {
      fit_dense(h, x, into, o, n_neighbors);
    });
  });
}

void cuml_cluster_spectral_clustering_fit_job(void* args, size_t args_len,
                                              rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_spectral_clustering_fit>(
      args, args_len, outcome, cumlrkt::last_error_slot());
}

int cuml_cluster_spectral_clustering_fit_graph(
    void* handle, const rr_view* rows, const rr_view* cols,
    const rr_view* vals, int64_t n_samples, rr_deposit* out,
    int64_t n_clusters, int64_t n_components, int32_t auto_components,
    int32_t n_init, double eigen_tol, uint64_t seed) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [&] {
    rr_deposit* const into = raftrkt::require(out, "out");
    const solver_options o{n_clusters, n_components, auto_components,
                           n_init,     eigen_tol,    seed};
    const graph_views g{rows, cols, vals};
    in_device_scope(handle, {{"rows", rows}, {"cols", cols}, {"vals", vals}},
                    [&](const raft::handle_t& h) {
                      fit_graph(h, g, n_samples, into, o);
                    });
  });
}

void cuml_cluster_spectral_clustering_fit_graph_job(void* args,
                                                    size_t args_len,
                                                    rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_spectral_clustering_fit_graph>(
      args, args_len, outcome, cumlrkt::last_error_slot());
}
}
