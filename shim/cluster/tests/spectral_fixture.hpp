#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cluster_fixture.hpp"
#include "cumlrkt/cluster.h"
#include "kmeans_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {

constexpr int64_t default_neighbours = 10;
constexpr int32_t default_n_init = 10;

struct spectral_call {
  const rr_view* x = nullptr;
  rr_deposit* out = nullptr;
  int64_t n_clusters = 2;
  int64_t n_components = 0;
  int32_t auto_components = 1;
  int64_t n_neighbors = default_neighbours;
  int32_t n_init = default_n_init;
  double eigen_tol = 0;
  uint64_t seed = 0;
};

inline std::string spectral(const resources& res, const spectral_call& c) {
  return outcome(cuml_cluster_spectral_clustering_fit(
      res.handle, c.x, c.out, c.n_clusters, c.n_components, c.auto_components,
      c.n_neighbors, c.n_init, c.eigen_tol, c.seed));
}

struct graph_call {
  const rr_view* rows = nullptr;
  const rr_view* cols = nullptr;
  const rr_view* vals = nullptr;
  int64_t n_samples = 0;
  rr_deposit* out = nullptr;
  int64_t n_clusters = 2;
  int64_t n_components = 0;
  int32_t auto_components = 1;
  int32_t n_init = default_n_init;
  double eigen_tol = 0;
  uint64_t seed = 0;
};

inline std::string spectral_graph(const resources& res, const graph_call& c) {
  return outcome(cuml_cluster_spectral_clustering_fit_graph(
      res.handle, c.rows, c.cols, c.vals, c.n_samples, c.out, c.n_clusters,
      c.n_components, c.auto_components, c.n_init, c.eigen_tol, c.seed));
}

struct host_graph {
  std::vector<int32_t> rows;
  std::vector<int32_t> cols;
  values vals;
};

struct device_graph {
  device_graph(const resources& res, const host_graph& g)
      : rows(res.r, RR_DTYPE_INT32, {static_cast<int64_t>(g.rows.size())}),
        cols(res.r, RR_DTYPE_INT32, {static_cast<int64_t>(g.cols.size())}),
        vals(
            upload<float>(res, {static_cast<int64_t>(g.vals.size())}, g.vals)) {
    rows.write(g.rows);
    cols.write(g.cols);
  }

  device_array rows;
  device_array cols;
  std::unique_ptr<device_array> vals;
};

}  // namespace cumlrkt::test
