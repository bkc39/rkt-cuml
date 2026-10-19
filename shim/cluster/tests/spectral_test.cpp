#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "cluster_fixture.hpp"
#include "cumlrkt/cluster.h"
#include "kmeans_fixture.hpp"
#include "raftrkt/c_api.h"
#include "spectral_fixture.hpp"

namespace cumlrkt::test {
namespace {

constexpr int64_t groups = 3;
constexpr int64_t per_group = 40;
constexpr int64_t blob_rows = groups * per_group;
constexpr int64_t blob_cols = 2;
constexpr double gap = 50;
constexpr int64_t clique = 10;
constexpr uint64_t lcg_a = 6364136223846793005ULL;
constexpr uint64_t lcg_c = 1442695040888963407ULL;
constexpr int lcg_shift = 33;
constexpr double lcg_scale = 0x1p-31;

values blobs() {
  values host;
  uint64_t state = 1;
  for (int64_t i = 0; i < blob_rows; ++i) {
    const int64_t group = i / per_group;
    for (int64_t j = 0; j < blob_cols; ++j) {
      state = (state * lcg_a) + lcg_c;
      host.push_back((gap * static_cast<double>(group)) +
                     (static_cast<double>(state >> lcg_shift) * lcg_scale));
    }
  }
  return host;
}

values truth(int64_t n_groups, int64_t size) {
  values labels;
  for (int64_t i = 0; i < n_groups * size; ++i) {
    const int64_t group = i / size;
    labels.push_back(static_cast<double>(group));
  }
  return labels;
}

values canonical(const values& labels) {
  std::map<double, double> names;
  values renamed;
  for (const double label : labels) {
    const auto [at, fresh] =
        names.try_emplace(label, static_cast<double>(names.size()));
    renamed.push_back(at->second);
  }
  return renamed;
}

double squared(const values& x, int64_t d, int64_t i, int64_t j) {
  double sum = 0;
  for (int64_t a = 0; a < d; ++a) {
    const double diff = x[static_cast<std::size_t>((i * d) + a)] -
                        x[static_cast<std::size_t>((j * d) + a)];
    sum += diff * diff;
  }
  return sum;
}

host_graph knn_graph(const values& x, int64_t k) {
  const int64_t n = blob_rows;
  const int64_t d = blob_cols;
  std::vector<double> a(static_cast<std::size_t>(n * n), 0);
  for (int64_t i = 0; i < n; ++i) {
    std::vector<int64_t> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::ranges::stable_sort(order, [&](int64_t p, int64_t q) {
      return squared(x, d, i, p) < squared(x, d, i, q);
    });
    for (int64_t m = 0; m < k; ++m) {
      a[static_cast<std::size_t>((i * n) +
                                 order[static_cast<std::size_t>(m)])] = 1;
    }
  }
  host_graph g;
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      const double w = 0.5 * (a[static_cast<std::size_t>((i * n) + j)] +
                              a[static_cast<std::size_t>((j * n) + i)]);
      if (i != j && w > 0) {
        g.rows.push_back(static_cast<int32_t>(i));
        g.cols.push_back(static_cast<int32_t>(j));
        g.vals.push_back(w);
      }
    }
  }
  return g;
}

host_graph cliques(int64_t count) {
  host_graph g;
  for (int64_t i = 0; i < count * clique; ++i) {
    for (int64_t j = 0; j < count * clique; ++j) {
      if (i != j && i / clique == j / clique) {
        g.rows.push_back(static_cast<int32_t>(i));
        g.cols.push_back(static_cast<int32_t>(j));
        g.vals.push_back(1);
      }
    }
  }
  return g;
}

struct fitted {
  std::string status;
  values labels;
  std::vector<int64_t> shapes;
};

fitted fit_dense(const resources& res, const values& host, int64_t n, int64_t d,
                 spectral_call call) {
  const auto x = upload<float>(res, {n, d}, host);
  record out(res);
  call.x = &x->view;
  call.out = &out.r;
  const std::string status = spectral(res, call);
  if (status != "accepted") {
    return {status, {}, {}};
  }
  return {status, download<int32_t>(out, 0), shapes(out)};
}

fitted fit_graph(const resources& res, const host_graph& host, int64_t n,
                 graph_call call) {
  const device_graph g(res, host);
  record out(res);
  call.rows = &g.rows.view;
  call.cols = &g.cols.view;
  call.vals = &g.vals->view;
  call.n_samples = n;
  call.out = &out.r;
  const std::string status = spectral_graph(res, call);
  if (status != "accepted") {
    return {status, {}, {}};
  }
  return {status, download<int32_t>(out, 0), shapes(out)};
}

void expect_labels_in_range(const fitted& f, int64_t k) {
  EXPECT_EQ(f.status, "accepted");
  EXPECT_EQ(f.shapes, (std::vector<int64_t>{1, RR_DTYPE_INT32, blob_rows}));
  for (const double label : f.labels) {
    EXPECT_GE(label, 0);
    EXPECT_LT(label, static_cast<double>(k));
  }
}

TEST(Spectral, LabelsAreInt32InRangeOnBothPaths) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const values x = blobs();
  const host_graph g = knn_graph(x, default_neighbours);
  for (const int64_t k : {2, 3, 5, 8}) {
    SCOPED_TRACE(k);
    expect_labels_in_range(
        fit_dense(res, x, blob_rows, blob_cols, {.n_clusters = k}), k);
    expect_labels_in_range(fit_graph(res, g, blob_rows, {.n_clusters = k}), k);
  }
}

TEST(Spectral, OneClusterIsAllZeros) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const values x = blobs();
  const values zeros(static_cast<std::size_t>(blob_rows), 0);
  EXPECT_EQ(fit_dense(res, x, blob_rows, blob_cols, {.n_clusters = 1}).labels,
            zeros);
  const host_graph g = knn_graph(x, default_neighbours);
  EXPECT_EQ(fit_graph(res, g, blob_rows, {.n_clusters = 1}).labels, zeros);
}

TEST(Spectral, BlobsArePartitionedAndDenseEqualsItsGraph) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const values x = blobs();
  const host_graph g = knn_graph(x, default_neighbours);
  for (const uint64_t seed : {0ULL, 1ULL, 12345ULL}) {
    SCOPED_TRACE(seed);
    const fitted dense = fit_dense(res, x, blob_rows, blob_cols,
                                   {.n_clusters = groups, .seed = seed});
    const fitted graph =
        fit_graph(res, g, blob_rows, {.n_clusters = groups, .seed = seed});
    EXPECT_EQ(canonical(dense.labels), truth(groups, per_group));
    EXPECT_EQ(dense.labels, graph.labels);
  }
}

TEST(Spectral, PinDisconnectedCliquesDoNotConverge) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const fitted two = fit_graph(res, cliques(2), 2 * clique, {.n_clusters = 2});
  EXPECT_NE(two.status.find("eigensolver couldn't converge"), std::string::npos)
      << two.status;
  const fitted three =
      fit_graph(res, cliques(3), 3 * clique, {.n_clusters = 3});
  EXPECT_EQ(three.status, "accepted");
  EXPECT_EQ(canonical(three.labels), truth(3, clique));
}

}  // namespace
}  // namespace cumlrkt::test
