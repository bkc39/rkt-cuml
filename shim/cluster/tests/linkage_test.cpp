#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "cluster_fixture.hpp"
#include "cumlrkt/cluster.h"
#include "kmeans_fixture.hpp"
#include "linkage_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {
namespace {

constexpr int64_t grid_rows = 5;
constexpr int64_t tie_rows = 8;
constexpr int64_t cloud_rows = 300;
constexpr int64_t cloud_cols = 2;
constexpr uint64_t lcg_a = 6364136223846793005ULL;
constexpr uint64_t lcg_c = 1442695040888963407ULL;
constexpr int lcg_shift = 33;
constexpr double lcg_scale = 0x1p-31;

const values grid{0, 1, 3, 7, 15};
const values permuted_grid{15, 3, 0, 7, 1};
const values grid_children{0, 1, 2, 5, 3, 6, 4, 7};
const values permuted_children{2, 4, 1, 5, 3, 6, 0, 7};
const std::vector<values> grid_labels{{0, 0, 0, 0, 0},
                                      {0, 0, 0, 0, 1},
                                      {0, 0, 0, 2, 1},
                                      {0, 0, 3, 2, 1},
                                      {4, 3, 2, 1, 0}};
const std::vector<values> permuted_labels{{0, 0, 0, 0, 0},
                                          {1, 0, 0, 0, 0},
                                          {2, 0, 0, 1, 0},
                                          {3, 2, 0, 1, 0},
                                          {4, 3, 2, 1, 0}};
const values tie_children{0, 1, 2, 8, 3, 9, 4, 5, 6, 11, 7, 12, 10, 13};
const values tie_labels{1, 1, 1, 1, 0, 0, 0, 2};

struct fitted {
  values labels;
  values children;
  std::vector<int64_t> shapes;
};

struct problem {
  values host;
  int64_t n;
  int64_t d;
};

fitted fit(const resources& res, const problem& p, linkage_call call) {
  const auto x = upload<float>(res, {p.n, p.d}, p.host);
  record out(res);
  call.x = &x->view;
  call.out = &out.r;
  EXPECT_EQ(linkage(res, call), "accepted");
  return {download<int32_t>(out, 0), download<int32_t>(out, 1), shapes(out)};
}

std::vector<int64_t> label_shapes(int64_t n) {
  return {2, RR_DTYPE_INT32, n, RR_DTYPE_INT32, n - 1, 2};
}

values cloud(int64_t n, int64_t d) {
  values host;
  uint64_t state = 1;
  for (int64_t i = 0; i < n * d; ++i) {
    state = (state * lcg_a) + lcg_c;
    host.push_back(static_cast<double>(state >> lcg_shift) * lcg_scale);
  }
  return host;
}

std::string dendrogram_problems(const values& children, int64_t n) {
  std::vector<int> seen(static_cast<std::size_t>((2 * n) - 2), 0);
  for (int64_t i = 0; i + 1 < n; ++i) {
    const double a = children[static_cast<std::size_t>(2 * i)];
    const double b = children[static_cast<std::size_t>((2 * i) + 1)];
    if (!(a < b)) {
      return "row " + std::to_string(i) + " is not ascending";
    }
    if (!(b < static_cast<double>(n + i))) {
      return "row " + std::to_string(i) + " names a later node";
    }
    ++seen[static_cast<std::size_t>(a)];
    ++seen[static_cast<std::size_t>(b)];
  }
  for (std::size_t id = 0; id < seen.size(); ++id) {
    if (seen[id] != 1) {
      return "node " + std::to_string(id) + " appears " +
             std::to_string(seen[id]) + " times";
    }
  }
  return "";
}

std::vector<std::vector<int64_t>> clusters_formed(const values& children,
                                                  int64_t n) {
  std::vector<std::vector<int64_t>> members;
  members.reserve(static_cast<std::size_t>((2 * n) - 1));
  for (int64_t i = 0; i < n; ++i) {
    members.push_back({i});
  }
  for (std::size_t row = 0; row + 1 < static_cast<std::size_t>(n); ++row) {
    std::vector<int64_t> joined =
        members[static_cast<std::size_t>(children[2 * row])];
    const std::vector<int64_t>& other =
        members[static_cast<std::size_t>(children[(2 * row) + 1])];
    joined.insert(joined.end(), other.begin(), other.end());
    std::ranges::sort(joined);
    members.push_back(joined);
  }
  members.erase(members.begin(), members.begin() + n);
  std::ranges::sort(members);
  return members;
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

void expect_grid_cut(const resources& res, const linkage_call& call) {
  const auto k = static_cast<std::size_t>(call.n_clusters);
  const fitted f = fit(res, {grid, grid_rows, 1}, call);
  EXPECT_EQ(f.children, grid_children);
  EXPECT_EQ(f.labels, grid_labels[k - 1]);
  EXPECT_EQ(f.shapes, label_shapes(grid_rows));
  const fitted p = fit(res, {permuted_grid, grid_rows, 1}, call);
  EXPECT_EQ(p.children, permuted_children);
  EXPECT_EQ(p.labels, permuted_labels[k - 1]);
}

TEST(Linkage, HandGridChildrenAndLabels) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (const int32_t connectivity : {pairwise, knn}) {
    for (int64_t k = 1; k <= grid_rows; ++k) {
      SCOPED_TRACE(std::to_string(connectivity) + " " + std::to_string(k));
      expect_grid_cut(res, {.n_clusters = k, .connectivity = connectivity});
    }
  }
}

TEST(Linkage, ChildRowsAscendAndFormADendrogram) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const values host = cloud(cloud_rows, cloud_cols);
  for (const int32_t connectivity : {pairwise, knn}) {
    SCOPED_TRACE(connectivity);
    const fitted f = fit(res, {host, cloud_rows, cloud_cols},
                         {.connectivity = connectivity});
    EXPECT_EQ(dendrogram_problems(f.children, cloud_rows), "");
  }
}

void expect_one_and_two_rows(const resources& res, int32_t connectivity) {
  const problem two{{1, 2, 5, 3}, 2, 2};
  const fitted one =
      fit(res, {{3}, 1, 1}, {.n_clusters = 1, .connectivity = connectivity});
  EXPECT_EQ(one.labels, values{0});
  EXPECT_EQ(one.children, values{});
  EXPECT_EQ(one.shapes, label_shapes(1));
  const fitted joined =
      fit(res, two, {.n_clusters = 1, .connectivity = connectivity});
  EXPECT_EQ(joined.children, (values{0, 1}));
  EXPECT_EQ(joined.labels, (values{0, 0}));
  const fitted apart = fit(res, two, {.connectivity = connectivity});
  EXPECT_EQ(apart.labels, (values{1, 0}));
}

TEST(Linkage, OneAndTwoRows) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (const int32_t connectivity : {pairwise, knn}) {
    SCOPED_TRACE(connectivity);
    expect_one_and_two_rows(res, connectivity);
  }
}

TEST(Linkage, CompleteKnnGraphIsPairwise) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const problem points{cloud(cloud_rows, cloud_cols), cloud_rows, cloud_cols};
  const auto n = static_cast<int32_t>(cloud_rows);
  for (const int64_t k : {1, 2, 3, 7, 19, 60, 151, 299, 300}) {
    SCOPED_TRACE(k);
    const fitted p =
        fit(res, points, {.n_clusters = k, .connectivity = pairwise});
    const fitted q =
        fit(res, points, {.n_clusters = k, .connectivity = knn, .c = n});
    EXPECT_EQ(clusters_formed(q.children, cloud_rows),
              clusters_formed(p.children, cloud_rows));
    EXPECT_EQ(canonical(q.labels), canonical(p.labels));
  }
}

TEST(Linkage, TiesComeInCumlsOrder) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  values line;
  for (int64_t i = 0; i < tie_rows; ++i) {
    line.push_back(static_cast<double>(i));
  }
  for (const int32_t connectivity : {pairwise, knn}) {
    SCOPED_TRACE(connectivity);
    const fitted f = fit(res, {line, tie_rows, 1},
                         {.n_clusters = 3, .connectivity = connectivity});
    EXPECT_EQ(f.children, tie_children);
    EXPECT_EQ(f.labels, tie_labels);
  }
}

}  // namespace
}  // namespace cumlrkt::test
