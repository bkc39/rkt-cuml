#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "cluster_fixture.hpp"
#include "cumlrkt/cluster.h"
#include "hdbscan_fixture.hpp"
#include "kmeans_fixture.hpp"
#include "raftrkt/c_api.h"
#include "refusal_fixture.hpp"

namespace cumlrkt::test {
namespace {

constexpr int32_t labels_region = 1;
constexpr int32_t parents_region = 2;
constexpr int32_t children_region = 3;
constexpr int32_t lambdas_region = 4;
constexpr int32_t sizes_region = 5;
constexpr int32_t stabilities_region = 6;
constexpr int32_t selected_region = 7;
constexpr int32_t identical_rows = 20;

struct condensed {
  std::vector<int64_t> parents;
  std::vector<int64_t> children;
  std::vector<float> lambdas;
  std::vector<int64_t> sizes;
  std::vector<int64_t> selected;
};

condensed tree_of(const fitted_model& f) {
  return {f.model.read<int64_t>(parents_region),
          f.model.read<int64_t>(children_region),
          f.model.read<float>(lambdas_region),
          f.model.read<int64_t>(sizes_region),
          f.model.read<int64_t>(selected_region)};
}

TEST(Hdbscan, OutputsHaveTheirShapes) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const fitted_model f(res, three_blobs());
  ASSERT_EQ(f.status, "accepted");
  const int64_t n = f.n;
  const int64_t k = f.clusters();
  ASSERT_EQ(f.model.r.count, model_regions);
  ASSERT_EQ(f.out.r.count, plain_regions);
  EXPECT_GT(f.opaque, 0);
  EXPECT_EQ(f.model.r.regions[selected_region].shape[0], k);
  const std::vector<int64_t> labels = f.model.read<int64_t>(labels_region);
  const std::vector<float> probabilities = f.out.read<float>(0);
  ASSERT_EQ(static_cast<int64_t>(labels.size()), n);
  std::set<int64_t> seen;
  for (int64_t i = 0; i < n; ++i) {
    const int64_t label = labels.at(i);
    const float p = probabilities.at(i);
    EXPECT_GE(label, -1);
    EXPECT_LT(label, k);
    EXPECT_GE(p, 0);
    EXPECT_LE(p, 1);
    if (label == -1) {
      EXPECT_EQ(p, 0);
    } else {
      EXPECT_GT(p, 0);
      seen.insert(label);
    }
  }
  EXPECT_EQ(static_cast<int64_t>(seen.size()), k);
  const condensed t = tree_of(f);
  const auto edges = static_cast<int64_t>(t.parents.size());
  std::vector<int32_t> as_child(static_cast<std::size_t>(n), 0);
  int64_t largest = 0;
  for (int64_t e = 0; e < edges; ++e) {
    EXPECT_GE(t.parents.at(e), n);
    largest = std::max({largest, t.parents.at(e), t.children.at(e)});
    if (t.children.at(e) < n) {
      ++as_child.at(t.children.at(e));
      EXPECT_EQ(t.sizes.at(e), 1);
    }
  }
  EXPECT_EQ(largest, edges);
  EXPECT_TRUE(std::ranges::all_of(as_child, [](int32_t c) { return c == 1; }));
  const std::vector<int64_t> children = f.out.read<int64_t>(1);
  const std::vector<float> distances = f.out.read<float>(2);
  const std::vector<int64_t> sizes = f.out.read<int64_t>(3);
  const std::vector<float> weights = f.out.read<float>(6);
  ASSERT_EQ(static_cast<int64_t>(children.size()), 2 * (n - 1));
  for (int64_t i = 0; i < n - 1; ++i) {
    EXPECT_LE(children.at(2 * i), children.at((2 * i) + 1));
    EXPECT_GE(children.at(2 * i), 0);
    EXPECT_LE(children.at((2 * i) + 1), (2 * n) - 2);
  }
  EXPECT_TRUE(std::ranges::is_sorted(distances));
  EXPECT_EQ(sizes.back(), n);
  EXPECT_EQ(weights, distances);
  EXPECT_EQ(f.out.read<int64_t>(4).size(), static_cast<std::size_t>(n - 1));
  EXPECT_EQ(f.out.read<int64_t>(5).size(), static_cast<std::size_t>(n - 1));
}

TEST(Hdbscan, LabelsDescendFromTheirSelectedClusters) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const fitted_model f(res, three_blobs());
  ASSERT_EQ(f.status, "accepted");
  const condensed t = tree_of(f);
  EXPECT_EQ(std::ranges::adjacent_find(t.selected, std::greater_equal<>{}),
            t.selected.end());
  std::unordered_map<int64_t, int64_t> parent;
  for (std::size_t e = 0; e < t.parents.size(); ++e) {
    parent[t.children.at(e)] = t.parents.at(e);
  }
  const std::vector<int64_t> labels = f.model.read<int64_t>(labels_region);
  for (int64_t i = 0; i < f.n; ++i) {
    std::set<int64_t> ancestors;
    for (auto at = parent.find(i); at != parent.end();
         at = parent.find(at->second)) {
      ancestors.insert(at->second);
    }
    const int64_t label = labels.at(i);
    for (std::size_t j = 0; j < t.selected.size(); ++j) {
      const bool own = label == static_cast<int64_t>(j);
      EXPECT_EQ(ancestors.contains(f.n + t.selected.at(j)), own)
          << "point " << i << " cluster " << j;
    }
  }
}

TEST(Hdbscan, PredictingTheTrainingRowsGivesTheirLabels) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const fitted_model f(res, three_blobs());
  ASSERT_EQ(f.status, "accepted");
  record out(res);
  ASSERT_EQ(
      approximate_predict(res, f.anchor(), &f.x->view, &f.x->view, &out.r),
      "accepted");
  EXPECT_EQ(out.read<int64_t>(0), f.model.read<int64_t>(labels_region));
  record all(res);
  ASSERT_EQ(all_points(res, f.anchor(), &f.x->view, &all.r), "accepted");
  EXPECT_EQ(shapes(all),
            (std::vector<int64_t>{1, RR_DTYPE_FLOAT32, f.n, f.clusters()}));
  record some(res);
  ASSERT_EQ(
      membership_vector(res, f.anchor(), &f.x->view, &f.x->view, &some.r, 7),
      "accepted");
  EXPECT_EQ(shapes(some), shapes(all));
}

TEST(Hdbscan, AReleasedModelRefusesUseAndKeepsItsRegions) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const fitted_model f(res, three_blobs());
  ASSERT_EQ(f.status, "accepted");
  const std::vector<int64_t> labels = f.model.read<int64_t>(labels_region);
  const std::vector<float> stabilities =
      f.model.read<float>(stabilities_region);
  EXPECT_EQ(release_model(res, f.anchor()), "accepted");
  EXPECT_EQ(release_model(res, f.anchor()), "accepted");
  record out(res);
  const std::string released =
      "model: expected a live model, got a released one";
  EXPECT_EQ(
      approximate_predict(res, f.anchor(), &f.x->view, &f.x->view, &out.r),
      released);
  EXPECT_EQ(membership_vector(res, f.anchor(), &f.x->view, &f.x->view, &out.r),
            released);
  EXPECT_EQ(all_points(res, f.anchor(), &f.x->view, &out.r), released);
  expect_untouched(out);
  EXPECT_EQ(f.model.read<int64_t>(labels_region), labels);
  EXPECT_EQ(f.model.read<float>(stabilities_region), stabilities);
}

TEST(Hdbscan, AnotherArrayIsNotAModel) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const fitted_model f(res, three_blobs());
  ASSERT_EQ(f.status, "accepted");
  const device_array foreign(res.r, RR_DTYPE_INT32, {1});
  foreign.write(std::vector<int32_t>{CUML_CLUSTER_MODEL_KIND_HDBSCAN});
  record out(res);
  const std::string message =
      "model: expected a model anchor, got another array";
  EXPECT_EQ(release_model(res, &foreign.view), message);
  EXPECT_EQ(release_model(res, &f.model.r.regions[labels_region]), message);
  EXPECT_EQ(
      approximate_predict(res, &foreign.view, &f.x->view, &f.x->view, &out.r),
      message);
  expect_untouched(out);
}

TEST(Hdbscan, ANoiseModelAnswersEmptyMemberships) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const fitted_model f(res, values(2 * identical_rows, 1.0));
  ASSERT_EQ(f.status, "accepted");
  ASSERT_EQ(f.clusters(), 0);
  EXPECT_EQ(f.model.r.regions[selected_region].shape[0], 0);
  record all(res);
  ASSERT_EQ(all_points(res, f.anchor(), &f.x->view, &all.r), "accepted");
  EXPECT_EQ(shapes(all),
            (std::vector<int64_t>{1, RR_DTYPE_FLOAT32, identical_rows, 0}));
  record some(res);
  ASSERT_EQ(membership_vector(res, f.anchor(), &f.x->view, &f.x->view, &some.r),
            "accepted");
  EXPECT_EQ(shapes(some),
            (std::vector<int64_t>{1, RR_DTYPE_FLOAT32, identical_rows, 0}));
  record predicted(res);
  ASSERT_EQ(approximate_predict(res, f.anchor(), &f.x->view, &f.x->view,
                                &predicted.r),
            "accepted");
  EXPECT_EQ(predicted.read<int64_t>(0),
            std::vector<int64_t>(identical_rows, -1));
  EXPECT_EQ(predicted.read<float>(1), std::vector<float>(identical_rows, 0));
}

}  // namespace
}  // namespace cumlrkt::test
