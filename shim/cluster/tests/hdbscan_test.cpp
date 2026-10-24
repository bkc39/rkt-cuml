#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
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
constexpr int32_t probabilities_region = 0;
constexpr int32_t linkage_region = 1;
constexpr int32_t distances_region = 2;
constexpr int32_t linkage_sizes_region = 3;
constexpr int32_t sources_region = 4;
constexpr int32_t destinations_region = 5;
constexpr int32_t weights_region = 6;
constexpr int32_t identical_rows = 20;
constexpr int64_t small_batch = 7;
constexpr std::size_t noise_values = std::size_t{2} * identical_rows;

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

std::string label_problems(const fitted_model& f) {
  const std::vector<int64_t> labels = f.model.read<int64_t>(labels_region);
  const std::vector<float> probabilities =
      f.out.read<float>(probabilities_region);
  const int64_t k = f.clusters();
  if (static_cast<int64_t>(labels.size()) != f.n) {
    return std::format("{} labels for {} rows; ", labels.size(), f.n);
  }
  std::string problems;
  std::set<int64_t> seen;
  for (int64_t i = 0; i < f.n; ++i) {
    const int64_t label = labels.at(i);
    const float p = probabilities.at(i);
    const bool noise = label == -1;
    const bool fits = noise ? p == 0 : p > 0;
    if (label < -1 || label >= k || p < 0 || p > 1 || !fits) {
      problems +=
          std::format("row {}: label {}, probability {}; ", i, label, p);
    }
    if (!noise) {
      seen.insert(label);
    }
  }
  if (static_cast<int64_t>(seen.size()) != k) {
    problems += std::format("{} of {} labels used; ", seen.size(), k);
  }
  return problems;
}

std::string condensed_problems(const fitted_model& f) {
  const condensed t = tree_of(f);
  const int64_t n = f.n;
  const auto edges = static_cast<int64_t>(t.parents.size());
  std::string problems;
  std::vector<int32_t> as_child(static_cast<std::size_t>(n), 0);
  int64_t largest = 0;
  for (int64_t e = 0; e < edges; ++e) {
    const int64_t child = t.children.at(e);
    if (t.parents.at(e) < n) {
      problems += std::format("edge {}: a point as parent; ", e);
    }
    largest = std::max({largest, t.parents.at(e), child});
    if (child < n) {
      ++as_child.at(child);
    }
    if (child < n && t.sizes.at(e) != 1) {
      problems += std::format("edge {}: point of size {}; ", e, t.sizes.at(e));
    }
  }
  if (largest != edges) {
    problems += std::format("largest node {} for {} edges; ", largest, edges);
  }
  if (!std::ranges::all_of(as_child, [](int32_t c) { return c == 1; })) {
    problems += "a point is not a child exactly once; ";
  }
  return problems;
}

std::string plain_problems(const fitted_model& f) {
  const int64_t n = f.n;
  const auto merges = static_cast<std::size_t>(n - 1);
  const std::vector<int64_t> children = f.out.read<int64_t>(linkage_region);
  const std::vector<float> distances = f.out.read<float>(distances_region);
  const std::vector<int64_t> sizes = f.out.read<int64_t>(linkage_sizes_region);
  if (children.size() != 2 * merges) {
    return std::format("{} children for {} merges; ", children.size(), merges);
  }
  std::string problems;
  for (int64_t i = 0; i < n - 1; ++i) {
    const int64_t low = children.at(2 * i);
    const int64_t high = children.at((2 * i) + 1);
    if (low > high || low < 0 || high > (2 * n) - 2) {
      problems += std::format("merge {}: children {} {}; ", i, low, high);
    }
  }
  if (!std::ranges::is_sorted(distances)) {
    problems += "distances descend; ";
  }
  if (sizes.back() != n) {
    problems += std::format("last merge of size {}; ", sizes.back());
  }
  if (f.out.read<float>(weights_region) != distances) {
    problems += "weights differ from distances; ";
  }
  if (f.out.read<int64_t>(sources_region).size() != merges ||
      f.out.read<int64_t>(destinations_region).size() != merges) {
    problems += "spanning tree ends are not n - 1 long; ";
  }
  return problems;
}

std::string descent_problems(const fitted_model& f) {
  const condensed t = tree_of(f);
  std::unordered_map<int64_t, int64_t> parent;
  for (std::size_t e = 0; e < t.parents.size(); ++e) {
    parent[t.children.at(e)] = t.parents.at(e);
  }
  const std::vector<int64_t> labels = f.model.read<int64_t>(labels_region);
  std::string problems;
  for (int64_t i = 0; i < f.n; ++i) {
    std::set<int64_t> ancestors;
    for (auto at = parent.find(i); at != parent.end();
         at = parent.find(at->second)) {
      ancestors.insert(at->second);
    }
    for (std::size_t j = 0; j < t.selected.size(); ++j) {
      const bool own = labels.at(i) == static_cast<int64_t>(j);
      if (ancestors.contains(f.n + t.selected.at(j)) != own) {
        problems += std::format("point {} cluster {}; ", i, j);
      }
    }
  }
  return problems;
}

std::string opaque_problems(const fitted_model& f) {
  const int64_t edges = f.model.r.regions[2].shape[0];
  const int64_t k = f.clusters();
  const int64_t least =
      (4 * f.n) + (8 * ((2 * k) + edges + 2)) + (4 * (edges + 1 - f.n));
  const int64_t most = least + (8 * f.n);
  if (f.opaque >= least && f.opaque <= most) {
    return "";
  }
  return std::format("opaque {} outside [{}, {}]; ", f.opaque, least, most);
}

TEST(Hdbscan, OutputsHaveTheirShapes) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const fitted_model f(res, three_blobs());
  ASSERT_EQ(f.status, "accepted");
  ASSERT_EQ((std::vector<int32_t>{f.model.r.count, f.out.r.count}),
            (std::vector<int32_t>{model_regions, plain_regions}));
  EXPECT_EQ(f.model.r.regions[selected_region].shape[0], f.clusters());
  EXPECT_EQ(opaque_problems(f) + label_problems(f) + condensed_problems(f) +
                plain_problems(f),
            "");
}

TEST(Hdbscan, LabelsDescendFromTheirSelectedClusters) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const fitted_model f(res, three_blobs());
  ASSERT_EQ(f.status, "accepted");
  const condensed t = tree_of(f);
  EXPECT_EQ(std::ranges::adjacent_find(t.selected, std::greater_equal<>{}),
            t.selected.end());
  EXPECT_EQ(descent_problems(f), "");
}

TEST(Hdbscan, PredictingTheTrainingRowsGivesTheirLabels) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const fitted_model f(res, three_blobs());
  ASSERT_EQ(f.status, "accepted");
  record out(res);
  record all(res);
  record some(res);
  const std::vector<std::string> calls{
      approximate_predict(res, f.anchor(), &f.x->view, &f.x->view, &out.r),
      all_points(res, f.anchor(), &f.x->view, &all.r),
      membership_vector(res, f.anchor(), &f.x->view, &f.x->view, &some.r,
                        small_batch)};
  ASSERT_EQ(calls, std::vector<std::string>(3, "accepted"));
  EXPECT_EQ(out.read<int64_t>(0), f.model.read<int64_t>(labels_region));
  const std::vector<int64_t> every{1, RR_DTYPE_FLOAT32, f.n, f.clusters()};
  EXPECT_EQ((std::vector{shapes(all), shapes(some)}),
            (std::vector{every, every}));
}

TEST(Hdbscan, AReleasedModelRefusesUseAndKeepsItsRegions) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const fitted_model f(res, three_blobs());
  ASSERT_EQ(f.status, "accepted");
  const std::vector<int64_t> labels = f.model.read<int64_t>(labels_region);
  const std::vector<float> stabilities =
      f.model.read<float>(stabilities_region);
  const std::vector<std::string> releases{release_model(res, f.anchor()),
                                          release_model(res, f.anchor())};
  EXPECT_EQ(releases, std::vector<std::string>(2, "accepted"));
  record out(res);
  const std::vector<std::string> uses{
      approximate_predict(res, f.anchor(), &f.x->view, &f.x->view, &out.r),
      membership_vector(res, f.anchor(), &f.x->view, &f.x->view, &out.r),
      all_points(res, f.anchor(), &f.x->view, &out.r)};
  EXPECT_EQ(uses, std::vector<std::string>(
                      3, "model: expected a live model, got a released one"));
  expect_untouched(out);
  EXPECT_TRUE(f.model.read<int64_t>(labels_region) == labels &&
              f.model.read<float>(stabilities_region) == stabilities);
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
  const fitted_model f(res, values(noise_values, 1.0));
  ASSERT_EQ(f.status, "accepted");
  ASSERT_EQ((std::vector<int64_t>{f.clusters(),
                                  f.model.r.regions[selected_region].shape[0]}),
            (std::vector<int64_t>{0, 0}));
  record all(res);
  record some(res);
  record predicted(res);
  const std::vector<std::string> calls{
      all_points(res, f.anchor(), &f.x->view, &all.r),
      membership_vector(res, f.anchor(), &f.x->view, &f.x->view, &some.r),
      approximate_predict(res, f.anchor(), &f.x->view, &f.x->view,
                          &predicted.r)};
  ASSERT_EQ(calls, std::vector<std::string>(3, "accepted"));
  const std::vector<int64_t> empty{1, RR_DTYPE_FLOAT32, identical_rows, 0};
  EXPECT_EQ((std::vector{shapes(all), shapes(some)}),
            (std::vector{empty, empty}));
  EXPECT_TRUE(
      predicted.read<int64_t>(0) == std::vector<int64_t>(identical_rows, -1) &&
      predicted.read<float>(1) == std::vector<float>(identical_rows, 0));
}

}  // namespace
}  // namespace cumlrkt::test
