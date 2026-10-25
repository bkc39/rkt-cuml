#include "cumlrkt/metrics.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include "metrics_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {
namespace {

constexpr int32_t euclidean = CUML_METRICS_METRIC_EUCLIDEAN;
constexpr int64_t chunk = 40000;
constexpr int64_t neighbours = 5;
constexpr int64_t batch = 512;
constexpr int64_t grid_rows = 20;

TEST(Metrics, Version) {
  EXPECT_EQ(std::string(cuml_metrics_libcuml_version()), "26.08.00");
}

TEST(Metrics, ContingencyCountsSortedLabels) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<int32_t> truth(res, {0, 0, 1, 1, -1}, {5});
  const filled<int64_t> found(res, {5, 5, 7, 7, 7}, {5});
  record out(res);
  ASSERT_EQ(outcome(cuml_metrics_contingency_matrix(res.handle, truth.view(),
                                                    found.view(), &out.r)),
            "accepted");
  ASSERT_EQ(out.r.count, 3);
  const rr_view* regions = out.r.regions;
  EXPECT_EQ((std::vector<int64_t>{regions[0].dtype, regions[0].shape[0],
                                  regions[0].shape[1], regions[1].dtype,
                                  regions[2].dtype}),
            (std::vector<int64_t>{RR_DTYPE_INT64, 3, 2, RR_DTYPE_INT32,
                                  RR_DTYPE_INT64}));
  EXPECT_EQ(
      std::tuple(out.read<int64_t>(0), out.read<int32_t>(1),
                 out.read<int64_t>(2)),
      std::tuple(std::vector<int64_t>{0, 1, 2, 0, 0, 2},
                 std::vector<int32_t>{-1, 0, 1}, std::vector<int64_t>{5, 7}));
}

TEST(Metrics, ContingencyOfNothingIsEmpty) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<int32_t> none(res, {}, {0});
  record out(res);
  ASSERT_EQ(outcome(cuml_metrics_contingency_matrix(res.handle, none.view(),
                                                    none.view(), &out.r)),
            "accepted");
  ASSERT_EQ(out.r.count, 3);
  EXPECT_EQ(out.r.regions[0].shape[0], 0);
  EXPECT_EQ(out.r.regions[0].shape[1], 0);
  EXPECT_EQ(out.r.regions[1].shape[0], 0);
}

double ari(const resources& res, const rr_view* a, const rr_view* b) {
  double score = -2;
  EXPECT_EQ(outcome(cuml_metrics_adjusted_rand_score(res.handle, a, b, &score)),
            "accepted");
  return score;
}

TEST(Metrics, AdjustedRandDegenerateCases) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<int32_t> a(res, {0, 0, 1, 1, 2}, {5});
  const filled<int64_t> relabelled(
      res, {-1, -1, int64_t{1} << 40, int64_t{1} << 40, 3}, {5});
  const filled<int32_t> one(res, {4, 4, 4, 4, 4}, {5});
  const filled<int32_t> singletons(res, {0, 1, 2, 3, 4}, {5});
  const filled<int32_t> none(res, {}, {0});
  const filled<int64_t> single(res, {9}, {1});
  EXPECT_EQ(ari(res, a.view(), relabelled.view()), 1.0);
  EXPECT_EQ(ari(res, one.view(), one.view()), 1.0);
  EXPECT_EQ(ari(res, one.view(), singletons.view()), 0.0);
  EXPECT_EQ(ari(res, none.view(), none.view()), 1.0);
  EXPECT_EQ(ari(res, single.view(), single.view()), 1.0);
}

TEST(Metrics, SilhouetteSingletonScoresZero) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> x(res, {0, 0, 0, 1, 1, 0, 10, 10, 10, 11, 20, 20},
                         {6, 2});
  const filled<int32_t> labels(res, {0, 0, 0, 1, 1, 2}, {6});
  record out(res);
  double score = 0;
  ASSERT_EQ(outcome(cuml_metrics_silhouette_samples(res.handle, x.view(),
                                                    labels.view(), &out.r,
                                                    euclidean, chunk, &score)),
            "accepted");
  const std::vector<double> samples = out.read<double>(0);
  ASSERT_EQ(samples.size(), 6U);
  EXPECT_TRUE(samples[5] == 0.0 && std::ranges::all_of(samples, [](double s) {
                return s >= -1.0 && s <= 1.0;
              }));
  double plain = 0;
  ASSERT_EQ(outcome(cuml_metrics_silhouette_score(
                res.handle, x.view(), labels.view(), euclidean, chunk, &plain)),
            "accepted");
  EXPECT_NEAR(plain, score, 1e-12);
}

TEST(Metrics, TrustworthinessOfItselfIsOne) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  std::vector<float> host;
  for (int64_t i = 0; i < grid_rows; ++i) {
    host.push_back(static_cast<float>(i * i));
    host.push_back(static_cast<float>((3 * i) + 1));
  }
  const filled<float> x(res, host, {grid_rows, 2});
  double score = 0;
  ASSERT_EQ(outcome(cuml_metrics_trustworthiness(res.handle, x.view(), x.view(),
                                                 neighbours, euclidean, batch,
                                                 &score)),
            "accepted");
  EXPECT_EQ(score, 1.0);
}

}  // namespace
}  // namespace cumlrkt::test
