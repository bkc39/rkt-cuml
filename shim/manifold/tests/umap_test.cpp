#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "cumlrkt/manifold.h"
#include "manifold_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {
namespace {

constexpr int64_t rows = 40;
constexpr int64_t cols = 3;
constexpr int64_t fresh = 10;
constexpr int64_t nn_rows = 200;
constexpr int64_t past_rows = 100;

bool all_finite(const std::vector<float>& host) {
  return std::ranges::all_of(host, [](float v) { return std::isfinite(v); });
}

std::vector<float> fitted(const resources& res, const rr_view* x,
                          const rr_view* y, const umap_options& o = {}) {
  record out(res);
  EXPECT_EQ(outcome(umap_fit(res, x, y, out, o)), "accepted");
  EXPECT_EQ(out.r.count, 1);
  EXPECT_EQ(out.r.regions[0].rank, 2);
  EXPECT_EQ(out.r.regions[0].dtype, RR_DTYPE_FLOAT32);
  return out.r.count == 1 ? out.read<float>(0) : std::vector<float>{};
}

TEST(UmapFit, RandomInitIsFiniteAndRepeatsBitwise) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, blobs(rows, cols), {rows, cols});
  const std::vector<float> first = fitted(res, x.view(), nullptr);
  EXPECT_EQ(first.size(), static_cast<std::size_t>(rows * 2));
  EXPECT_TRUE(all_finite(first));
  EXPECT_EQ(first, fitted(res, x.view(), nullptr));
}

TEST(UmapFit, SpectralInitAndEveryMetricAreFinite) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, blobs(rows, cols), {rows, cols});
  for (const int32_t metric :
       {CUML_MANIFOLD_METRIC_EUCLIDEAN, CUML_MANIFOLD_METRIC_SQEUCLIDEAN,
        CUML_MANIFOLD_METRIC_MANHATTAN, CUML_MANIFOLD_METRIC_CHEBYSHEV,
        CUML_MANIFOLD_METRIC_COSINE}) {
    EXPECT_TRUE(all_finite(fitted(
        res, x.view(), nullptr,
        {.init = CUML_MANIFOLD_INIT_SPECTRAL, .metric = metric})))
        << metric;
  }
}

TEST(UmapFit, LabelsChangeTheEmbedding) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, blobs(rows, cols), {rows, cols});
  std::vector<int64_t> host(static_cast<std::size_t>(rows));
  for (int64_t i = 0; i < rows; ++i) {
    host[static_cast<std::size_t>(i)] = i % 3 == 0 ? -1 : (i % 2) * 1000;
  }
  const filled<int64_t> y(res, host, {rows});
  const std::vector<float> supervised = fitted(res, x.view(), y.view());
  EXPECT_TRUE(all_finite(supervised));
  EXPECT_EQ(supervised, fitted(res, x.view(), y.view()));
  EXPECT_NE(supervised, fitted(res, x.view(), nullptr));
}

TEST(UmapFit, NeighboursClampToTheRows) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, blobs(rows, cols), {rows, cols});
  EXPECT_EQ(fitted(res, x.view(), nullptr, {.n_neighbors = past_rows}),
            fitted(res, x.view(), nullptr, {.n_neighbors = rows}));
}

TEST(UmapFit, NnDescentIsFinite) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, blobs(nn_rows, cols), {nn_rows, cols});
  EXPECT_TRUE(all_finite(fitted(
      res, x.view(), nullptr, {.build_algo = CUML_MANIFOLD_BUILD_NN_DESCENT})));
}

TEST(UmapTransform, NewRowsAreFiniteAndRepeatBitwise) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, blobs(rows, cols), {rows, cols});
  record trained(res);
  ASSERT_EQ(outcome(umap_fit(res, x.view(), nullptr, trained)), "accepted");
  const filled<float> fresh_x(res, blobs(fresh, cols), {fresh, cols});
  const umap_trained t{x.view(), &trained.r.regions[0]};
  record first(res);
  record second(res);
  ASSERT_EQ(outcome(umap_transform(res, fresh_x.view(), t, first)),
            "accepted");
  ASSERT_EQ(outcome(umap_transform(res, fresh_x.view(), t, second)),
            "accepted");
  EXPECT_EQ(first.r.regions[0].shape[0], fresh);
  EXPECT_EQ(first.r.regions[0].shape[1], 2);
  EXPECT_TRUE(all_finite(first.read<float>(0)));
  EXPECT_EQ(first.read<float>(0), second.read<float>(0));
}

}  // namespace
}  // namespace cumlrkt::test
