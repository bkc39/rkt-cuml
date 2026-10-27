#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "cumlrkt/manifold.h"
#include "manifold_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {
namespace {

constexpr int64_t rows = 40;
constexpr int64_t cols = 3;
constexpr int64_t few_rows = 10;
constexpr int64_t short_run = 300;
constexpr float inside_x = 1278245504.0F;
constexpr float inside_y = 819714240.0F;
constexpr float inside_manhattan = 2173969371284635648.0F;
constexpr double inside_pca = 0x1p61 - 0x1p37;
constexpr int64_t pca_rows = 16;

bool all_finite(const std::vector<float>& host) {
  return std::ranges::all_of(host, [](float v) { return std::isfinite(v); });
}

std::vector<float> fitted(const resources& res, const rr_view* x,
                          const tsne_options& o = {},
                          tsne_stats* stats = nullptr) {
  record out(res);
  EXPECT_EQ(outcome(tsne_fit(res, x, out, o, stats)), "accepted");
  EXPECT_EQ(out.r.count, 1);
  EXPECT_EQ(out.r.regions[0].rank, 2);
  EXPECT_EQ(out.r.regions[0].dtype, RR_DTYPE_FLOAT32);
  EXPECT_EQ(out.r.regions[0].shape[1], 2);
  return out.r.count == 1 ? out.read<float>(0) : std::vector<float>{};
}

void check_repeats(const resources& res, const rr_view* x, int32_t init) {
  tsne_stats stats;
  const std::vector<float> first =
      fitted(res, x, {.max_iter = short_run, .init = init}, &stats);
  EXPECT_EQ(first.size(), static_cast<std::size_t>(rows * 2));
  EXPECT_TRUE(all_finite(first));
  EXPECT_EQ(stats.n_iter, short_run);
  EXPECT_GE(stats.kl_divergence, 0);
  tsne_stats again;
  EXPECT_EQ(first,
            fitted(res, x, {.max_iter = short_run, .init = init}, &again));
  EXPECT_EQ(again.kl_divergence, stats.kl_divergence);
}

TEST(TsneFit, FftIsFiniteAndRepeatsBitwise) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, blobs(rows, cols), {rows, cols});
  check_repeats(res, x.view(), CUML_MANIFOLD_TSNE_INIT_RANDOM);
  check_repeats(res, x.view(), CUML_MANIFOLD_TSNE_INIT_PCA);
}

TEST(TsneFit, ExactAndEveryMetricAreFinite) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, blobs(rows, cols), {rows, cols});
  for (const int32_t metric :
       {CUML_MANIFOLD_METRIC_EUCLIDEAN, CUML_MANIFOLD_METRIC_SQEUCLIDEAN,
        CUML_MANIFOLD_METRIC_MANHATTAN, CUML_MANIFOLD_METRIC_CHEBYSHEV,
        CUML_MANIFOLD_METRIC_COSINE}) {
    for (const int32_t method :
         {CUML_MANIFOLD_TSNE_FFT, CUML_MANIFOLD_TSNE_EXACT}) {
      EXPECT_TRUE(all_finite(
          fitted(res, x.view(),
                 {.max_iter = short_run, .method = method, .metric = metric})))
          << metric << " " << method;
    }
  }
}

void expect_finite_under_both_methods(const resources& res, const rr_view* x,
                                      tsne_options o) {
  for (const int32_t method :
       {CUML_MANIFOLD_TSNE_FFT, CUML_MANIFOLD_TSNE_EXACT}) {
    o.method = method;
    EXPECT_TRUE(all_finite(fitted(res, x, o))) << method;
  }
}

std::vector<float> opposite_rows(const std::array<float, cols>& point) {
  std::vector<float> host = blobs(rows, cols);
  const std::size_t width = point.size();
  for (std::size_t j = 0; j < width; ++j) {
    host[j] = point[j];
    host[width + j] = -point[j];
  }
  return host;
}

TEST(TsneFit, OppositeTwoColumnRowsJustInsideTheSqeuclideanLimitAreFinite) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, opposite_rows({inside_x, inside_y, 0}),
                        {rows, cols});
  expect_finite_under_both_methods(
      res, x.view(),
      {.max_iter = short_run, .metric = CUML_MANIFOLD_METRIC_SQEUCLIDEAN});
}

TEST(TsneFit, OppositeRowsJustInsideTheManhattanLimitAreFinite) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(
      res,
      opposite_rows({inside_manhattan, inside_manhattan, inside_manhattan}),
      {rows, cols});
  expect_finite_under_both_methods(
      res, x.view(),
      {.max_iter = short_run, .metric = CUML_MANIFOLD_METRIC_MANHATTAN});
}

TEST(TsneFit, RowsJustInsideThePcaSpreadLimitAreFinite) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, spread_rows(inside_pca), {pca_rows, cols});
  expect_finite_under_both_methods(res, x.view(),
                                   {.perplexity = 3,
                                    .max_iter = short_run,
                                    .init = CUML_MANIFOLD_TSNE_INIT_PCA});
}

TEST(TsneFit, PcaInitLeavesXUntouched) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const std::vector<float> host = blobs(rows, cols);
  const filled<float> x(res, host, {rows, cols});
  for (const int32_t method :
       {CUML_MANIFOLD_TSNE_FFT, CUML_MANIFOLD_TSNE_EXACT}) {
    fitted(res, x.view(),
           {.max_iter = short_run,
            .method = method,
            .init = CUML_MANIFOLD_TSNE_INIT_PCA});
    std::vector<float> after(host.size());
    EXPECT_EQ(
        rr_copy_d2h(after.data(), x.array.buffer, after.size() * sizeof(float)),
        RR_OK)
        << rr_last_error();
    EXPECT_EQ(after, host);
  }
}

TEST(TsneFit, NeighboursClampToTheRows) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, blobs(few_rows, cols), {few_rows, cols});
  EXPECT_TRUE(all_finite(fitted(res, x.view(), {.max_iter = short_run})));
}

TEST(TsneFit, XOnAnotherDeviceIsRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, blobs(rows, cols), {rows, cols});
  rr_view elsewhere = *x.view();
  elsewhere.device = 1;
  record out(res);
  EXPECT_EQ(outcome(tsne_fit(res, &elsewhere, out)),
            "X: on device 1, but the resources are on device 0");
  expect_untouched(out);
}

}  // namespace
}  // namespace cumlrkt::test
