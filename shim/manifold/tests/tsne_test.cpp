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
constexpr int64_t few_rows = 10;
constexpr int64_t short_run = 300;
constexpr float squarable = 0x1p31F - 0x1p7F;

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

TEST(TsneFit, OppositeRowsAtTheLargestSquarableNormAreFinite) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  std::vector<float> host = blobs(rows, cols);
  std::fill_n(host.begin(), 2 * cols, 0.0F);
  host[0] = squarable;
  host[cols] = -squarable;
  const filled<float> x(res, host, {rows, cols});
  for (const int32_t method :
       {CUML_MANIFOLD_TSNE_FFT, CUML_MANIFOLD_TSNE_EXACT}) {
    EXPECT_TRUE(
        all_finite(fitted(res, x.view(),
                          {.max_iter = short_run,
                           .method = method,
                           .metric = CUML_MANIFOLD_METRIC_SQEUCLIDEAN})))
        << method;
  }
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
