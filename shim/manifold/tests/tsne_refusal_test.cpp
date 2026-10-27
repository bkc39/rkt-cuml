#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <rmm/cuda_stream_view.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cumlrkt/manifold.h"
#include "cumlrkt/support.hpp"
#include "manifold_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {
namespace {

constexpr int64_t rows = 8;
constexpr int64_t cols = 3;
constexpr int64_t pool = 64;
constexpr std::size_t far_row = 4;
constexpr int64_t past_index = int64_t{1} << 31;
constexpr int64_t half_index = int64_t{1} << 30;
constexpr int64_t past_exact = 2097121;
constexpr double past_float = 1e39;
constexpr double divergent_rate = 1e30;
constexpr double large = 0x1p66;
constexpr double past_sqeuclidean_y = 0x1p30;
constexpr double past_sqeuclidean_z = 0x1p30 + 0x1p7;
constexpr double diverging_x = 1807712000;
constexpr double diverging_y = 1159251072;
constexpr double past_manhattan = 3.5e18;
constexpr double past_pca = 5 * 0x1p60;
constexpr int64_t pca_rows = 16;
constexpr double rounds_to_one = 1.0 + 1e-9;
constexpr int64_t copies = 16;
constexpr const char* advice =
    "; lower learning-rate, centre or rescale X, or raise n-neighbors";
constexpr double nan = std::numeric_limits<double>::quiet_NaN();
constexpr double inf = std::numeric_limits<double>::infinity();
constexpr double denormal = 1e-40;
constexpr double underflow = 1e-50;

void expect_refused(const resources& res,
                    const std::function<int(record&)>& run,
                    const std::string& message) {
  record out(res);
  const int status = run(out);
  EXPECT_EQ(status, RR_ERROR) << message;
  EXPECT_EQ(outcome(status), message);
  EXPECT_EQ(cuml_manifold_last_error_kind(), RR_ERROR_LOGIC) << message;
  expect_untouched(out);
}

std::vector<float> with(
    std::vector<float> host,
    std::initializer_list<std::pair<std::size_t, double>> cells) {
  for (const auto& [at, value] : cells) {
    host[at] = static_cast<float>(value);
  }
  return host;
}

std::vector<float> antipodal(const std::array<double, cols>& point) {
  std::vector<float> host = blobs(rows, cols);
  const std::size_t width = point.size();
  for (std::size_t j = 0; j < width; ++j) {
    host[(far_row * width) + j] = static_cast<float>(point[j]);
    host[((far_row + 1) * width) + j] = static_cast<float>(-point[j]);
  }
  return host;
}

std::vector<float> zero_row(std::vector<float> host, int64_t row, int64_t d) {
  for (int64_t j = 0; j < d; ++j) {
    host[static_cast<std::size_t>((row * d) + j)] = 0.0F;
  }
  return host;
}

struct pools {
  explicit pools(const resources& res)
      : floats(res.r, RR_DTYPE_FLOAT32, {pool}),
        doubles(res.r, RR_DTYPE_FLOAT64, {pool}) {}
  device_array floats;
  device_array doubles;
};

TEST(TsneFitRefusals, ParametersInTableOrder) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const pools p(res);
  const filled<float> x(res, blobs(rows, cols), {rows, cols});
  const rr_view doubles = shaped(p.doubles.view, {rows, cols});
  const auto fit = [&](const tsne_options& o, const rr_view* v = nullptr) {
    return [&res, &x, v, o](record& out) {
      return tsne_fit(res, v == nullptr ? x.view() : v, out, o);
    };
  };
  const std::vector<std::pair<std::function<int(record&)>, std::string>> cases{
      {fit({.method = -1}, &doubles), "method: expected fft or exact"},
      {fit({.method = 2}), "method: expected fft or exact"},
      {fit({.metric = -1}),
       "metric: expected euclidean, sqeuclidean, manhattan, chebyshev or "
       "cosine"},
      {fit({.init = -1}), "init: expected random or pca"},
      {fit({.n_components = 3}), "n-components: expected 2, got 3"},
      {fit({.n_components = 1}), "n-components: expected 2, got 1"},
      {fit({.n_neighbors = 1}, &doubles),
       "n-neighbors: expected at least 2, got 1"},
      {fit({.n_neighbors = 1024}),
       "n-neighbors: expected at most 1023, got 1024"},
      {fit({.perplexity = 1}),
       "perplexity: expected a finite value above 1 in float32, got 1"},
      {fit({.perplexity = rounds_to_one}),
       "perplexity: expected a finite value above 1 in float32, got "
       "1.000000001"},
      {fit({.perplexity = nan}),
       "perplexity: expected a finite value above 1 in float32, got nan"},
      {fit({.perplexity = inf}),
       "perplexity: expected a finite value above 1 in float32, got inf"},
      {fit({.perplexity = past_float}),
       "perplexity: expected at most 3.4028234663852886e+38, got 1e+39"},
      {fit({.early_exaggeration = 0.5}),
       "early-exaggeration: expected a finite value of at least 1, got 0.5"},
      {fit({.early_exaggeration = nan}),
       "early-exaggeration: expected a finite value of at least 1, got nan"},
      {fit({.early_exaggeration = past_float}),
       "early-exaggeration: expected at most 3.4028234663852886e+38, got "
       "1e+39"},
      {fit({.learning_rate = 0}),
       "learning-rate: expected a finite value above 0, got 0"},
      {fit({.learning_rate = -inf}),
       "learning-rate: expected a finite value above 0, got -inf"},
      {fit({.learning_rate = past_float}),
       "learning-rate: expected at most 1.7014117331926443e+38, got 1e+39"},
      {fit({.learning_rate = 2e38}),
       "learning-rate: expected at most 1.7014117331926443e+38, got 2e+38"},
      {fit({.learning_rate = denormal}),
       "learning-rate: expected at least 1.1754943508222875e-38, got 1e-40"},
      {fit({.learning_rate = underflow}),
       "learning-rate: expected at least 1.1754943508222875e-38, got 1e-50"},
      {fit({.max_iter = 0}), "max-iter: expected at least 1, got 0"},
      {fit({.max_iter = past_index}),
       "max-iter: expected at most 2147483647, got 2147483648"},
      {fit({.random_state = -1}), "random-state: expected at least 0, got -1"},
  };
  for (const auto& [run, message] : cases) {
    expect_refused(res, run, message);
  }
}

TEST(TsneFitRefusals, InputsInTableOrder) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const pools p(res);
  const filled<float> x(res, blobs(rows, cols), {rows, cols});
  const rr_view doubles = shaped(p.doubles.view, {rows, cols});
  const rr_view rank1 = shaped(p.floats.view, {rows});
  const rr_view col_major = shaped(x.array.view, {rows, cols}, true);
  const rr_view one_row = shaped(p.floats.view, {1, cols});
  const rr_view one_col = shaped(p.floats.view, {rows, 1});
  const rr_view tall = shaped(p.floats.view, {past_index, cols});
  const rr_view wide = shaped(p.floats.view, {2, past_index});
  const rr_view huge = shaped(p.floats.view, {half_index, 2});
  const rr_view exact_rows = shaped(p.floats.view, {past_exact, 2});
  const filled<float> nan_x(
      res, with(blobs(rows, cols), {{10, nan}, {(4 * cols) + 1, large}}),
      {rows, cols});
  const filled<float> large_x(
      res,
      with(zero_row(blobs(rows, cols), 3, cols), {{(4 * cols) + 1, large}}),
      {rows, cols});
  const filled<float> zero_x(res, zero_row(blobs(rows, 2), 3, 2), {rows, 2});
  const filled<float> far_x(
      res, antipodal({0, past_sqeuclidean_y, past_sqeuclidean_z}),
      {rows, cols});
  const filled<float> diverging_x_rows(
      res, antipodal({diverging_x, diverging_y, 0}), {rows, cols});
  const filled<float> long_x(
      res, antipodal({past_manhattan, past_manhattan, past_manhattan}),
      {rows, cols});
  const filled<float> spread_x(res, spread_rows(past_pca), {pca_rows, cols});
  const filled<float> zeros(res, std::vector<float>(rows * 2, 0.0F), {rows, 2});
  const filled<float> same(res, std::vector<float>(rows * cols, 1.5F),
                           {rows, cols});
  const tsne_options cosine{.metric = CUML_MANIFOLD_METRIC_COSINE};
  const tsne_options exact{.method = CUML_MANIFOLD_TSNE_EXACT};
  const tsne_options sqeuclidean{.metric = CUML_MANIFOLD_METRIC_SQEUCLIDEAN};
  const tsne_options manhattan{.metric = CUML_MANIFOLD_METRIC_MANHATTAN};
  const tsne_options pca{.init = CUML_MANIFOLD_TSNE_INIT_PCA};
  const tsne_options pca_exact{.method = CUML_MANIFOLD_TSNE_EXACT,
                               .init = CUML_MANIFOLD_TSNE_INIT_PCA};
  const auto fit = [&](const rr_view* v, const tsne_options& o = {}) {
    return [&res, v, o](record& out) { return tsne_fit(res, v, out, o); };
  };
  const std::vector<std::pair<std::function<int(record&)>, std::string>> cases{
      {fit(&doubles), "X: expected float32, got float64"},
      {fit(&rank1), "X: expected rank 2, got rank 1"},
      {fit(&col_major), "X: expected row-major, got col-major 8x3"},
      {fit(&one_row, {.perplexity = 30}), "X: expected at least 2 rows, got 1"},
      {fit(&one_col), "X: expected at least 2 columns, got 1"},
      {fit(&tall), "X: 2147483648 rows do not fit the index type"},
      {fit(&wide), "X: 2147483648 columns do not fit the index type"},
      {fit(&huge),
       "X: random init of 1073741824 rows needs 2147483648 entries, more "
       "than 2147483647"},
      {fit(x.view(), {.perplexity = 9}),
       "perplexity: expected at most the 8 neighbours, got 9"},
      {fit(x.view(), {.perplexity = 5, .n_neighbors = 4}),
       "perplexity: expected at most the 4 neighbours, got 5"},
      {fit(&exact_rows, exact),
       "method: exact supports at most 2097120 rows, got 2097121"},
      {fit(nan_x.view()),
       "X: row 3, column 1: expected a finite value, got nan"},
      {fit(large_x.view(), cosine),
       std::format("X: row 4: expected a squared norm of at most "
                   "4.253529586511731e+37, got {}",
                   large * large)},
      {fit(large_x.view(), sqeuclidean),
       std::format("X: row 4: expected a squared norm of at most "
                   "4.253529586511731e+37, got {}",
                   large * large)},
      {fit(far_x.view(), sqeuclidean),
       "X: row 4: expected a squared norm of at most 2305843009213693952 "
       "under sqeuclidean, got 2305843284091617280"},
      {fit(diverging_x_rows.view(), sqeuclidean),
       "X: row 4: expected a squared norm of at most 2305843009213693952 "
       "under sqeuclidean, got 4611685722877149184"},
      {fit(long_x.view(), manhattan),
       std::format("X: row 4: expected an L1 norm of at most {} under "
                   "manhattan, got {}",
                   std::numbers::sqrt2 * 0x1p62,
                   3.0 * static_cast<float>(past_manhattan))},
      {fit(spread_x.view(), pca),
       "X: expected a total squared distance from the mean of at most "
       "8.507059173023462e+37 under pca init, got 5.316911983433481e+38"},
      {fit(spread_x.view(), pca_exact),
       "X: expected a total squared distance from the mean of at most "
       "8.507059173023462e+37 under pca init, got 5.316911983433481e+38"},
      {fit(zero_x.view(), cosine),
       "X: row 3: expected a squared norm of at least 2.350988701644575e-38 "
       "under cosine, got 0"},
      {fit(zeros.view(), cosine),
       "X: row 0: expected a squared norm of at least 2.350988701644575e-38 "
       "under cosine, got 0"},
      {fit(same.view()),
       "X: every row equals row 0; t-SNE needs two distinct rows"},
      {fit(zeros.view()),
       "X: every row equals row 0; t-SNE needs two distinct rows"},
  };
  for (const auto& [run, message] : cases) {
    expect_refused(res, run, message);
  }
}

TEST(TsneFitRefusals, ADivergedEmbeddingIsFreed) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, blobs(rows, cols), {rows, cols});
  record out(res);
  const int status =
      tsne_fit(res, x.view(), out, {.learning_rate = divergent_rate});
  EXPECT_EQ(status, RR_ERROR);
  EXPECT_TRUE(outcome(status).starts_with("embedding: row "));
  EXPECT_TRUE(outcome(status).ends_with(advice));
  expect_untouched(out);
}

TEST(TsneFitRefusals, DuplicatesThatFillEveryNeighbourhoodDiverge) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  std::vector<float> host(copies * 4, 0.0F);
  std::fill(host.begin() + (copies * 2), host.end(), 1.0F);
  const filled<float> x(res, host, {copies * 2, 2});
  record out(res);
  EXPECT_EQ(outcome(tsne_fit(res, x.view(), out)),
            std::string("embedding: row 0, column 0: the optimisation "
                        "diverged to nan") +
                advice);
  expect_untouched(out);
  record fitted(res);
  EXPECT_EQ(
      outcome(tsne_fit(res, x.view(), fitted, {.n_neighbors = copies * 2})),
      "accepted");
}

std::string unlike_first_refusal(const std::vector<float>& host) {
  const resources res;
  const filled<float> y(res, host, {rows, 2});
  try {
    require_unlike_first("embedding",
                         static_cast<const float*>(y.array.view.data), rows, 2,
                         "constant", rmm::cuda_stream_view{});
  } catch (const std::logic_error& e) {
    return e.what();
  }
  return "";
}

TEST(TsneFitRefusals, AConstantEmbeddingIsRefusedAndASpreadOnePasses) {
  CUMLRKT_REQUIRE_GPU();
  EXPECT_EQ(unlike_first_refusal(std::vector<float>(rows * 2, 1.5F)),
            "embedding: every row equals row 0; constant");
  EXPECT_EQ(unlike_first_refusal(blobs(rows, 2)), "");
}

}  // namespace
}  // namespace cumlrkt::test
