#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "cumlrkt/manifold.h"
#include "manifold_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {
namespace {

constexpr int64_t rows = 8;
constexpr int64_t cols = 3;
constexpr int64_t pool = 64;
constexpr int64_t past_index = int64_t{1} << 31;
constexpr int64_t half_index = int64_t{1} << 30;
constexpr int64_t past_exact = 2097121;
constexpr double past_float = 1e39;
constexpr double divergent_rate = 1e30;
constexpr double large = 0x1p66;
constexpr double unsquarable = 0x1p31;
constexpr double least_above_one = 1.0 + 1e-9;
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

std::vector<float> with(std::vector<float> host, std::size_t at, double value) {
  host[at] = static_cast<float>(value);
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
       "perplexity: expected a finite value above 1, got 1"},
      {fit({.perplexity = least_above_one}),
       "perplexity: expected a finite value above 1, got 1.000000001"},
      {fit({.perplexity = nan}),
       "perplexity: expected a finite value above 1, got nan"},
      {fit({.perplexity = inf}),
       "perplexity: expected a finite value above 1, got inf"},
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
      res, with(with(blobs(rows, cols), 10, nan), (4 * cols) + 1, large),
      {rows, cols});
  const filled<float> large_x(
      res, with(zero_row(blobs(rows, cols), 3, cols), (4 * cols) + 1, large),
      {rows, cols});
  const filled<float> zero_x(res, zero_row(blobs(rows, 2), 3, 2), {rows, 2});
  const filled<float> far_x(
      res,
      with(zero_row(blobs(rows, cols), 4, cols), (4 * cols) + 1, unsquarable),
      {rows, cols});
  const filled<float> zeros(res, std::vector<float>(rows * 2, 0.0F), {rows, 2});
  const filled<float> same(res, std::vector<float>(rows * cols, 1.5F),
                           {rows, cols});
  const tsne_options cosine{.metric = CUML_MANIFOLD_METRIC_COSINE};
  const tsne_options exact{.method = CUML_MANIFOLD_TSNE_EXACT};
  const tsne_options sqeuclidean{.metric = CUML_MANIFOLD_METRIC_SQEUCLIDEAN};
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
       "X: row 4: expected a squared norm of at most 4611685743549480960 "
       "under sqeuclidean, got 4611686018427387904"},
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
  EXPECT_TRUE(outcome(status).ends_with("; lower learning-rate"));
  expect_untouched(out);
}

}  // namespace
}  // namespace cumlrkt::test
