#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
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
constexpr int64_t quarter_index = int64_t{1} << 29;
constexpr int64_t most_labels = int64_t{1} << 24;
constexpr int64_t few_rows = 149;
constexpr int64_t wide_components = 4;
constexpr double past_float = 1e39;
constexpr double divergent_rate = 1e35;
constexpr double most_learning_rate = 3e38;
constexpr double large = 0x1p66;
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
      : ints(res.r, RR_DTYPE_INT32, {pool}),
        floats(res.r, RR_DTYPE_FLOAT32, {pool}),
        doubles(res.r, RR_DTYPE_FLOAT64, {pool}) {}
  device_array ints;
  device_array floats;
  device_array doubles;
};

TEST(UmapFitRefusals, ParametersInTableOrder) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, blobs(rows, cols), {rows, cols});
  const auto fit = [&](const umap_options& o) {
    return [&res, &x, o](record& out) {
      return umap_fit(res, x.view(), nullptr, out, o);
    };
  };
  const std::vector<std::pair<umap_options, std::string>> cases{
      {{.metric = -1},
       "metric: expected euclidean, sqeuclidean, manhattan, chebyshev or "
       "cosine"},
      {{.init = -1}, "init: expected spectral or random"},
      {{.build_algo = -1}, "build-algo: expected brute-force or nn-descent"},
      {{.build_algo = CUML_MANIFOLD_BUILD_NN_DESCENT,
        .metric = CUML_MANIFOLD_METRIC_MANHATTAN},
       "metric: nn-descent supports euclidean, sqeuclidean and cosine, got "
       "manhattan"},
      {{.build_algo = CUML_MANIFOLD_BUILD_NN_DESCENT,
        .metric = CUML_MANIFOLD_METRIC_CHEBYSHEV},
       "metric: nn-descent supports euclidean, sqeuclidean and cosine, got "
       "chebyshev"},
      {{.n_neighbors = 1}, "n-neighbors: expected at least 2, got 1"},
      {{.n_neighbors = past_index},
       "n-neighbors: expected at most 2147483647, got 2147483648"},
      {{.n_components = 0}, "n-components: expected at least 1, got 0"},
      {{.n_components = past_index},
       "n-components: expected at most 2147483647, got 2147483648"},
      {{.epochs_given = 1, .n_epochs = 0},
       "n-epochs: expected at least 1, got 0"},
      {{.epochs_given = 1, .n_epochs = past_index},
       "n-epochs: expected at most 2147483647, got 2147483648"},
      {{.learning_rate = 0},
       "learning-rate: expected a finite value above 0, got 0"},
      {{.learning_rate = nan},
       "learning-rate: expected a finite value above 0, got nan"},
      {{.learning_rate = inf},
       "learning-rate: expected a finite value above 0, got inf"},
      {{.learning_rate = past_float},
       "learning-rate: expected at most 3.4028234663852886e+38, got 1e+39"},
      {{.learning_rate = denormal},
       "learning-rate: expected at least 1.1754943508222875e-38, got 1e-40"},
      {{.learning_rate = underflow},
       "learning-rate: expected at least 1.1754943508222875e-38, got 1e-50"},
      {{.negative_sample_rate = -1},
       "negative-sample-rate: expected at least 0, got -1"},
      {{.negative_sample_rate = past_index},
       "negative-sample-rate: expected at most 2147483647, got 2147483648"},
      {{.a = 0}, "a: expected a finite float32 value above 0, got 0"},
      {{.a = nan}, "a: expected a finite float32 value above 0, got nan"},
      {{.a = denormal},
       "a: expected a finite float32 value above 0, got 1e-40"},
      {{.b = past_float},
       "b: expected a finite float32 value above 0, got 1e+39"},
  };
  for (const auto& [o, message] : cases) {
    expect_refused(res, fit(o), message);
  }
}

TEST(UmapFitRefusals, InputsInTableOrder) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const pools p(res);
  const filled<float> x(res, blobs(rows, cols), {rows, cols});
  const filled<int64_t> labels(res, std::vector<int64_t>(rows, 0), {rows});
  const rr_view doubles = shaped(p.doubles.view, {rows, cols});
  const rr_view rank1 = shaped(p.floats.view, {rows});
  const rr_view col_major = shaped(x.array.view, {rows, cols}, true);
  const rr_view one_row = shaped(p.floats.view, {1, cols});
  const rr_view no_cols = shaped(p.floats.view, {rows, 0});
  const rr_view tall = shaped(p.floats.view, {past_index, cols});
  const rr_view wide = shaped(p.floats.view, {2, past_index});
  const rr_view three = shaped(p.floats.view, {3, cols});
  const rr_view huge = shaped(p.floats.view, {half_index, 2});
  const rr_view short_x = shaped(p.floats.view, {few_rows, cols});
  const rr_view float_y = shaped(p.floats.view, {rows});
  const rr_view rank2_y = shaped(p.ints.view, {rows, 1});
  rr_view strided_y = shaped(p.ints.view, {rows});
  strided_y.strides[0] = 2;
  const rr_view short_y = shaped(p.ints.view, {rows - 1});
  const filled<float> nan_x(
      res, with(with(blobs(rows, cols), 10, nan), (4 * cols) + 1, large),
      {rows, cols});
  const filled<float> large_x(
      res, with(zero_row(blobs(rows, cols), 4, cols), (4 * cols) + 1, large),
      {rows, cols});
  const filled<float> zero_x(res, zero_row(blobs(rows, 2), 3, 2), {rows, 2});
  const umap_options spectral{.init = CUML_MANIFOLD_INIT_SPECTRAL};
  const umap_options nn_descent{.build_algo = CUML_MANIFOLD_BUILD_NN_DESCENT};
  const umap_options cosine{.metric = CUML_MANIFOLD_METRIC_COSINE};
  const auto fit = [&](const rr_view* v, const rr_view* y,
                       const umap_options& o = {}) {
    return [&res, v, y, o](record& out) { return umap_fit(res, v, y, out, o); };
  };
  const std::vector<std::pair<std::function<int(record&)>, std::string>> cases{
      {fit(&doubles, nullptr), "X: expected float32, got float64"},
      {fit(&rank1, nullptr), "X: expected rank 2, got rank 1"},
      {fit(&col_major, nullptr), "X: expected row-major, got col-major 8x3"},
      {fit(&one_row, nullptr), "X: expected at least 2 rows, got 1"},
      {fit(&no_cols, nullptr), "X: expected at least 1 column, got 0"},
      {fit(&tall, nullptr), "X: 2147483648 rows do not fit the index type"},
      {fit(&wide, nullptr), "X: 2147483648 columns do not fit the index type"},
      {fit(&three, nullptr, {.n_components = 3, .init = 1}),
       "n-components: spectral init needs fewer components than the 3 rows, "
       "got 3"},
      {fit(&huge, nullptr, spectral),
       "n-components: spectral init of 1073741824 rows with 2 components "
       "needs 3221225472 entries, more than 2147483647"},
      {fit(&short_x, nullptr, nn_descent),
       "X: nn-descent needs at least 150 rows, got 149"},
      {fit(x.view(), &float_y), "y: expected int32 or int64, got float32"},
      {fit(x.view(), &rank2_y), "y: expected rank 1, got rank 2"},
      {fit(x.view(), &strided_y),
       "y: expected contiguous elements, got stride 2"},
      {fit(x.view(), &short_y), "y: expected 8 elements, got 7"},
      {fit(nan_x.view(), labels.view()),
       "X: row 3, column 1: expected a finite value, got nan"},
      {fit(large_x.view(), nullptr),
       std::format("X: row 4: expected a squared norm of at most "
                   "4.253529586511731e+37, got {}",
                   large * large)},
      {fit(zero_x.view(), nullptr, cosine),
       "X: row 3: expected a squared norm of at least 2.350988701644575e-38 "
       "under cosine, got 0"},
  };
  for (const auto& [run, message] : cases) {
    expect_refused(res, run, message);
  }
}

TEST(UmapFitRefusals, MoreThanFloatExactLabels) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const int64_t n = most_labels + 1;
  const filled<float> x(res, std::vector<float>(static_cast<std::size_t>(n)),
                        {n, 1});
  std::vector<int32_t> host(static_cast<std::size_t>(n));
  std::iota(host.begin(), host.end(), 0);
  const filled<int32_t> y(res, host, {n});
  expect_refused(
      res, [&](record& out) { return umap_fit(res, x.view(), y.view(), out); },
      "y: expected at most 16777216 distinct labels besides -1, got 16777217");
}

TEST(UmapFitRefusals, ADivergedEmbeddingIsFreed) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, blobs(rows, cols), {rows, cols});
  record out(res);
  const int status =
      umap_fit(res, x.view(), nullptr, out, {.learning_rate = divergent_rate});
  EXPECT_EQ(status, RR_ERROR);
  EXPECT_TRUE(outcome(status).starts_with("embedding: row "));
  EXPECT_TRUE(outcome(status).ends_with("; lower learning-rate"));
  expect_untouched(out);
}

std::vector<float> counting(int64_t n) {
  std::vector<float> host(static_cast<std::size_t>(n));
  std::iota(host.begin(), host.end(), 0.0F);
  return host;
}

struct trained {
  explicit trained(const resources& res, int64_t components = 2)
      : x(res, blobs(rows, cols), {rows, cols}),
        embedding(res, counting(rows * components), {rows, components}) {}
  filled<float> x;
  filled<float> embedding;
  [[nodiscard]] umap_trained views() const {
    return {x.view(), embedding.view()};
  }
};

TEST(UmapTransformRefusals, EachConditionInTableOrder) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const pools p(res);
  const trained t(res);
  const trained wide_t(res, wide_components);
  const filled<float> x(res, blobs(2, cols), {2, cols});
  const rr_view ints = shaped(p.ints.view, {2, cols});
  const rr_view rank1 = shaped(p.floats.view, {cols});
  const rr_view no_rows = shaped(p.floats.view, {0, cols});
  const rr_view tall = shaped(p.floats.view, {past_index, cols});
  const rr_view train_doubles = shaped(p.doubles.view, {rows, cols});
  const rr_view train_one = shaped(p.floats.view, {1, cols});
  const rr_view emb_doubles = shaped(p.doubles.view, {rows, 2});
  const rr_view emb_short = shaped(p.floats.view, {rows - 1, 2});
  const rr_view emb_wide = shaped(p.floats.view, {rows, 3});
  const rr_view narrow = shaped(p.floats.view, {2, cols - 1});
  const rr_view big = shaped(p.floats.view, {half_index, cols});
  const rr_view bigger = shaped(p.floats.view, {quarter_index, cols});
  const rr_view big_train = shaped(p.floats.view, {half_index, cols});
  const rr_view big_embedding = shaped(p.floats.view, {half_index, 2});
  const filled<float> nan_train(res, with(blobs(rows, cols), 4, nan),
                                {rows, cols});
  const filled<float> zero_train(res, zero_row(blobs(rows, cols), 3, cols),
                                 {rows, cols});
  const filled<float> zero_x(res, zero_row(blobs(2, cols), 0, cols),
                             {2, cols});
  const filled<float> nan_x(res, with(blobs(2, cols), 4, nan), {2, cols});
  const umap_trained good = t.views();
  const auto apply = [&](const rr_view* v, const umap_trained& on,
                         const umap_options& o = {}) {
    return [&res, v, on, o](record& out) {
      return umap_transform(res, v, on, out, o);
    };
  };
  const std::vector<std::pair<std::function<int(record&)>, std::string>> cases{
      {apply(&ints, good), "X: expected float32, got int32"},
      {apply(&rank1, good), "X: expected rank 2, got rank 1"},
      {apply(&no_rows, good), "X: expected at least 1 row, got 0"},
      {apply(&tall, good), "X: 2147483648 rows do not fit the index type"},
      {apply(x.view(), {&train_doubles, t.embedding.view()}),
       "result X: expected float32, got float64"},
      {apply(x.view(), {&train_one, t.embedding.view()}),
       "result X: expected at least 2 rows, got 1"},
      {apply(x.view(), {t.x.view(), &emb_doubles}),
       "embedding: expected float32, got float64"},
      {apply(x.view(), {t.x.view(), &emb_short}),
       "embedding: expected 8 rows, got 7"},
      {apply(x.view(), {t.x.view(), &emb_wide}),
       "embedding: expected 2 columns, got 3"},
      {apply(&narrow, good), "X: expected 3 columns, got 2"},
      {apply(x.view(), good, {.metric = -1}),
       "metric: expected euclidean, sqeuclidean, manhattan, chebyshev or "
       "cosine"},
      {apply(x.view(), good, {.n_neighbors = 1}),
       "n-neighbors: expected at least 2, got 1"},
      {apply(x.view(), good, {.learning_rate = underflow}),
       "learning-rate: expected at least 1.1754943508222875e-38, got 1e-50"},
      {apply(x.view(), good, {.a = 0}),
       "a: expected a finite float32 value above 0, got 0"},
      {apply(x.view(), good, {.epochs_given = 1, .n_epochs = 2}),
       "n-epochs: transform runs n-epochs/3 epochs and needs at least 3, got "
       "2"},
      {apply(&big, good, {.n_neighbors = 2}),
       "X: 1073741824 rows with 2 neighbours each need 2147483648 entries, "
       "more than 2147483647"},
      {apply(&bigger, wide_t.views(),
             {.n_neighbors = 2, .n_components = wide_components}),
       "X: 536870912 rows with 4 components each need 2147483648 entries, "
       "more than 2147483647"},
      {apply(x.view(), {&big_train, &big_embedding}, {.n_neighbors = 2}),
       "result X: 1073741824 rows with 2 components each need 2147483648 "
       "entries, more than 2147483647"},
      {apply(x.view(), {nan_train.view(), t.embedding.view()}),
       "result X: row 1, column 1: expected a finite value, got nan"},
      {apply(x.view(), {zero_train.view(), t.embedding.view()},
             {.metric = CUML_MANIFOLD_METRIC_COSINE}),
       "result X: row 3: expected a squared norm of at least "
       "3.5264830524668625e-38 under cosine, got 0"},
      {apply(nan_x.view(), {nan_train.view(), t.embedding.view()}),
       "result X: row 1, column 1: expected a finite value, got nan"},
      {apply(nan_x.view(), good),
       "X: row 1, column 1: expected a finite value, got nan"},
      {apply(zero_x.view(), good, {.metric = CUML_MANIFOLD_METRIC_COSINE}),
       "X: row 0: expected a squared norm of at least 3.5264830524668625e-38 "
       "under cosine, got 0"},
  };
  for (const auto& [run, message] : cases) {
    expect_refused(res, run, message);
  }
}

TEST(UmapTransformRefusals, ADivergedOutputIsFreed) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, blobs(rows, cols), {rows, cols});
  record trained(res);
  ASSERT_EQ(outcome(umap_fit(res, x.view(), nullptr, trained)), "accepted");
  record out(res);
  const int status =
      umap_transform(res, x.view(), {x.view(), &trained.r.regions[0]}, out,
                     {.learning_rate = most_learning_rate});
  EXPECT_EQ(status, RR_ERROR);
  EXPECT_TRUE(outcome(status).starts_with("output: row "));
  EXPECT_TRUE(outcome(status).ends_with("; lower learning-rate"));
  expect_untouched(out);
}

}  // namespace
}  // namespace cumlrkt::test
