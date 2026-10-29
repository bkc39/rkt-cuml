#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <limits>
#include <numbers>
#include <rmm/cuda_stream_view.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cumlrkt/graph.hpp"
#include "cumlrkt/manifold.h"
#include "manifold_fixture.hpp"
#include "raftrkt/c_api.h"
#include "raftrkt/view.hpp"
#include "spectral_checks.hpp"
#include "spectral_residual.hpp"

namespace cumlrkt::test {
namespace {

constexpr int64_t rows = 8;
constexpr int64_t cols = 3;
constexpr int64_t pool = 64;
constexpr int64_t past_index = int64_t{1} << 31;
constexpr int64_t basis_rows = int64_t{1} << 27;
constexpr int64_t ring = 10;
constexpr int64_t ring_entries = 2 * ring;
constexpr double exact = 1e-12;
constexpr double single = 1e-6;
constexpr double cosine_spread = 2.0;
constexpr double large = 0x1p66;
constexpr double past_sum_weight = 0x1p125;
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

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

std::vector<float> with_cell(std::vector<float> host, std::size_t at,
                             double value) {
  host[at] = static_cast<float>(value);
  return host;
}

struct pools {
  explicit pools(const resources& res)
      : floats(res.r, RR_DTYPE_FLOAT32, {pool}),
        doubles(res.r, RR_DTYPE_FLOAT64, {pool}),
        ints(res.r, RR_DTYPE_INT32, {pool}) {}
  device_array floats;
  device_array doubles;
  device_array ints;
};

TEST(SpectralEmbeddingRefusals, DenseInTableOrder) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const pools p(res);
  const filled<float> x(res, blobs(rows, cols), {rows, cols});
  const rr_view doubles = shaped(p.doubles.view, {rows, cols});
  const rr_view rank1 = shaped(p.floats.view, {rows});
  const rr_view col_major = shaped(x.array.view, {rows, cols}, true);
  const rr_view one_col = shaped(p.floats.view, {rows, 1});
  const rr_view tall = shaped(p.floats.view, {past_index, cols});
  const rr_view wide = shaped(p.floats.view, {rows, past_index});
  const rr_view seven = shaped(x.array.view, {rows - 1, cols});
  const rr_view five = shaped(x.array.view, {rows - 3, cols});
  const rr_view basis = shaped(p.floats.view, {basis_rows, 2});
  const filled<float> nan_x(
      res, with_cell(with_cell(blobs(rows, cols), 10, nan), 13, large),
      {rows, cols});
  const filled<float> large_x(res, with_cell(blobs(rows, cols), 13, large),
                              {rows, cols});
  const filled<float> same(res, std::vector<float>(rows * cols, 1.5F),
                           {rows, cols});
  const auto fit = [&](const rr_view* v, const spectral_options& o = {}) {
    return [&res, v, o](record& out) { return spectral_fit(res, v, out, o); };
  };
  const std::vector<std::pair<std::function<int(record&)>, std::string>> cases{
      {fit(&doubles, {.n_components = 0}),
       "n-components: expected at least 1, got 0"},
      {fit(&doubles, {.n_components = past_index}),
       "n-components: expected at most 2147483647, got 2147483648"},
      {fit(&doubles, {.n_neighbors = 1}),
       "n-neighbors: expected at least 2, got 1"},
      {fit(&doubles), "X: expected float32, got float64"},
      {fit(&rank1), "X: expected rank 2, got rank 1"},
      {fit(&col_major), "X: expected row-major, got col-major 8x3"},
      {fit(&one_col, {.n_components = 9}),
       "X: expected at least 2 columns, got 1"},
      {fit(&tall), "X: 2147483648 rows do not fit the index type"},
      {fit(&wide), "X: 2147483648 columns do not fit the index type"},
      {fit(&seven, {.n_neighbors = rows}),
       "n-components: 2 components with drop-first need at least 8 rows, "
       "got 7"},
      {fit(&five, {.drop_first = 0}),
       "n-components: 2 components need at least 6 rows, got 5"},
      {fit(&basis),
       "n-components: 134217728 rows and 3 eigenvectors need a 134217728x20 "
       "eigensolver basis, more than 2147483647 elements"},
      {fit(nan_x.view(), {.n_neighbors = rows + 1}),
       "n-neighbors: expected at most the 8 rows, got 9"},
      {fit(nan_x.view()),
       "X: row 3, column 1: expected a finite value, got nan"},
      {fit(large_x.view()),
       std::format("X: row 4: expected a squared norm of at most "
                   "4.253529586511731e+37, got {}",
                   large * large)},
      {fit(same.view()),
       "X: every row equals row 0; a spectral embedding needs two distinct "
       "rows"},
  };
  for (const auto& [run, message] : cases) {
    expect_refused(res, run, message);
  }
}

struct host_graph {
  std::vector<int32_t> rows;
  std::vector<int32_t> cols;
  std::vector<float> vals;
};

host_graph ring_graph() {
  host_graph g;
  for (int32_t i = 0; i < ring; ++i) {
    const auto j = static_cast<int32_t>((i + 1) % ring);
    g.rows.insert(g.rows.end(), {i, j});
    g.cols.insert(g.cols.end(), {j, i});
    g.vals.insert(g.vals.end(), {1.0F, 1.0F});
  }
  return g;
}

struct device_graph {
  device_graph(const resources& res, const host_graph& g)
      : rows(res, g.rows, {static_cast<int64_t>(g.rows.size())}),
        cols(res, g.cols, {static_cast<int64_t>(g.cols.size())}),
        vals(res, g.vals, {static_cast<int64_t>(g.vals.size())}) {}
  [[nodiscard]] graph_input input(int64_t n) const {
    return {rows.view(), cols.view(), vals.view(), n};
  }
  filled<int32_t> rows;
  filled<int32_t> cols;
  filled<float> vals;
};

host_graph edited(const std::function<void(host_graph&)>& edit) {
  host_graph g = ring_graph();
  edit(g);
  return g;
}

TEST(SpectralEmbeddingRefusals, GraphInTableOrder) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const pools p(res);
  const device_graph ring_g(res, ring_graph());
  const device_graph loop(res, edited([](host_graph& g) {
                            g.cols[3] = g.rows[3];
                            g.vals[4] = -1;
                          }));
  const device_graph negative(res, edited([](host_graph& g) {
                                g.vals[1] = -1;
                                g.vals[2] = static_cast<float>(past_sum_weight);
                              }));
  const device_graph heavy(res, edited([](host_graph& g) {
                             for (float& v : g.vals) {
                               v = static_cast<float>(past_sum_weight);
                             }
                           }));
  const device_graph zeros(res, edited([](host_graph& g) {
                             for (float& v : g.vals) {
                               v = 0;
                             }
                           }));
  const rr_view float_rows = shaped(p.floats.view, {ring_entries});
  const rr_view short_cols = shaped(p.ints.view, {ring_entries - 1});
  const rr_view many = shaped(p.ints.view, {past_index - ring});
  const rr_view many_vals = shaped(p.floats.view, {past_index - ring});
  const auto fit = [&](const graph_input& g, const spectral_options& o = {}) {
    return [&res, g, o](record& out) {
      return spectral_fit_graph(res, g, out, o);
    };
  };
  const graph_input whole = ring_g.input(ring);
  const std::vector<std::pair<std::function<int(record&)>, std::string>> cases{
      {fit({&float_rows, &short_cols, whole.vals, 1}, {.n_components = 0}),
       "n-components: expected at least 1, got 0"},
      {fit({&float_rows, whole.cols, whole.vals, 1}),
       "rows: expected int32, got float32"},
      {fit({whole.rows, &short_cols, whole.vals, 1}),
       "cols: expected 20 entries, as rows has, got 19"},
      {fit(ring_g.input(7)),
       "n-components: 2 components with drop-first need at least 8 samples, "
       "got 7"},
      {fit(ring_g.input(past_index)),
       "n-samples: expected at most 2147483647, got 2147483648"},
      {fit({&many, &many, &many_vals, ring + 1}, {.n_components = 1}),
       "rows: 2147483638 entries and 11 samples do not fit the index type"},
      {fit(loop.input(ring)),
       "rows: entry 3: self-loop at sample 2; leave the diagonal out"},
      {fit(negative.input(ring)),
       "vals: entry 1: expected a finite weight of at least 0, got -1"},
      {fit(heavy.input(ring)),
       "vals: the weights sum to 8.507059173023462e+38, more than "
       "3.4028234663852886e+38"},
      {fit(zeros.input(ring)),
       "vals: expected at least one positive weight, got none"},
  };
  for (const auto& [run, message] : cases) {
    expect_refused(res, run, message);
  }
}

TEST(SpectralEmbeddingRefusals, AGraphTooWideForTheBasisIsRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const pools p(res);
  const rr_view none = shaped(p.ints.view, {0});
  const rr_view no_vals = shaped(p.floats.view, {0});
  record out(res);
  const int status = spectral_fit_graph(
      res, {&none, &none, &no_vals, basis_rows}, out, {.n_components = 2});
  EXPECT_EQ(outcome(status),
            "n-components: 134217728 samples and 3 eigenvectors need a "
            "134217728x20 eigensolver basis, more than 2147483647 elements");
  expect_untouched(out);
}

std::string solved_refusal(const std::vector<float>& host, int64_t n,
                           int64_t k) {
  const resources res;
  const filled<float> e(res, host, {n, k});
  try {
    manifold::require_solved(static_cast<const float*>(e.array.view.data), n, k,
                             rmm::cuda_stream_view{});
  } catch (const std::logic_error& error) {
    return error.what();
  }
  return "";
}

std::string zero_column_refusal(const std::vector<float>& columns, int64_t n,
                                int64_t k) {
  const resources res;
  const filled<float> e(res, columns, {k, n});
  try {
    manifold::require_nonzero_columns(
        static_cast<const float*>(e.array.view.data), n, k, "; advice",
        rmm::cuda_stream_view{});
  } catch (const std::logic_error& error) {
    return error.what();
  }
  return "";
}

TEST(SpectralEmbeddingRefusals, ANonFiniteEmbeddingIsRefusedRowMajor) {
  CUMLRKT_REQUIRE_GPU();
  const float inf = std::numeric_limits<float>::infinity();
  EXPECT_EQ(solved_refusal({1, 2, 3, 4, 5, static_cast<float>(nan)}, 3, 2),
            "embedding: row 2, column 1: the eigensolver returned nan");
  EXPECT_EQ(solved_refusal({1, 2, -inf, 4, inf, 6}, 3, 2),
            "embedding: row 1, column 0: the eigensolver returned -inf");
  EXPECT_EQ(solved_refusal({1, 2, 3, 4, 5, 6}, 3, 2), "");
}

TEST(SpectralEmbeddingRefusals, TheFirstZeroColumnIsRefused) {
  CUMLRKT_REQUIRE_GPU();
  EXPECT_EQ(zero_column_refusal({1, 0, 0, 0, 0, 0, 0, 0, 0}, 3, 3),
            "embedding: column 1 is zero: the eigensolver found no "
            "eigenvector of this graph; advice");
  EXPECT_EQ(zero_column_refusal({0, 0, 0, 0, 0, 0}, 3, 2),
            "embedding: column 0 is zero: the eigensolver found no "
            "eigenvector of this graph; advice");
  EXPECT_EQ(zero_column_refusal({0, 0, -1e-30F, 0, 1, 0}, 3, 2), "");
}

std::vector<manifold::column_fit> ring_fits(const std::vector<float>& columns,
                                            int64_t k, bool normalised) {
  const resources res;
  const device_graph g(res, ring_graph());
  const filled<float> e(res, columns, {k, ring});
  const coo c{static_cast<int32_t*>(g.rows.array.view.data),
              static_cast<int32_t*>(g.cols.array.view.data),
              static_cast<float*>(g.vals.array.view.data), ring_entries};
  return manifold::column_fits(raftrkt::handle_of(res.handle), c, ring,
                               static_cast<const float*>(e.array.view.data), k,
                               normalised);
}

std::vector<float> ones_cosine_and_first() {
  std::vector<float> columns(3 * ring, 0.0F);
  for (int64_t i = 0; i < ring; ++i) {
    columns[i] = 1.0F;
    columns[ring + i] = static_cast<float>(
        std::cos(2 * std::numbers::pi * static_cast<double>(i) / ring));
  }
  columns[2 * ring] = 1.0F;
  return columns;
}

void expect_fit(const manifold::column_fit& found,
                const manifold::column_fit& expected,
                const manifold::column_fit& tolerance) {
  EXPECT_NEAR(found.residual, expected.residual, tolerance.residual);
  EXPECT_NEAR(found.spread, expected.spread, tolerance.spread);
}

TEST(SpectralEmbeddingRefusals, ColumnFitsMeasureTheEigenResidual) {
  CUMLRKT_REQUIRE_GPU();
  const std::vector<float> columns = ones_cosine_and_first();
  for (const bool normalised : {false, true}) {
    SCOPED_TRACE(normalised);
    const std::vector<manifold::column_fit> fits =
        ring_fits(columns, 3, normalised);
    ASSERT_EQ(fits.size(), 3U);
    expect_fit(fits[0], {.residual = 0.0, .spread = 0.0},
               {.residual = exact, .spread = 0.0});
    expect_fit(fits[1], {.residual = 0.0, .spread = cosine_spread},
               {.residual = single, .spread = single});
    expect_fit(fits[2], {.residual = 1.0 / std::numbers::sqrt2, .spread = 1.0},
               {.residual = exact, .spread = 0.0});
  }
}

}  // namespace
}  // namespace cumlrkt::test
