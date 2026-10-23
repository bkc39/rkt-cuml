#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "cluster_fixture.hpp"
#include "cumlrkt/cluster.h"
#include "kmeans_fixture.hpp"
#include "raftrkt/c_api.h"
#include "refusal_fixture.hpp"
#include "spectral_fixture.hpp"

namespace cumlrkt::test {
namespace {

constexpr int64_t entries = 12;
constexpr int64_t graph_rows = 102261127;
constexpr int64_t basis_rows = 107374183;
constexpr int64_t trial_clusters = 65659970;
constexpr int64_t forged_entries = 2147483000;
constexpr int64_t forged_samples = 1000;
constexpr int64_t past_index = 2147483648;
constexpr int64_t past_components = 4;
constexpr double past_float = 1e39;

template <typename Call>
Call many_clusters(Call c, int64_t clusters = trial_clusters) {
  c.n_clusters = clusters;
  c.n_components = 1;
  c.auto_components = 0;
  return c;
}

void expect_refused(const allocation_meter& meter, const record& out,
                    const std::string& status, const std::string& message) {
  EXPECT_EQ(status, message);
  EXPECT_EQ(cuml_cluster_last_error_kind(), RR_ERROR_LOGIC) << status;
  EXPECT_EQ(meter.peak(), 0) << status;
  expect_untouched(out);
}

TEST(SpectralRefusals, DenseNameTheArgument) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const refusal_case<float> r(res, rows, cols);
  record out(res);
  const allocation_meter meter;
  const rr_view* const x = &r.x;
  rr_deposit* const o = &out.r;
  const rr_view x_double = shaped(r.other, {rows, cols});
  const rr_view x_int = shaped(r.ints, {rows, cols});
  const rr_view x_rank1 = shaped(r.data, {rows});
  const rr_view x_col = shaped(r.data, {rows, cols}, true);
  const rr_view x_three = shaped(r.data, {3, cols});
  const rr_view x_three_double = shaped(r.other, {3, cols});
  const rr_view x_no_rows = shaped(r.data, {0, cols});
  const rr_view x_no_cols = shaped(r.data, {rows, 0});
  const rr_view x_three_no_cols = shaped(r.data, {3, 0});
  const rr_view x_wide = shaped(r.data, {wide_rows, wide_cols});
  const rr_view x_graph = shaped(r.data, {graph_rows, 1});
  const rr_view x_basis = shaped(r.data, {basis_rows, 1});
  const rr_view x_basis_most = shaped(r.data, {basis_rows - 1, 1});
  const rr_view x_trials_most = shaped(r.data, {graph_rows - 1, 1});
  const rr_view x_elsewhere = on_device(r.x, 1);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const std::string components =
      "n-components: expected at most 3 for 10 rows, got 4";
  const std::string tol_most =
      "eigen-tol: expected at most 3.4028234663852886e+38, got ";
  const std::vector<refusal<spectral_call>> cases{
      {{.x = &x_double, .out = o}, "X: expected float32, got float64"},
      {{.x = &x_int, .out = o}, "X: expected float32, got int32"},
      {{.x = &x_elsewhere, .out = o},
       "X: on device 1, but the resources are on device 0"},
      {{.x = x, .out = nullptr}, "out is NULL"},
      {{.x = &x_rank1, .out = o}, "X: expected rank 2, got rank 1"},
      {{.x = &x_col, .out = o}, "X: expected row-major, got col-major 10x2"},
      {{.x = &x_three, .out = o}, "X: expected at least 4 rows, got 3"},
      {{.x = &x_no_rows, .out = o}, "X: expected at least 4 rows, got 0"},
      {{.x = &x_no_cols, .out = o}, "X: expected at least 1 column, got 0"},
      {{.x = &x_wide, .out = o},
       "X: 65536x32769 elements do not fit the index type"},
      {{.x = x, .out = o, .n_clusters = 0},
       "n-clusters: expected at least 1, got 0"},
      {{.x = x, .out = o, .n_clusters = -1},
       "n-clusters: expected at least 1, got -1"},
      {{.x = x, .out = o, .n_clusters = rows + 1},
       "X: 10 samples are fewer than the 11 clusters"},
      {{.x = x, .out = o, .n_components = 0, .auto_components = 0},
       "n-components: expected at least 1, got 0"},
      {{.x = x, .out = o, .n_components = -3, .auto_components = 0},
       "n-components: expected at least 1, got -3"},
      {{.x = x,
        .out = o,
        .n_components = past_components,
        .auto_components = 0},
       components},
      {{.x = x, .out = o, .n_neighbors = 1},
       "n-neighbors: expected at least 2, got 1"},
      {{.x = x, .out = o, .n_neighbors = -1},
       "n-neighbors: expected at least 2, got -1"},
      {{.x = x, .out = o, .n_init = 0}, "n-init: expected at least 1, got 0"},
      {{.x = x, .out = o, .n_init = -1}, "n-init: expected at least 1, got -1"},
      {{.x = x, .out = o, .eigen_tol = -1},
       "eigen-tol: expected at least 0, got -1"},
      {{.x = x, .out = o, .eigen_tol = nan},
       "eigen-tol: expected at least 0, got nan"},
      {{.x = x, .out = o, .eigen_tol = inf}, tol_most + "inf"},
      {{.x = x, .out = o, .eigen_tol = past_float}, tol_most + "1e+39"},
      {{.x = &x_graph, .out = o},
       "X: 102261127 rows with 10 neighbours need 2147483667 graph entries, "
       "more than 2147483647"},
      {{.x = &x_basis, .out = o, .n_neighbors = 2},
       "X: 107374183 rows and 2 components need a 107374183x20 eigensolver "
       "basis, more than 2147483647 elements"},
      {many_clusters(spectral_call{.x = &x_graph, .out = o}),
       "X: 102261127 samples and 65659970 clusters need a 21x102261127 "
       "k-means++ buffer, more than 2147483647 elements"},
      {many_clusters(
           spectral_call{.x = &x_trials_most, .out = o, .n_neighbors = 1}),
       "n-neighbors: expected at least 2, got 1"},
      {{.x = &x_basis_most, .out = o, .n_neighbors = 1},
       "n-neighbors: expected at least 2, got 1"},
      {{.x = &x_three_double, .out = o}, "X: expected float32, got float64"},
      {{.x = &x_three_no_cols, .out = o}, "X: expected at least 4 rows, got 3"},
      {{.x = &x_wide, .out = o, .n_clusters = 0},
       "X: 65536x32769 elements do not fit the index type"},
      {{.x = x,
        .out = o,
        .n_clusters = 0,
        .n_components = 0,
        .auto_components = 0},
       "n-clusters: expected at least 1, got 0"},
      {{.x = x,
        .out = o,
        .n_clusters = rows + 1,
        .n_components = past_components,
        .auto_components = 0},
       "X: 10 samples are fewer than the 11 clusters"},
      {{.x = x,
        .out = o,
        .n_components = past_components,
        .auto_components = 0,
        .n_neighbors = 1},
       components},
      {{.x = x, .out = o, .n_neighbors = 1, .n_init = 0},
       "n-neighbors: expected at least 2, got 1"},
      {{.x = x, .out = o, .n_init = 0, .eigen_tol = -1},
       "n-init: expected at least 1, got 0"},
      {{.x = &x_graph, .out = o, .eigen_tol = -1},
       "eigen-tol: expected at least 0, got -1"}};
  for (const refusal<spectral_call>& c : cases) {
    expect_refused(meter, out, spectral(res, c.call), c.message);
  }
}

TEST(SpectralRefusals, ExplicitComponentsAtTheBoundFit) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  values host;
  for (int64_t i = 0; i < rows * cols; ++i) {
    host.push_back(static_cast<double>((i * i) % spread));
  }
  const auto x = upload<float>(res, {rows, cols}, host);
  record out(res);
  const spectral_call call{
      .x = &x->view, .out = &out.r, .n_components = 3, .auto_components = 0};
  EXPECT_EQ(spectral(res, call), "accepted");
  EXPECT_EQ(shapes(out), (std::vector<int64_t>{1, RR_DTYPE_INT32, rows}));
}

struct norm_case {
  std::vector<double> row;
  std::string got;
};

void expect_norm_refusal(const resources& res, const norm_case& c) {
  SCOPED_TRACE(c.got);
  values host = with_row(c.row, bad_row, rows, cols);
  host[later_bad_row * cols] = std::numeric_limits<double>::quiet_NaN();
  const auto x = upload<float>(res, {rows, cols}, host);
  record out(res);
  const allocation_meter meter;
  EXPECT_EQ(spectral(res, {.x = &x->view, .out = &out.r}),
            norm_refusal(bad_row, c.got));
  EXPECT_EQ(cuml_cluster_last_error_kind(), RR_ERROR_LOGIC);
  EXPECT_LE(meter.peak(), scratch_bytes);
  EXPECT_EQ(meter.current(), 0);
  expect_untouched(out);
}

TEST(SpectralRefusals, LargeRowsAreRefusedBeforeAnyOutput) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const std::vector<norm_case> cases{
      {.row = {nan, 1}, .got = "nan"},
      {.row = {1, nan}, .got = "nan"},
      {.row = {inf, 0}, .got = "inf"},
      {.row = {-inf, 0}, .got = "inf"},
      {.row = {float_overflow, 0}, .got = "1.0000000400817551e+40"},
      {.row = {past_most, 0}, .got = "8.507059173023462e+37"}};
  for (const norm_case& c : cases) {
    expect_norm_refusal(res, c);
  }
}

TEST(SpectralRefusals, BoundaryRowsFit) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  values host;
  for (int64_t i = 0; i < rows * cols; ++i) {
    host.push_back(static_cast<double>(i));
  }
  host[bad_row * cols] = half_most;
  host[(bad_row * cols) + 1] = half_most;
  host[later_bad_row * cols] = -half_most;
  host[(later_bad_row * cols) + 1] = -half_most;
  host[(later_bad_row + 1) * cols] = std::numeric_limits<float>::denorm_min();
  const auto x = upload<float>(res, {rows, cols}, host);
  record out(res);
  EXPECT_EQ(spectral(res, {.x = &x->view, .out = &out.r}), "accepted");
  EXPECT_EQ(shapes(out), (std::vector<int64_t>{1, RR_DTYPE_INT32, rows}));
}

template <typename Edit>
graph_call edited(graph_call c, Edit edit) {
  edit(c);
  return c;
}

TEST(SpectralRefusals, GraphNameTheArgument) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const refusal_case<float> r(res, rows, cols);
  record out(res);
  const allocation_meter meter;
  rr_deposit* const o = &out.r;
  const rr_view ix = shaped(r.ints, {entries});
  const rr_view w = shaped(r.data, {entries});
  const rr_view ix_float = shaped(r.data, {entries});
  const rr_view ix_double = shaped(r.other, {entries});
  const rr_view w_int = shaped(r.ints, {entries});
  const rr_view w_double = shaped(r.other, {entries});
  const rr_view ix_rank2 = shaped(r.ints, {entries / 2, 2});
  const rr_view w_rank2 = shaped(r.data, {entries / 2, 2});
  const rr_view ix_short = shaped(r.ints, {entries - 1});
  const rr_view w_short = shaped(r.data, {entries - 1});
  const rr_view ix_elsewhere = on_device(ix, 1);
  const rr_view ix_forged = shaped(r.ints, {forged_entries});
  const rr_view w_forged = shaped(r.data, {forged_entries});
  const std::string components =
      "n-components: expected at most 3 for 10 samples, got 4";
  const graph_call ok{
      .rows = &ix, .cols = &ix, .vals = &w, .n_samples = rows, .out = o};
  const std::vector<refusal<graph_call>> cases{
      {{.rows = &ix_float,
        .cols = &ix,
        .vals = &w,
        .n_samples = rows,
        .out = o},
       "rows: expected int32, got float32"},
      {{.rows = &ix_double,
        .cols = &ix,
        .vals = &w,
        .n_samples = rows,
        .out = o},
       "rows: expected int32, got float64"},
      {{.rows = &ix,
        .cols = &ix_float,
        .vals = &w,
        .n_samples = rows,
        .out = o},
       "cols: expected int32, got float32"},
      {{.rows = &ix, .cols = &ix, .vals = &w_int, .n_samples = rows, .out = o},
       "vals: expected float32, got int32"},
      {{.rows = &ix,
        .cols = &ix,
        .vals = &w_double,
        .n_samples = rows,
        .out = o},
       "vals: expected float32, got float64"},
      {{.rows = &ix_rank2,
        .cols = &ix,
        .vals = &w,
        .n_samples = rows,
        .out = o},
       "rows: expected rank 1, got rank 2"},
      {{.rows = &ix,
        .cols = &ix,
        .vals = &w_rank2,
        .n_samples = rows,
        .out = o},
       "vals: expected rank 1, got rank 2"},
      {{.rows = &ix_elsewhere,
        .cols = &ix,
        .vals = &w,
        .n_samples = rows,
        .out = o},
       "rows: on device 1, but the resources are on device 0"},
      {{.rows = &ix,
        .cols = &ix,
        .vals = &w,
        .n_samples = rows,
        .out = nullptr},
       "out is NULL"},
      {{.rows = &ix,
        .cols = &ix_short,
        .vals = &w,
        .n_samples = rows,
        .out = o},
       "cols: expected 12 entries, as rows has, got 11"},
      {{.rows = &ix,
        .cols = &ix,
        .vals = &w_short,
        .n_samples = rows,
        .out = o},
       "vals: expected 12 entries, as rows has, got 11"},
      {{.rows = &ix, .cols = &ix, .vals = &w, .n_samples = 3, .out = o},
       "n-samples: expected at least 4, got 3"},
      {{.rows = &ix, .cols = &ix, .vals = &w, .n_samples = -1, .out = o},
       "n-samples: expected at least 4, got -1"},
      {{.rows = &ix,
        .cols = &ix,
        .vals = &w,
        .n_samples = past_index,
        .out = o},
       "n-samples: expected at most 2147483647, got 2147483648"},
      {{.rows = &ix_forged,
        .cols = &ix_forged,
        .vals = &w_forged,
        .n_samples = forged_samples,
        .out = o},
       "rows: 2147483000 entries and 1000 samples do not fit the index type"},
      {edited(ok, [](graph_call& c) { c.n_clusters = 0; }),
       "n-clusters: expected at least 1, got 0"},
      {edited(ok, [](graph_call& c) { c.n_clusters = rows + 1; }),
       "n-samples: 10 samples are fewer than the 11 clusters"},
      {edited(ok,
              [](graph_call& c) {
                c.auto_components = 0;
                c.n_components = 0;
              }),
       "n-components: expected at least 1, got 0"},
      {edited(ok,
              [](graph_call& c) {
                c.auto_components = 0;
                c.n_components = past_components;
              }),
       components},
      {edited(many_clusters(ok),
              [](graph_call& c) { c.n_samples = graph_rows; }),
       "n-samples: 102261127 samples and 65659970 clusters need a "
       "21x102261127 k-means++ buffer, more than 2147483647 elements"},
      {edited(many_clusters(ok),
              [](graph_call& c) {
                c.n_samples = graph_rows - 1;
                c.eigen_tol = -1;
              }),
       "eigen-tol: expected at least 0, got -1"},
      {edited(many_clusters(ok, trial_clusters - 1),
              [](graph_call& c) {
                c.n_samples = graph_rows;
                c.eigen_tol = -1;
              }),
       "eigen-tol: expected at least 0, got -1"},
      {edited(ok, [](graph_call& c) { c.n_samples = basis_rows; }),
       "n-samples: 107374183 samples and 2 components need a 107374183x20 "
       "eigensolver basis, more than 2147483647 elements"},
      {edited(ok,
              [](graph_call& c) {
                c.n_samples = basis_rows - 1;
                c.eigen_tol = -1;
              }),
       "eigen-tol: expected at least 0, got -1"},
      {edited(ok, [](graph_call& c) { c.n_init = 0; }),
       "n-init: expected at least 1, got 0"},
      {edited(ok, [](graph_call& c) { c.eigen_tol = -1; }),
       "eigen-tol: expected at least 0, got -1"},
      {edited(ok,
              [](graph_call& c) {
                c.eigen_tol = std::numeric_limits<double>::quiet_NaN();
              }),
       "eigen-tol: expected at least 0, got nan"},
      {edited(ok, [](graph_call& c) { c.eigen_tol = past_float; }),
       "eigen-tol: expected at most 3.4028234663852886e+38, got 1e+39"},
      {{.rows = &ix, .cols = &ix_short, .vals = &w, .n_samples = 3, .out = o},
       "cols: expected 12 entries, as rows has, got 11"},
      {{.rows = &ix_forged,
        .cols = &ix_forged,
        .vals = &w_forged,
        .n_samples = 3,
        .out = o},
       "n-samples: expected at least 4, got 3"},
      {edited(ok,
              [](graph_call& c) {
                c.n_clusters = 0;
                c.n_init = 0;
              }),
       "n-clusters: expected at least 1, got 0"}};
  for (const refusal<graph_call>& g : cases) {
    expect_refused(meter, out, spectral_graph(res, g.call), g.message);
  }
}

}  // namespace
}  // namespace cumlrkt::test
