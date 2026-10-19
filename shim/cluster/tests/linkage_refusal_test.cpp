#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "cluster_fixture.hpp"
#include "cumlrkt/cluster.h"
#include "kmeans_fixture.hpp"
#include "linkage_fixture.hpp"
#include "raftrkt/c_api.h"
#include "refusal_fixture.hpp"

namespace cumlrkt::test {
namespace {

constexpr int64_t pairwise_most_rows = 46340;
constexpr int64_t knn_rows = 28000000;
constexpr int32_t most_c = 2147483617;
constexpr int32_t unknown_code = 2;
constexpr int64_t capped_rows = 2048;
constexpr int64_t capped_side = 64;
constexpr std::size_t capped_bytes = 4 << 20;

TEST(LinkageRefusals, NameTheArgument) {
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
  const rr_view x_no_rows = shaped(r.data, {0, cols});
  const rr_view x_no_cols = shaped(r.data, {rows, 0});
  const rr_view x_nothing = shaped(r.data, {0, 0});
  const rr_view x_wide = shaped(r.data, {wide_rows, wide_cols});
  const rr_view x_pairwise = shaped(r.data, {pairwise_most_rows + 1, 1});
  const rr_view x_knn = shaped(r.data, {knn_rows, 1});
  const rr_view x_elsewhere = on_device(r.x, 1);
  const std::string connectivity_message =
      "connectivity: expected knn or pairwise";
  const std::string metric_message = "metric: expected euclidean";
  const std::string not_float = "X: expected float32, got int32";
  const std::string n_clusters_message =
      "n-clusters: expected at least 1, got 0";
  const std::string c_message =
      "c: expected at most 2147483617, got 2147483618";
  const std::string pairwise_message =
      "X: pairwise connectivity holds at most 46340 rows, got 46341; use knn";
  const std::vector<refusal<linkage_call>> cases{
      {{.x = &x_double, .out = o}, "X: expected float32, got float64"},
      {{.x = &x_int, .out = o}, not_float},
      {{.x = &x_elsewhere, .out = o},
       "X: on device 1, but the resources are on device 0"},
      {{.x = x, .out = nullptr}, "out is NULL"},
      {{.x = x, .out = o, .connectivity = unknown_code}, connectivity_message},
      {{.x = x, .out = o, .connectivity = -1}, connectivity_message},
      {{.x = x, .out = o, .metric = 1}, metric_message},
      {{.x = x, .out = o, .metric = -1}, metric_message},
      {{.x = &x_rank1, .out = o}, "X: expected rank 2, got rank 1"},
      {{.x = &x_col, .out = o}, "X: expected row-major, got col-major 10x2"},
      {{.x = &x_no_rows, .out = o}, "X: expected at least 1 row, got 0"},
      {{.x = &x_no_cols, .out = o}, "X: expected at least 1 column, got 0"},
      {{.x = &x_wide, .out = o},
       "X: 65536x32769 elements do not fit the index type"},
      {{.x = x, .out = o, .n_clusters = 0}, n_clusters_message},
      {{.x = x, .out = o, .n_clusters = -1},
       "n-clusters: expected at least 1, got -1"},
      {{.x = x, .out = o, .n_clusters = rows + 1},
       "X: 10 samples are fewer than the 11 clusters"},
      {{.x = x, .out = o, .c = most_c + 1}, c_message},
      {{.x = x, .out = o, .c = std::numeric_limits<int32_t>::max()},
       "c: expected at most 2147483617, got 2147483647"},
      {{.x = &x_pairwise, .out = o, .connectivity = pairwise},
       pairwise_message},
      {{.x = &x_knn, .out = o},
       "X: 28000000 rows with 39 neighbours need 2184000000 knn graph "
       "entries, more than 2147483647"},
      {{.x = &x_int, .out = o, .connectivity = unknown_code}, not_float},
      {{.x = x, .out = o, .metric = 1, .connectivity = unknown_code},
       connectivity_message},
      {{.x = &x_rank1, .out = o, .metric = 1}, metric_message},
      {{.x = &x_nothing, .out = o}, "X: expected at least 1 row, got 0"},
      {{.x = &x_wide, .out = o, .n_clusters = 0},
       "X: 65536x32769 elements do not fit the index type"},
      {{.x = x, .out = o, .n_clusters = 0, .c = most_c + 1},
       n_clusters_message},
      {{.x = x, .out = o, .n_clusters = rows + 1, .c = most_c + 1},
       "X: 10 samples are fewer than the 11 clusters"},
      {{.x = &x_pairwise, .out = o, .connectivity = pairwise, .c = most_c + 1},
       c_message},
      {{.x = &x_knn, .out = o, .connectivity = pairwise},
       "X: pairwise connectivity holds at most 46340 rows, got 28000000; use "
       "knn"}};
  for (const refusal<linkage_call>& c : cases) {
    const std::string status = linkage(res, c.call);
    EXPECT_EQ(status, c.message);
    EXPECT_EQ(cuml_cluster_last_error_kind(), RR_ERROR_LOGIC) << status;
    EXPECT_EQ(meter.peak(), 0) << status;
    expect_untouched(out);
  }
}

struct norm_case {
  std::vector<double> row;
  std::string got;
};

void expect_norm_refusal(const resources& res, int32_t connectivity,
                         const norm_case& c) {
  values host = with_row(c.row, bad_row, rows, cols);
  host[later_bad_row * cols] = std::numeric_limits<double>::quiet_NaN();
  const auto x = upload<float>(res, {rows, cols}, host);
  record out(res);
  const allocation_meter meter;
  const linkage_call call{
      .x = &x->view, .out = &out.r, .connectivity = connectivity};
  EXPECT_EQ(linkage(res, call), norm_refusal(bad_row, c.got));
  EXPECT_EQ(cuml_cluster_last_error_kind(), RR_ERROR_LOGIC);
  EXPECT_LE(meter.peak(), scratch_bytes);
  EXPECT_EQ(meter.current(), 0);
  expect_untouched(out);
}

TEST(LinkageRefusals, LargeRowsAreRefusedBeforeAnyOutput) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const std::vector<norm_case> cases{
      {.row = {nan, 1}, .got = "nan"},
      {.row = {1, nan}, .got = "nan"},
      {.row = {inf, 0}, .got = "inf"},
      {.row = {-inf, 0}, .got = "inf"},
      {.row = {inf, nan}, .got = "nan"},
      {.row = {float_overflow, 0}, .got = "1.0000000400817551e+40"},
      {.row = {past_most, 0}, .got = "8.507059173023462e+37"},
      {.row = {-past_most, 0}, .got = "8.507059173023462e+37"}};
  for (const int32_t connectivity : {pairwise, knn}) {
    for (const norm_case& c : cases) {
      SCOPED_TRACE(std::to_string(c.row[0]) + " " + std::to_string(c.row[1]));
      expect_norm_refusal(res, connectivity, c);
    }
  }
}

TEST(LinkageRefusals, BoundaryRowsFit) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const float subnormal = std::numeric_limits<float>::denorm_min();
  for (const int32_t connectivity : {pairwise, knn}) {
    values host = with_row({half_most, half_most}, bad_row, rows, cols);
    host[later_bad_row * cols] = -half_most;
    host[(later_bad_row * cols) + 1] = -half_most;
    host[(later_bad_row + 1) * cols] = subnormal;
    const auto x = upload<float>(res, {rows, cols}, host);
    record out(res);
    const linkage_call call{
        .x = &x->view, .out = &out.r, .connectivity = connectivity};
    EXPECT_EQ(linkage(res, call), "accepted");
    EXPECT_EQ(shapes(out), (std::vector<int64_t>{2, RR_DTYPE_INT32, rows,
                                                 RR_DTYPE_INT32, rows - 1, 2}));
  }
}

TEST(LinkageRefusals, PairwiseLimitPrecedesTheRowScan) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  values host(static_cast<std::size_t>(pairwise_most_rows + 1), 1.0);
  host[bad_row] = std::numeric_limits<double>::quiet_NaN();
  const auto x = upload<float>(res, {pairwise_most_rows + 1, 1}, host);
  record out(res);
  const allocation_meter meter;
  const linkage_call call{
      .x = &x->view, .out = &out.r, .connectivity = pairwise};
  EXPECT_EQ(linkage(res, call),
            "X: pairwise connectivity holds at most 46340 rows, got 46341; "
            "use knn");
  EXPECT_EQ(meter.peak(), 0);
  expect_untouched(out);
  const linkage_call by_knn{.x = &x->view, .out = &out.r};
  EXPECT_EQ(linkage(res, by_knn), norm_refusal(bad_row, "nan"));
  expect_untouched(out);
}

TEST(LinkageRefusals, LargestCIsAccepted) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  values host;
  for (int64_t i = 0; i < rows * cols; ++i) {
    host.push_back(static_cast<double>(i * i));
  }
  const auto x = upload<float>(res, {rows, cols}, host);
  for (const int32_t c : {most_c, std::numeric_limits<int32_t>::min()}) {
    record out(res);
    const linkage_call call{.x = &x->view, .out = &out.r, .c = c};
    EXPECT_EQ(linkage(res, call), "accepted") << c;
  }
}

values capped_grid() {
  values host;
  for (int64_t a = 0; a < capped_side; ++a) {
    for (int64_t b = 0; b < capped_rows / capped_side; ++b) {
      host.push_back(static_cast<double>(a));
      host.push_back(static_cast<double>(b));
    }
  }
  return host;
}

TEST(LinkageRefusals, PairwiseOutOfMemoryIsOomAndFreesTheOutputs) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const auto x = upload<float>(res, {capped_rows, cols}, capped_grid());
  record out(res);
  const allocation_meter meter;
  const pool_cap cap(capped_bytes);
  const linkage_call call{
      .x = &x->view, .out = &out.r, .connectivity = pairwise};
  const std::string status = linkage(res, call);
  EXPECT_NE(status.find("out_of_memory"), std::string::npos) << status;
  EXPECT_NE(status.find("Requested bytes 16777216"), std::string::npos)
      << status;
  EXPECT_EQ(cuml_cluster_last_error_kind(), RR_ERROR_OOM) << status;
  EXPECT_GT(meter.peak(), 0);
  EXPECT_EQ(meter.current(), 0);
  expect_untouched(out);
}

TEST(LinkageRefusals, FilledOutIsRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  values host;
  for (int64_t i = 0; i < rows * cols; ++i) {
    host.push_back(static_cast<double>(i));
  }
  const auto x = upload<float>(res, {rows, cols}, host);
  record filled(res);
  const linkage_call c{.x = &x->view, .out = &filled.r};
  ASSERT_EQ(linkage(res, c), "accepted");
  const void* owner = filled.r.owner;
  EXPECT_EQ(linkage(res, c), "out: already filled");
  EXPECT_EQ(filled.r.owner, owner);
  EXPECT_EQ(filled.r.count, 2);
}

}  // namespace
}  // namespace cumlrkt::test
