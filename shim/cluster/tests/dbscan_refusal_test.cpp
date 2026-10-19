#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "cluster_fixture.hpp"
#include "cumlrkt/cluster.h"
#include "dbscan_fixture.hpp"
#include "kmeans_fixture.hpp"
#include "raftrkt/c_api.h"
#include "refusal_fixture.hpp"

namespace cumlrkt::test {
namespace {

constexpr int64_t rows = 10;
constexpr int64_t cols = 2;
constexpr int64_t zero_row = 2;
constexpr std::size_t pool = 64;
constexpr int64_t forged_square = 46341;
constexpr int64_t int32_most_rows = 2147483645;
constexpr int64_t most_rows = 2147483647;
constexpr int64_t forged_mbytes = 30000;
constexpr int64_t crashing_rows = 150000;
constexpr int64_t automatic_rows = 69000;
constexpr int32_t unknown_metric = 3;
constexpr int64_t most_mbytes = 18446744073709;
constexpr double tiny = 1e-50;
constexpr double huge = 1e300;
constexpr double float_overflow = 1e20;
constexpr double float_large = 1e19;
constexpr double float_small = 1e-20;
constexpr double double_underflow = 1e-160;
constexpr double double_overflow = 1e200;
constexpr double pair_eps = 0.1;
constexpr std::size_t usable_percent = 80;
constexpr std::size_t percent = 100;
constexpr std::size_t automatic_target = 1500000;

std::string out_name(int32_t out_dtype) {
  return out_dtype == RR_DTYPE_INT32 ? "int32" : "int64";
}

template <typename T>
struct refusal_case {
  explicit refusal_case(const resources& res)
      : data(res.r, float_dtype<T>, {pool}),
        other(res.r, float_dtype<other_t<T>>, {pool}),
        ints(res.r, RR_DTYPE_INT32, {pool}),
        x(shaped(data, {rows, cols})) {
    data.write(std::vector<T>(pool, 1));
  }

  [[nodiscard]] dbscan_call call(record& out, int32_t out_dtype) const {
    return {.x = &x, .out = &out.r, .out_dtype = out_dtype};
  }

  device_array data;
  device_array other;
  device_array ints;
  rr_view x;
};

rr_view strided(rr_view v, int64_t stride) {
  v.strides[0] = stride;
  return v;
}

int64_t automatic_columns() {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  EXPECT_EQ(cudaMemGetInfo(&free_bytes, &total_bytes), cudaSuccess);
  const std::size_t usable = usable_percent * total_bytes / percent;
  const std::size_t row_bytes = automatic_rows * sizeof(float);
  return static_cast<int64_t>((usable - automatic_target) / row_bytes);
}

template <typename T>
void fit_refusals(int32_t out_dtype) {
  SCOPED_TRACE(name<T>() + " " + out_name(out_dtype));
  const resources res;
  const refusal_case<T> r(res);
  record out(res);
  const allocation_meter meter;
  const auto refused = [&](const dbscan_call& c) {
    std::string status = dbscan(res, c);
    EXPECT_EQ(cuml_cluster_last_error_kind(), RR_ERROR_LOGIC) << status;
    EXPECT_EQ(meter.peak(), 0) << status;
    expect_untouched(out);
    return status;
  };
  const std::string t = name<T>();
  const std::string u = name<other_t<T>>();
  const bool narrow = out_dtype == RR_DTYPE_INT32;
  const rr_view x_int = shaped(r.ints, {rows, cols});
  const rr_view x_rank1 = shaped(r.data, {rows});
  const rr_view x_col = shaped(r.data, {rows, cols}, true);
  const rr_view x_no_rows = shaped(r.data, {0, cols});
  const rr_view x_no_cols = shaped(r.data, {rows, 0});
  const rr_view x_square = shaped(r.data, {forged_square, forged_square});
  const rr_view x_crashing = shaped(r.data, {crashing_rows, cols});
  const rr_view x_int32_most = shaped(r.data, {int32_most_rows, 1});
  const rr_view x_int32_over = shaped(r.data, {int32_most_rows + 1, 1});
  const rr_view x_most = shaped(r.data, {most_rows, 1});
  const rr_view x_over = shaped(r.data, {most_rows + 1, 1});
  const rr_view x_elsewhere = on_device(r.x, 1);
  const rr_view w_other = shaped(r.other, {rows});
  const rr_view w_short = shaped(r.data, {rows - 1});
  const rr_view w_rank2 = shaped(r.data, {rows, 1});
  const rr_view w_strided = strided(shaped(r.data, {rows}), 2);
  const dbscan_call base = r.call(out, out_dtype);
  dbscan_call c = base;

  c.x = &x_int;
  EXPECT_EQ(refused(c), "X: expected float32 or float64, got int32");
  c.x = &x_elsewhere;
  EXPECT_EQ(refused(c), "X: on device 1, but the resources are on device 0");
  c = base;
  c.out = nullptr;
  EXPECT_EQ(refused(c), "out is NULL");
  c = base;
  c.out_dtype = RR_DTYPE_FLOAT32;
  EXPECT_EQ(refused(c), "out-dtype: expected int32 or int64");
  c = base;
  c.metric = unknown_metric;
  EXPECT_EQ(refused(c), "metric: expected euclidean, cosine or precomputed");
  c.metric = -1;
  EXPECT_EQ(refused(c), "metric: expected euclidean, cosine or precomputed");
  c = base;
  c.x = &x_rank1;
  EXPECT_EQ(refused(c), "X: expected rank 2, got rank 1");
  c.x = &x_col;
  EXPECT_EQ(refused(c), "X: expected row-major, got col-major 10x2");
  c.x = &x_no_rows;
  EXPECT_EQ(refused(c), "X: expected at least 1 row, got 0");
  c.x = &x_no_cols;
  EXPECT_EQ(refused(c), "X: expected at least 1 column, got 0");
  c = base;
  c.metric = CUML_CLUSTER_DBSCAN_METRIC_PRECOMPUTED;
  EXPECT_EQ(refused(c), "X: expected a square distance matrix, got 10x2");
  c = base;
  c.x = &x_square;
  c.metric = CUML_CLUSTER_DBSCAN_METRIC_PRECOMPUTED;
  c.eps = 0;
  EXPECT_EQ(refused(c),
            narrow ? "X: 46341x46341 elements do not fit int32 labels; use "
                     "int64"
                   : "eps: expected greater than 0, got 0");
  c = base;
  c.x = &x_int32_over;
  c.max_mbytes_per_batch = forged_mbytes;
  EXPECT_EQ(refused(c),
            narrow ? "X: 2147483646 rows do not fit int32 labels; use int64"
                   : "max-mbytes-per-batch: expected at least 38655 for "
                     "2147483646 rows, got 30000");
  c.x = &x_int32_most;
  c.eps = 0;
  EXPECT_EQ(refused(c), "eps: expected greater than 0, got 0");
  c = base;
  c.x = &x_over;
  c.eps = 0;
  EXPECT_EQ(refused(c), "X: expected at most 2147483647 rows, got 2147483648");
  c.x = &x_most;
  EXPECT_EQ(refused(c),
            narrow ? "X: 2147483647 rows do not fit int32 labels; use int64"
                   : "eps: expected greater than 0, got 0");
  c = base;
  c.sample_weight = &w_other;
  EXPECT_EQ(refused(c), "sample-weight: expected " + t + ", got " + u);
  c.sample_weight = &w_short;
  EXPECT_EQ(refused(c), "sample-weight: expected 10 elements, got 9");
  c.sample_weight = &w_rank2;
  EXPECT_EQ(refused(c), "sample-weight: expected rank 1, got rank 2");
  c.sample_weight = &w_strided;
  EXPECT_EQ(refused(c),
            "sample-weight: expected contiguous elements, got stride 2");
  c = base;
  c.eps = 0;
  EXPECT_EQ(refused(c), "eps: expected greater than 0, got 0");
  c.eps = -0.0;
  EXPECT_EQ(refused(c), "eps: expected greater than 0, got 0");
  c.eps = -1;
  EXPECT_EQ(refused(c), "eps: expected greater than 0, got -1");
  c.eps = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(refused(c), "eps: expected greater than 0, got nan");
  c.eps = std::numeric_limits<double>::infinity();
  EXPECT_EQ(refused(c),
            std::string("eps: expected at most ") +
                (std::is_same_v<T, float> ? "3.4028234663852886e+38"
                                          : "1.7976931348623157e+308") +
                ", got inf");
  c.eps = tiny;
  if (std::is_same_v<T, float>) {
    EXPECT_EQ(refused(c), "eps: expected greater than 0, got 1e-50");
    c.eps = huge;
    EXPECT_EQ(refused(c),
              "eps: expected at most 3.4028234663852886e+38, got 1e+300");
  }
  c = base;
  c.min_samples = 0;
  EXPECT_EQ(refused(c), "min-samples: expected at least 1, got 0");
  c = base;
  c.max_mbytes_per_batch = -1;
  EXPECT_EQ(refused(c), "max-mbytes-per-batch: expected at least 0, got -1");
  c.max_mbytes_per_batch = most_mbytes + 1;
  EXPECT_EQ(refused(c),
            "max-mbytes-per-batch: expected at most 18446744073709, got "
            "18446744073710");
  c = base;
  c.x = &x_crashing;
  c.max_mbytes_per_batch = 1;
  EXPECT_EQ(refused(c),
            std::string("max-mbytes-per-batch: expected at least ") +
                (narrow ? "2" : "3") + " for 150000 rows, got 1");
  c = base;
  c.min_samples = 0;
  c.eps = 0;
  EXPECT_EQ(refused(c), "eps: expected greater than 0, got 0");
  c = base;
  c.x = &x_int;
  c.out_dtype = RR_DTYPE_FLOAT32;
  EXPECT_EQ(refused(c), "X: expected float32 or float64, got int32");
  c = base;
  c.out_dtype = RR_DTYPE_FLOAT32;
  c.metric = unknown_metric;
  EXPECT_EQ(refused(c), "out-dtype: expected int32 or int64");
  c = base;
  c.metric = unknown_metric;
  c.x = &x_rank1;
  EXPECT_EQ(refused(c), "metric: expected euclidean, cosine or precomputed");
  c = base;
  c.sample_weight = &w_short;
  c.eps = 0;
  EXPECT_EQ(refused(c), "sample-weight: expected 10 elements, got 9");
  c = base;
  c.min_samples = 0;
  c.max_mbytes_per_batch = -1;
  EXPECT_EQ(refused(c), "min-samples: expected at least 1, got 0");
}

template <typename T>
void automatic_batch_refusal() {
  SCOPED_TRACE(name<T>());
  const resources res;
  const refusal_case<T> r(res);
  record out(res);
  const int64_t columns = automatic_columns();
  const rr_view x = shaped(r.data, {automatic_rows, columns});
  const allocation_meter meter;
  dbscan_call c = r.call(out, RR_DTYPE_INT64);
  c.x = &x;
  if (std::is_same_v<T, float>) {
    EXPECT_EQ(dbscan(res, c),
              "max-mbytes-per-batch: the automatic size holds no batch of "
              "69000 rows; pass at least 2");
    EXPECT_EQ(cuml_cluster_last_error_kind(), RR_ERROR_LOGIC);
  }
  c.max_mbytes_per_batch = 1;
  c.eps = 0;
  EXPECT_EQ(dbscan(res, c), "eps: expected greater than 0, got 0");
  EXPECT_EQ(meter.peak(), 0);
  expect_untouched(out);
}

template <typename T>
void zero_cosine_rows() {
  SCOPED_TRACE(name<T>());
  const resources res;
  for (const std::vector<double>& bad :
       std::vector<std::vector<double>>{{0, 0}, {-0.0, -0.0}}) {
    SCOPED_TRACE(std::to_string(bad[0]));
    values host(static_cast<std::size_t>(rows * cols), 1.0);
    std::copy(bad.begin(), bad.end(),
              host.begin() + static_cast<std::ptrdiff_t>(zero_row * cols));
    const auto x = upload<T>(res, {rows, cols}, host);
    record out(res);
    const allocation_meter meter;
    const dbscan_call c{.x = &x->view,
                        .out = &out.r,
                        .metric = CUML_CLUSTER_DBSCAN_METRIC_COSINE};
    EXPECT_EQ(dbscan(res, c), "X: row 2 is zero, so no cosine distance");
    EXPECT_EQ(cuml_cluster_last_error_kind(), RR_ERROR_LOGIC);
    EXPECT_EQ(meter.current(), 0);
    expect_untouched(out);
  }
}

struct cosine_row {
  std::vector<double> row;
  values labels;
};

template <typename T>
void extreme_cosine_rows() {
  SCOPED_TRACE(name<T>());
  const resources res;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const values paired{0, 0, 1, 1, 0, 0, 0, 0, 0, 0};
  const values joined{0, 0, 0, -1, 0, 0, 0, 0, 0, 0};
  const values noise{0, 0, -1, -1, 0, 0, 0, 0, 0, 0};
  std::vector<cosine_row> cases{
      {.row = {float_large, 0}, .labels = paired},
      {.row = {float_small, 0}, .labels = paired},
      {.row = {std::numeric_limits<T>::denorm_min(), 0}, .labels = paired},
      {.row = {float_overflow, 0}, .labels = paired},
      {.row = {float_overflow, -0.0}, .labels = paired},
      {.row = {float_overflow, float_overflow}, .labels = joined},
      {.row = {nan, 1}, .labels = noise},
      {.row = {inf, 0}, .labels = noise},
      {.row = {-inf, inf}, .labels = noise},
      {.row = {nan, nan}, .labels = noise}};
  if (std::is_same_v<T, double>) {
    cases.push_back({.row = {double_underflow, double_underflow}, .labels = joined});
    cases.push_back({.row = {double_overflow, 0}, .labels = paired});
  }
  for (const cosine_row& edge : cases) {
    SCOPED_TRACE(std::to_string(edge.row[0]) + " " +
                 std::to_string(edge.row[1]));
    values host(static_cast<std::size_t>(rows * cols), 1.0);
    std::copy(edge.row.begin(), edge.row.end(),
              host.begin() + static_cast<std::ptrdiff_t>(zero_row * cols));
    host[(zero_row + 1) * cols] = 2;
    host[((zero_row + 1) * cols) + 1] = 0;
    const auto x = upload<T>(res, {rows, cols}, host);
    record out(res);
    const dbscan_call c{.x = &x->view,
                        .out = &out.r,
                        .eps = pair_eps,
                        .min_samples = 2,
                        .metric = CUML_CLUSTER_DBSCAN_METRIC_COSINE};
    ASSERT_EQ(dbscan(res, c), "accepted");
    EXPECT_EQ(download<int32_t>(out, 0), edge.labels);
  }
}

TEST(DbscanRefusals, NameTheArgument) {
  CUMLRKT_REQUIRE_GPU();
  for (const int32_t out_dtype : {RR_DTYPE_INT32, RR_DTYPE_INT64}) {
    fit_refusals<float>(out_dtype);
    fit_refusals<double>(out_dtype);
  }
}

TEST(DbscanRefusals, AutomaticBatchSize) {
  CUMLRKT_REQUIRE_GPU();
  automatic_batch_refusal<float>();
  automatic_batch_refusal<double>();
}

TEST(DbscanRefusals, ZeroCosineRow) {
  CUMLRKT_REQUIRE_GPU();
  zero_cosine_rows<float>();
  zero_cosine_rows<double>();
}

TEST(DbscanRefusals, ExtremeCosineRowsAreLabelled) {
  CUMLRKT_REQUIRE_GPU();
  extreme_cosine_rows<float>();
  extreme_cosine_rows<double>();
}

TEST(DbscanRefusals, FilledOutIsRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const auto x = upload<double>(res, {rows, cols}, values(rows * cols, 1.0));
  record filled(res);
  const dbscan_call c{.x = &x->view, .out = &filled.r};
  ASSERT_EQ(dbscan(res, c), "accepted");
  const void* owner = filled.r.owner;
  EXPECT_EQ(dbscan(res, c), "out: already filled");
  EXPECT_EQ(filled.r.owner, owner);
  EXPECT_EQ(filled.r.count, 2);
}

}  // namespace
}  // namespace cumlrkt::test
