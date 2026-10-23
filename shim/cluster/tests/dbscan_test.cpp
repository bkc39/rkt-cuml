#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "cluster_fixture.hpp"
#include "cosine_eps.hpp"
#include "cumlrkt/cluster.h"
#include "dbscan_fixture.hpp"
#include "kmeans_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {
namespace {

constexpr int64_t grid_rows = 8;
constexpr int32_t grid_min_samples = 4;
constexpr double grid_eps = 2;
constexpr int64_t cloud_rows = 1000;
constexpr int64_t cloud_cols = 2;
constexpr int64_t cloud_extent = 30;
constexpr double cloud_eps = 1.5;
constexpr int64_t small_rows = 40;
constexpr int64_t small_cols = 3;
constexpr double third = 1.0 / 3.0;
constexpr uint64_t lcg_a = 6364136223846793005ULL;
constexpr uint64_t lcg_c = 1442695040888963407ULL;
constexpr int lcg_shift = 33;
constexpr int64_t wide_cols = 8;
constexpr double wide_eps = 1e-3;
constexpr double wide_scale = 3;
constexpr int sweep_steps = 5000;
constexpr double sweep_step = 1e-3;
constexpr double tiny_eps = 1e-30;
constexpr double small_eps = 1e-6;
constexpr double wide_cosine_eps = 9;
constexpr double float_second_step_eps = 0x1.062566p-11;
constexpr double subnormal_eps = 1e-318;
constexpr double deep_subnormal_eps = 1e-320;
constexpr double float_min_eps = 7.1e-46;
constexpr double float_subnormal_eps = 1e-40;

const values grid{0, 1, 2, 4, 6, 7, 8, 20};
const values grid_weights{4, 1, 1, 1, 1, 1, 1, 1};
const values grid_labels{0, 0, 0, 0, 1, 1, 1, -1};
const values grid_core{2, 4};
const values weighted_core{0, 1, 2, 4};

template <typename T>
std::string name() {
  return std::is_same_v<T, float> ? "float32" : "float64";
}

values labels_of(const record& out, int32_t out_dtype, int32_t region) {
  return out_dtype == RR_DTYPE_INT32 ? download<int32_t>(out, region)
                                     : download<int64_t>(out, region);
}

struct grid_shape {
  int64_t rows;
  int64_t cols;
};

values cloud(grid_shape shape, int64_t extent) {
  values host;
  uint64_t state = 1;
  for (int64_t i = 0; i < shape.rows * shape.cols; ++i) {
    state = (state * lcg_a) + lcg_c;
    host.push_back(static_cast<double>((state >> lcg_shift) %
                                       static_cast<uint64_t>(extent)));
  }
  return host;
}

template <typename T>
values distances(const values& x, grid_shape shape) {
  const int64_t rows = shape.rows;
  const int64_t cols = shape.cols;
  values d;
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < rows; ++j) {
      double s = 0;
      for (int64_t k = 0; k < cols; ++k) {
        const double e = x[(i * cols) + k] - x[(j * cols) + k];
        s += e * e;
      }
      d.push_back(static_cast<double>(std::sqrt(static_cast<T>(s))));
    }
  }
  return d;
}

struct labelling {
  std::string status;
  values labels;
  values core;
};

labelling fitted(const resources& res, dbscan_call c) {
  record out(res);
  c.out = &out.r;
  const std::string status = dbscan(res, c);
  if (status != "accepted") {
    return {status, {}, {}};
  }
  return {status, labels_of(out, c.out_dtype, 0),
          labels_of(out, c.out_dtype, 1)};
}

template <typename T>
void hand_grid(int32_t out_dtype) {
  SCOPED_TRACE(name<T>() + (out_dtype == RR_DTYPE_INT32 ? " int32" : " int64"));
  const resources res;
  const auto x = upload<T>(res, {grid_rows, 1}, grid);
  const auto w = upload<T>(res, {grid_rows}, grid_weights);
  record out(res);
  const dbscan_call c{.x = &x->view,
                      .out = &out.r,
                      .eps = grid_eps,
                      .min_samples = grid_min_samples,
                      .out_dtype = out_dtype};
  ASSERT_EQ(dbscan(res, c), "accepted");
  EXPECT_EQ(shapes(out),
            (std::vector<int64_t>{2, out_dtype, grid_rows, out_dtype, 2}));
  EXPECT_EQ(labels_of(out, out_dtype, 0), grid_labels);
  EXPECT_EQ(labels_of(out, out_dtype, 1), grid_core);
  dbscan_call weighted = c;
  weighted.sample_weight = &w->view;
  const labelling found = fitted(res, weighted);
  EXPECT_EQ(found.status, "accepted");
  EXPECT_EQ(found.core, weighted_core);
}

TEST(Dbscan, HandGrid) {
  CUMLRKT_REQUIRE_GPU();
  for (const int32_t out_dtype : {RR_DTYPE_INT32, RR_DTYPE_INT64}) {
    hand_grid<float>(out_dtype);
    hand_grid<double>(out_dtype);
  }
}

template <typename T>
void leaves_x_unchanged() {
  SCOPED_TRACE(name<T>());
  const resources res;
  const values host = cloud({small_rows, small_cols}, cloud_extent);
  const values shifted = [&] {
    values v = host;
    for (double& e : v) {
      e += third;
    }
    return v;
  }();
  const auto x = upload<T>(res, {small_rows, small_cols}, shifted);
  const auto d = upload<T>(res, {small_rows, small_rows},
                           distances<T>(shifted, {small_rows, small_cols}));
  const std::vector<T> x_before = x->template read<T>();
  const std::vector<T> d_before = d->template read<T>();
  for (const int32_t metric : {CUML_CLUSTER_DBSCAN_METRIC_EUCLIDEAN,
                               CUML_CLUSTER_DBSCAN_METRIC_COSINE}) {
    const labelling found = fitted(res, {.x = &x->view,
                                         .eps = cloud_eps,
                                         .metric = metric,
                                         .max_mbytes_per_batch = 1});
    EXPECT_EQ(found.status, "accepted");
    EXPECT_EQ(x->template read<T>(), x_before) << metric;
  }
  const labelling found =
      fitted(res, {.x = &d->view,
                   .eps = cloud_eps,
                   .metric = CUML_CLUSTER_DBSCAN_METRIC_PRECOMPUTED});
  EXPECT_EQ(found.status, "accepted");
  EXPECT_EQ(d->template read<T>(), d_before);
}

TEST(Dbscan, LeavesXUnchanged) {
  CUMLRKT_REQUIRE_GPU();
  leaves_x_unchanged<float>();
  leaves_x_unchanged<double>();
}

template <typename T>
void int64_matches_int32() {
  SCOPED_TRACE(name<T>());
  const resources res;
  const auto x = upload<T>(res, {cloud_rows, cloud_cols},
                           cloud({cloud_rows, cloud_cols}, cloud_extent));
  const dbscan_call narrow{
      .x = &x->view, .eps = cloud_eps, .min_samples = grid_min_samples};
  dbscan_call wide = narrow;
  wide.out_dtype = RR_DTYPE_INT64;
  const labelling a = fitted(res, narrow);
  const labelling b = fitted(res, wide);
  EXPECT_EQ(a.status, "accepted");
  EXPECT_EQ(b.status, "accepted");
  EXPECT_EQ(a.labels, b.labels);
  EXPECT_EQ(a.core, b.core);
  EXPECT_FALSE(a.core.empty());
}

TEST(Dbscan, Int64MatchesInt32) {
  CUMLRKT_REQUIRE_GPU();
  int64_matches_int32<float>();
  int64_matches_int32<double>();
}

template <typename T>
void precomputed_partial_batch() {
  SCOPED_TRACE(name<T>());
  const resources res;
  const values host = cloud({cloud_rows, cloud_cols}, cloud_extent);
  const auto x = upload<T>(res, {cloud_rows, cloud_cols}, host);
  const auto d = upload<T>(res, {cloud_rows, cloud_rows},
                           distances<T>(host, {cloud_rows, cloud_cols}));
  for (const int64_t mbytes : {int64_t{0}, int64_t{1}}) {
    const labelling direct = fitted(res, {.x = &x->view,
                                          .eps = cloud_eps,
                                          .min_samples = grid_min_samples,
                                          .max_mbytes_per_batch = mbytes});
    const labelling precomputed =
        fitted(res, {.x = &d->view,
                     .eps = cloud_eps,
                     .min_samples = grid_min_samples,
                     .metric = CUML_CLUSTER_DBSCAN_METRIC_PRECOMPUTED,
                     .max_mbytes_per_batch = mbytes});
    EXPECT_EQ(direct.status, "accepted");
    EXPECT_EQ(precomputed.status, "accepted");
    EXPECT_EQ(precomputed.labels, direct.labels) << mbytes;
    EXPECT_EQ(precomputed.core, direct.core) << mbytes;
  }
}

TEST(Dbscan, PrecomputedPartialBatch) {
  CUMLRKT_REQUIRE_GPU();
  precomputed_partial_batch<float>();
  precomputed_partial_batch<double>();
}

TEST(Dbscan, CosineWiderThanTall) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  values host;
  for (const double scale : {1.0, wide_scale}) {
    for (int64_t j = 1; j <= wide_cols; ++j) {
      host.push_back(scale * static_cast<double>(j));
    }
  }
  const auto x = upload<float>(res, {2, wide_cols}, host);
  const labelling found =
      fitted(res, {.x = &x->view,
                   .eps = wide_eps,
                   .min_samples = 2,
                   .metric = CUML_CLUSTER_DBSCAN_METRIC_COSINE});
  EXPECT_EQ(found.status, "accepted");
  EXPECT_EQ(found.labels, (values{0, 0}));
  EXPECT_EQ(found.core, (values{0, 1}));
}

template <typename T>
bool check_cosine_eps(double eps) {
  const T target = cluster::cosine_target<T>(eps);
  const T e = cluster::cosine_eps<T>(eps);
  const T prev = std::nextafter(e, T{0});
  const T inf = std::numeric_limits<T>::infinity();
  const T one_step = std::nextafter(target, inf);
  EXPECT_GE(e * e, target) << "eps " << eps;
  EXPECT_LT(prev * prev, target) << "eps " << eps;
  EXPECT_LE(e * e, std::nextafter(one_step, inf)) << "eps " << eps;
  return e * e > one_step;
}

TEST(Dbscan, CosineEpsIsTheSmallestWhoseSquareReachesTwiceEps) {
  std::vector<double> sweep{tiny_eps, small_eps, wide_cosine_eps,
                            std::numeric_limits<double>::max(),
                            float_second_step_eps};
  for (int k = 1; k <= sweep_steps; ++k) {
    sweep.push_back(k * sweep_step);
  }
  const std::vector<double> double_only{
      std::numeric_limits<double>::denorm_min(), subnormal_eps,
      deep_subnormal_eps, std::numeric_limits<double>::min() / 2};
  const std::vector<double> float_only{float_min_eps, float_subnormal_eps};
  int float_second_steps = 0;
  for (const double eps : sweep) {
    float_second_steps += check_cosine_eps<float>(eps) ? 1 : 0;
    check_cosine_eps<double>(eps);
  }
  for (const double eps : double_only) {
    check_cosine_eps<double>(eps);
  }
  for (const double eps : float_only) {
    check_cosine_eps<float>(eps);
  }
  EXPECT_GE(float_second_steps, 1);
}

}  // namespace
}  // namespace cumlrkt::test
