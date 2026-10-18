#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <type_traits>
#include <vector>

#include "cluster_fixture.hpp"
#include "cumlrkt/cluster.h"
#include "kmeans_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {
namespace {

constexpr int64_t rows = 10;
constexpr int64_t cols = 2;
constexpr int32_t k = 3;
constexpr int64_t wide = 3;
constexpr int64_t forged_square = 65536;
constexpr int64_t forged_rows = 100000;
constexpr int64_t forged_clusters = 50000;
constexpr std::size_t pool = 64;
constexpr int32_t unknown_method = -1;
constexpr int32_t too_many = 11;

template <typename T>
using other_t = std::conditional_t<std::is_same_v<T, float>, double, float>;

template <typename T>
std::string name() {
  return std::is_same_v<T, float> ? "float32" : "float64";
}

rr_view shaped(const device_array& a, std::initializer_list<int64_t> shape,
               bool col_major = false) {
  rr_view v = a.view;
  v.rank = static_cast<int32_t>(shape.size());
  int32_t i = 0;
  for (const int64_t extent : shape) {
    v.shape[i++] = extent;
  }
  if (v.rank == 1) {
    v.strides[0] = 1;
  } else if (col_major) {
    v.strides[0] = 1;
    v.strides[1] = v.shape[0];
  } else {
    v.strides[0] = v.shape[1];
    v.strides[1] = 1;
  }
  return v;
}

rr_view on_device(rr_view v, int32_t device) {
  v.device = device;
  return v;
}

bool cuvs_refused(const std::string& message, const std::string& parameter) {
  return message.rfind("RAFT failure at file=", 0) == 0 &&
         message.find("invalid parameter (" + parameter + ")") !=
             std::string::npos;
}

template <typename T>
struct refusal_case {
  explicit refusal_case(const resources& res)
      : data(res.r, float_dtype<T>, {pool}),
        other(res.r, float_dtype<other_t<T>>, {pool}),
        ints(res.r, RR_DTYPE_INT32, {pool}),
        outs(res.r, float_dtype<T>, {pool}),
        x(shaped(data, {rows, cols})),
        w(shaped(data, {rows})),
        init(shaped(outs, {k, cols})),
        centroids(shaped(outs, {k, cols})),
        labels(shaped(ints, {rows})),
        distances(shaped(outs, {rows, k})) {
    data.write(std::vector<T>(pool, 0));
    outs.write(std::vector<T>(pool, 0));
  }

  [[nodiscard]] fit_call call() const {
    return {.x = &x,
            .centroids = &centroids,
            .labels = &labels,
            .n_clusters = k,
            .init_method = CUML_CLUSTER_KMEANS_INIT_KMEANS_PLUS_PLUS};
  }

  device_array data;
  device_array other;
  device_array ints;
  device_array outs;
  rr_view x;
  rr_view w;
  rr_view init;
  rr_view centroids;
  rr_view labels;
  rr_view distances;
};

// NOLINTBEGIN(readability-function-cognitive-complexity) -- one EXPECT per
// refusal row, each a branch to clang-tidy
template <typename T>
void fit_refusals() {
  SCOPED_TRACE(name<T>());
  const resources res;
  const refusal_case<T> r(res);
  const auto refused = [&](fit_call c) {
    const fit_outcome o = fit(res, c);
    EXPECT_EQ(cuml_cluster_last_error_kind(), RR_ERROR_LOGIC) << o.status;
    EXPECT_EQ(o.inertia, 0);
    EXPECT_EQ(o.n_iter, 0);
    return o.status;
  };
  const std::string t = name<T>();
  const std::string u = name<other_t<T>>();
  const rr_view x_int = shaped(r.ints, {rows, cols});
  const rr_view x_rank1 = shaped(r.data, {rows});
  const rr_view x_col = shaped(r.data, {rows, cols}, true);
  const rr_view x_huge = shaped(r.data, {forged_square, forged_square});
  const rr_view x_no_cols = shaped(r.data, {rows, 0});
  const rr_view w_other = shaped(r.other, {rows});
  const rr_view w_short = shaped(r.data, {rows - 1});
  const rr_view init_other = shaped(r.other, {k, cols});
  const rr_view init_short = shaped(r.outs, {k - 1, cols});
  const rr_view init_wide = shaped(r.outs, {k, wide});
  const rr_view init_col = shaped(r.outs, {k, cols}, true);
  const rr_view centroids_short = shaped(r.outs, {k - 1, cols});
  const rr_view labels_short = shaped(r.ints, {rows - 1});
  const rr_view x_elsewhere = on_device(r.x, 1);
  fit_call c = r.call();
  const fit_call base = c;
  const fit_call with_init = [&] {
    fit_call a = base;
    a.init_method = CUML_CLUSTER_KMEANS_INIT_ARRAY;
    a.init = &r.init;
    return a;
  }();

  c.x = &x_int;
  EXPECT_EQ(refused(c), "X: expected float32 or float64, got int32");
  c.x = &x_rank1;
  EXPECT_EQ(refused(c), "X: expected rank 2, got rank 1");
  c.x = &x_col;
  EXPECT_EQ(refused(c), "X: expected row-major, got col-major 10x2");
  c.x = &x_huge;
  EXPECT_EQ(refused(c), "X: 65536x65536 elements do not fit the index type");
  c.x = &x_no_cols;
  EXPECT_EQ(refused(c), "X: expected at least 1 column, got 0");
  c = base;
  c.n_clusters = 0;
  EXPECT_EQ(refused(c), "n-clusters: expected at least 1, got 0");
  c.n_clusters = too_many;
  EXPECT_EQ(refused(c), "X: 10 samples are fewer than the 11 clusters");
  c = base;
  c.sample_weight = &w_other;
  EXPECT_EQ(refused(c), "sample-weight: expected " + t + ", got " + u);
  c.sample_weight = &w_short;
  EXPECT_EQ(refused(c), "sample-weight: expected 10 elements, got 9");
  c = base;
  c.init_method = unknown_method;
  EXPECT_EQ(refused(c),
            "init: expected scalable-k-means++, k-means++, random or a "
            "centroids matrix");
  c.init_method = CUML_CLUSTER_KMEANS_INIT_ARRAY;
  EXPECT_EQ(refused(c), "init: an array init needs a centroids matrix");
  c = base;
  c.init = &r.init;
  EXPECT_EQ(refused(c), "init: a centroids matrix needs the array init");
  c = with_init;
  c.init = &init_other;
  EXPECT_EQ(refused(c), "init: expected " + t + ", got " + u);
  c.init = &init_short;
  EXPECT_EQ(refused(c), "init: expected 3 rows, got 2");
  c.init = &init_wide;
  EXPECT_EQ(refused(c), "init: expected 2 columns, got 3");
  c.init = &init_col;
  EXPECT_EQ(refused(c), "init: expected row-major, got col-major 3x2");
  c = base;
  c.max_iter = -1;
  EXPECT_EQ(refused(c), "max-iter: expected at least 0, got -1");
  c = base;
  c.n_init = 0;
  EXPECT_EQ(refused(c), "n-init: expected at least 1, got 0");
  c = base;
  c.batch_samples = -1;
  EXPECT_EQ(refused(c), "max-samples-per-batch: expected at least 0, got -1");
  c = base;
  c.centroids = &centroids_short;
  EXPECT_EQ(refused(c), "centroids: expected 3 rows, got 2");
  c.centroids = nullptr;
  EXPECT_EQ(refused(c), "centroids is NULL");
  c = base;
  c.labels = &labels_short;
  EXPECT_EQ(refused(c), "labels: expected 10 elements, got 9");
  c = base;
  c.x = &x_elsewhere;
  EXPECT_EQ(refused(c), "X: on device 1, but the resources are on device 0");
  c = base;
  c.tol = 0;
  EXPECT_TRUE(cuvs_refused(refused(c), "tol<=0"));
  c = base;
  c.oversampling_factor = -1;
  EXPECT_TRUE(cuvs_refused(refused(c), "oversampling_factor<0"));
}
// NOLINTEND(readability-function-cognitive-complexity)

// NOLINTBEGIN(readability-function-cognitive-complexity) -- one EXPECT per
// refusal row, each a branch to clang-tidy
template <typename T>
void apply_refusals() {
  SCOPED_TRACE(name<T>());
  const resources res;
  const refusal_case<T> r(res);
  const std::string t = name<T>();
  const std::string u = name<other_t<T>>();
  const rr_view c_int = shaped(r.ints, {k, cols});
  const rr_view c_rank1 = shaped(r.outs, {k});
  const rr_view c_col = shaped(r.outs, {k, cols}, true);
  const rr_view c_no_rows = shaped(r.outs, {0, cols});
  const rr_view c_no_cols = shaped(r.outs, {k, 0});
  const rr_view x_other = shaped(r.other, {rows, cols});
  const rr_view x_wide = shaped(r.data, {rows, wide});
  const rr_view x_col = shaped(r.data, {rows, cols}, true);
  const rr_view x_rank1 = shaped(r.data, {rows});
  const rr_view x_no_rows = shaped(r.data, {0, cols});
  const rr_view x_elsewhere = on_device(r.x, 1);
  const struct {
    const rr_view* centroids;
    const rr_view* x;
    std::string message;
  } rows_of_table[] = {
      {&c_int, &r.x, "centroids: expected float32 or float64, got int32"},
      {&c_rank1, &r.x, "centroids: expected rank 2, got rank 1"},
      {&c_col, &r.x, "centroids: expected row-major, got col-major 3x2"},
      {&c_no_rows, &r.x, "centroids: expected at least 1 row, got 0"},
      {&c_no_cols, &r.x, "centroids: expected at least 1 column, got 0"},
      {&r.centroids, &x_other, "X: expected " + t + ", got " + u},
      {&r.centroids, &x_wide, "X: expected 2 columns, got 3"},
      {&r.centroids, &x_col, "X: expected row-major, got col-major 10x2"},
      {&r.centroids, &x_rank1, "X: expected rank 2, got rank 1"},
      {&r.centroids, &x_no_rows, "X: expected at least 1 row, got 0"},
      {&r.centroids, &x_elsewhere,
       "X: on device 1, but the resources are on device 0"},
  };
  for (const auto& row : rows_of_table) {
    EXPECT_EQ(predict(res, row.centroids, row.x, nullptr, &r.labels),
              row.message);
    EXPECT_EQ(predict(res, row.centroids, row.x, nullptr, nullptr),
              row.message);
    EXPECT_EQ(cuml_cluster_last_error_kind(), RR_ERROR_LOGIC);
    EXPECT_EQ(transform(res, row.centroids, row.x, &r.distances), row.message);
  }
  const rr_view w_other = shaped(r.other, {rows});
  const rr_view w_short = shaped(r.data, {rows - 1});
  EXPECT_EQ(predict(res, &r.centroids, &r.x, &w_other, nullptr),
            "sample-weight: expected " + t + ", got " + u);
  EXPECT_EQ(predict(res, &r.centroids, &r.x, &w_short, nullptr),
            "sample-weight: expected 10 elements, got 9");
  const std::string batch =
      "max-samples-per-batch: expected at least 0, got -1";
  EXPECT_EQ(predict(res, &r.centroids, &r.x, nullptr, &r.labels, -1), batch);
  EXPECT_EQ(transform(res, &r.centroids, &r.x, &r.distances, -1), batch);
  const rr_view labels_short = shaped(r.ints, {rows - 1});
  const rr_view distances_short = shaped(r.outs, {rows - 1, k});
  EXPECT_EQ(predict(res, &r.centroids, &r.x, nullptr, &labels_short),
            "labels: expected 10 elements, got 9");
  EXPECT_EQ(transform(res, &r.centroids, &r.x, &distances_short),
            "distances: expected 10 rows, got 9");
  const rr_view many_x = shaped(r.data, {forged_rows, cols});
  const rr_view many_c = shaped(r.outs, {forged_clusters, cols});
  const rr_view huge = shaped(r.outs, {forged_rows, forged_clusters});
  EXPECT_EQ(transform(res, &many_c, &many_x, &huge),
            "distances: 100000x50000 elements do not fit the index type");
  double inertia = -1;
  EXPECT_NE(cuml_cluster_kmeans_predict(res.handle, &c_int, &r.x, nullptr,
                                        nullptr, default_batch, &inertia),
            RR_OK);
  EXPECT_EQ(inertia, 0);
}
// NOLINTEND(readability-function-cognitive-complexity)

// NOLINTBEGIN(readability-function-cognitive-complexity) -- one EXPECT per
// refusal row, each a branch to clang-tidy
template <typename T>
void first_in_order_wins() {
  SCOPED_TRACE(name<T>());
  const resources res;
  const refusal_case<T> r(res);
  const rr_view x_int = shaped(r.ints, {rows, cols});
  const rr_view x_rank1 = shaped(r.data, {rows});
  const rr_view int_elsewhere = on_device(x_int, 1);
  const rr_view w_short = shaped(r.data, {rows - 1});
  const rr_view c_int = shaped(r.ints, {k, cols});
  const rr_view x_other = shaped(r.other, {rows, cols});
  fit_call c = r.call();
  c.x = &x_int;
  c.n_clusters = 0;
  EXPECT_EQ(fit(res, c).status, "X: expected float32 or float64, got int32");
  c.x = &int_elsewhere;
  EXPECT_EQ(fit(res, c).status,
            "X: on device 1, but the resources are on device 0");
  c = r.call();
  c.x = &x_rank1;
  c.n_clusters = too_many;
  EXPECT_EQ(fit(res, c).status, "X: expected rank 2, got rank 1");
  c = r.call();
  c.n_clusters = 0;
  c.sample_weight = &w_short;
  EXPECT_EQ(fit(res, c).status, "n-clusters: expected at least 1, got 0");
  c = r.call();
  c.sample_weight = &w_short;
  c.init_method = unknown_method;
  EXPECT_EQ(fit(res, c).status, "sample-weight: expected 10 elements, got 9");
  c = r.call();
  c.max_iter = -1;
  c.n_init = 0;
  c.centroids = nullptr;
  EXPECT_EQ(fit(res, c).status, "max-iter: expected at least 0, got -1");
  EXPECT_EQ(predict(res, &c_int, &x_other, nullptr, nullptr),
            "centroids: expected float32 or float64, got int32");
}
// NOLINTEND(readability-function-cognitive-complexity)

TEST(Refusals, NameTheArgument) {
  CUMLRKT_REQUIRE_GPU();
  fit_refusals<float>();
  fit_refusals<double>();
  apply_refusals<float>();
  apply_refusals<double>();
}

TEST(Refusals, FirstInOrderWins) {
  CUMLRKT_REQUIRE_GPU();
  first_in_order_wins<float>();
  first_in_order_wins<double>();
}

}  // namespace
}  // namespace cumlrkt::test
