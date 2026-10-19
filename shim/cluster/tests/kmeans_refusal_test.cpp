#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cluster_fixture.hpp"
#include "cumlrkt/cluster.h"
#include "kmeans_fixture.hpp"
#include "raftrkt/c_api.h"
#include "refusal_fixture.hpp"

namespace cumlrkt::test {
namespace {

constexpr int32_t k = 3;
constexpr int64_t wide = 3;
constexpr int64_t forged_square = 65536;
constexpr int64_t forged_rows = 100000;
constexpr int64_t forged_clusters = 50000;
constexpr std::size_t pool = 64;
constexpr int32_t unknown_method = -1;
constexpr int32_t too_many = 11;

template <typename T>
struct kmeans_case {
  explicit kmeans_case(const resources& res)
      : data(res.r, float_dtype<T>, {pool}),
        other(res.r, float_dtype<other_t<T>>, {pool}),
        ints(res.r, RR_DTYPE_INT32, {pool}),
        outs(res.r, float_dtype<T>, {pool}),
        x(shaped(data, {rows, cols})),
        w(shaped(data, {rows})),
        init(shaped(outs, {k, cols})),
        centroids(shaped(outs, {k, cols})) {
    data.write(std::vector<T>(pool, 0));
    outs.write(std::vector<T>(pool, 0));
  }

  [[nodiscard]] fit_call call(record& out) const {
    return {.x = &x,
            .out = &out.r,
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
};

struct status_check {
  std::string got;
  std::string want;
};

void expect_statuses(const std::vector<status_check>& checks) {
  for (const status_check& c : checks) {
    EXPECT_EQ(c.got, c.want);
  }
}

template <typename T>
void fit_refusals() {
  SCOPED_TRACE(name<T>());
  const resources res;
  const kmeans_case<T> r(res);
  record out(res);
  const std::string t = name<T>();
  const std::string u = name<other_t<T>>();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const int32_t pp = CUML_CLUSTER_KMEANS_INIT_KMEANS_PLUS_PLUS;
  const int32_t array = CUML_CLUSTER_KMEANS_INIT_ARRAY;
  const rr_view* const x = &r.x;
  rr_deposit* const o = &out.r;
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
  const rr_view x_elsewhere = on_device(r.x, 1);
  const std::string tol_zero = "tol: expected greater than 0, got 0";
  const std::string tol_nan = "tol: expected greater than 0, got nan";
  const std::vector<refusal<fit_call>> cases{
      {{.x = &x_int, .out = o, .n_clusters = k, .init_method = pp},
       "X: expected float32 or float64, got int32"},
      {{.x = &x_rank1, .out = o, .n_clusters = k, .init_method = pp},
       "X: expected rank 2, got rank 1"},
      {{.x = &x_col, .out = o, .n_clusters = k, .init_method = pp},
       "X: expected row-major, got col-major 10x2"},
      {{.x = &x_huge, .out = o, .n_clusters = k, .init_method = pp},
       "X: 65536x65536 elements do not fit the index type"},
      {{.x = &x_no_cols, .out = o, .n_clusters = k, .init_method = pp},
       "X: expected at least 1 column, got 0"},
      {{.x = x, .out = o, .n_clusters = 0, .init_method = pp},
       "n-clusters: expected at least 1, got 0"},
      {{.x = x, .out = o, .n_clusters = too_many, .init_method = pp},
       "X: 10 samples are fewer than the 11 clusters"},
      {{.x = x,
        .sample_weight = &w_other,
        .out = o,
        .n_clusters = k,
        .init_method = pp},
       "sample-weight: expected " + t + ", got " + u},
      {{.x = x,
        .sample_weight = &w_short,
        .out = o,
        .n_clusters = k,
        .init_method = pp},
       "sample-weight: expected 10 elements, got 9"},
      {{.x = x, .out = o, .n_clusters = k, .init_method = unknown_method},
       "init: expected scalable-k-means++, k-means++, random or a centroids "
       "matrix"},
      {{.x = x, .out = o, .n_clusters = k, .init_method = array},
       "init: an array init needs a centroids matrix"},
      {{.x = x, .init = &r.init, .out = o, .n_clusters = k, .init_method = pp},
       "init: a centroids matrix needs the array init"},
      {{.x = x,
        .init = &init_other,
        .out = o,
        .n_clusters = k,
        .init_method = array},
       "init: expected " + t + ", got " + u},
      {{.x = x,
        .init = &init_short,
        .out = o,
        .n_clusters = k,
        .init_method = array},
       "init: expected 3 rows, got 2"},
      {{.x = x,
        .init = &init_wide,
        .out = o,
        .n_clusters = k,
        .init_method = array},
       "init: expected 2 columns, got 3"},
      {{.x = x,
        .init = &init_col,
        .out = o,
        .n_clusters = k,
        .init_method = array},
       "init: expected row-major, got col-major 3x2"},
      {{.x = x, .out = o, .n_clusters = k, .init_method = pp, .max_iter = -1},
       "max-iter: expected at least 0, got -1"},
      {{.x = x, .out = o, .n_clusters = k, .init_method = pp, .n_init = 0},
       "n-init: expected at least 1, got 0"},
      {{.x = x,
        .out = o,
        .n_clusters = k,
        .init_method = pp,
        .batch_samples = -1},
       "max-samples-per-batch: expected at least 0, got -1"},
      {{.x = x, .out = nullptr, .n_clusters = k, .init_method = pp},
       "out is NULL"},
      {{.x = &x_elsewhere, .out = o, .n_clusters = k, .init_method = pp},
       "X: on device 1, but the resources are on device 0"},
      {{.x = x, .out = o, .n_clusters = k, .init_method = pp, .tol = 0},
       tol_zero},
      {{.x = x, .out = o, .n_clusters = k, .init_method = pp, .tol = nan},
       tol_nan},
      {{.x = x, .out = o, .n_clusters = k, .init_method = pp, .tol = -nan},
       tol_nan},
      {{.x = x, .out = o, .n_clusters = k, .init_method = pp, .tol = -0.0},
       tol_zero},
      {{.x = x,
        .out = o,
        .n_clusters = k,
        .init_method = pp,
        .oversampling_factor = -1},
       "oversampling-factor: expected at least 0, got -1"}};
  for (const refusal<fit_call>& c : cases) {
    const fit_outcome got = fit(res, c.call);
    EXPECT_EQ(got.status, c.message);
    EXPECT_EQ(cuml_cluster_last_error_kind(), RR_ERROR_LOGIC) << got.status;
    EXPECT_EQ(got.inertia, 0) << got.status;
    EXPECT_EQ(got.n_iter, 0) << got.status;
    expect_untouched(out);
  }
}

template <typename T>
void filled_out_refusal() {
  SCOPED_TRACE(name<T>());
  const resources res;
  const kmeans_case<T> r(res);
  record filled(res);
  ASSERT_EQ(fit(res, r.call(filled)).status, "accepted");
  const void* owner = filled.r.owner;
  EXPECT_EQ(fit(res, r.call(filled)).status, "out: already filled");
  EXPECT_EQ(filled.r.owner, owner);
  EXPECT_EQ(filled.r.count, 2);
}

template <typename T>
void apply_table_refusals() {
  SCOPED_TRACE(name<T>());
  const resources res;
  const kmeans_case<T> r(res);
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
  record out(res);
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
    EXPECT_EQ(predict(res, row.centroids, row.x, &out.r), row.message);
    EXPECT_EQ(cuml_cluster_last_error_kind(), RR_ERROR_LOGIC);
    EXPECT_EQ(score(res, row.centroids, row.x, nullptr), row.message);
    EXPECT_EQ(transform(res, row.centroids, row.x, &out.r), row.message);
    expect_untouched(out);
  }
}

template <typename T>
void apply_argument_refusals() {
  SCOPED_TRACE(name<T>());
  const resources res;
  const kmeans_case<T> r(res);
  const std::string t = name<T>();
  const std::string u = name<other_t<T>>();
  const rr_view c_int = shaped(r.ints, {k, cols});
  const rr_view w_other = shaped(r.other, {rows});
  const rr_view w_short = shaped(r.data, {rows - 1});
  const rr_view many_x = shaped(r.data, {forged_rows, cols});
  const rr_view many_c = shaped(r.outs, {forged_clusters, cols});
  const rr_view* const c = &r.centroids;
  const rr_view* const x = &r.x;
  record out(res);
  const std::string batch =
      "max-samples-per-batch: expected at least 0, got -1";
  expect_statuses({
      {score(res, c, x, &w_other),
       "sample-weight: expected " + t + ", got " + u},
      {score(res, c, x, &w_short),
       "sample-weight: expected 10 elements, got 9"},
      {predict(res, c, x, &out.r, -1), batch},
      {score(res, c, x, nullptr, -1), batch},
      {transform(res, c, x, &out.r, -1), batch},
      {predict(res, c, x, nullptr), "out is NULL"},
      {transform(res, c, x, nullptr), "out is NULL"},
      {transform(res, &many_c, &many_x, &out.r),
       "distances: 100000x50000 elements do not fit the index type"},
  });
  expect_untouched(out);
  double inertia = -1;
  EXPECT_NE(cuml_cluster_kmeans_score(res.handle, &c_int, x, nullptr,
                                      default_batch, &inertia),
            RR_OK);
  EXPECT_EQ(inertia, 0);
}

template <typename T>
void first_in_order_wins() {
  SCOPED_TRACE(name<T>());
  const resources res;
  const kmeans_case<T> r(res);
  const rr_view x_int = shaped(r.ints, {rows, cols});
  const rr_view x_rank1 = shaped(r.data, {rows});
  const rr_view int_elsewhere = on_device(x_int, 1);
  const rr_view w_short = shaped(r.data, {rows - 1});
  const rr_view c_int = shaped(r.ints, {k, cols});
  const rr_view x_other = shaped(r.other, {rows, cols});
  const int32_t pp = CUML_CLUSTER_KMEANS_INIT_KMEANS_PLUS_PLUS;
  const rr_view* const x = &r.x;
  record out(res);
  rr_deposit* const o = &out.r;
  expect_statuses({
      {fit(res, {.x = &x_int, .out = o, .n_clusters = 0, .init_method = pp})
           .status,
       "X: expected float32 or float64, got int32"},
      {fit(res,
           {.x = &int_elsewhere, .out = o, .n_clusters = 0, .init_method = pp})
           .status,
       "X: on device 1, but the resources are on device 0"},
      {fit(res,
           {.x = &x_rank1, .out = o, .n_clusters = too_many, .init_method = pp})
           .status,
       "X: expected rank 2, got rank 1"},
      {fit(res, {.x = x,
                 .sample_weight = &w_short,
                 .out = o,
                 .n_clusters = 0,
                 .init_method = pp})
           .status,
       "n-clusters: expected at least 1, got 0"},
      {fit(res, {.x = x,
                 .sample_weight = &w_short,
                 .out = o,
                 .n_clusters = k,
                 .init_method = unknown_method})
           .status,
       "sample-weight: expected 10 elements, got 9"},
      {fit(res, {.x = x,
                 .out = o,
                 .n_clusters = k,
                 .init_method = pp,
                 .max_iter = -1,
                 .n_init = 0})
           .status,
       "max-iter: expected at least 0, got -1"},
      {predict(res, &c_int, &x_other, o),
       "centroids: expected float32 or float64, got int32"},
  });
}

template <typename T>
void refused_parameters_allocate_nothing() {
  SCOPED_TRACE(name<T>());
  const resources res;
  const kmeans_case<T> r(res);
  record accepted(res);
  ASSERT_EQ(fit(res, r.call(accepted)).status, "accepted");
  record out(res);
  fit_call tol = r.call(out);
  tol.tol = 0;
  fit_call oversampling = r.call(out);
  oversampling.oversampling_factor = -1;
  const rr_view many_x = shaped(r.data, {forged_rows, cols});
  const rr_view many_c = shaped(r.outs, {forged_clusters, cols});
  const allocation_meter meter;
  EXPECT_EQ(fit(res, tol).status, "tol: expected greater than 0, got 0");
  EXPECT_EQ(fit(res, oversampling).status,
            "oversampling-factor: expected at least 0, got -1");
  EXPECT_EQ(transform(res, &many_c, &many_x, &out.r),
            "distances: 100000x50000 elements do not fit the index type");
  EXPECT_EQ(meter.peak(), 0);
  expect_untouched(out);
}

TEST(Refusals, NameTheArgument) {
  CUMLRKT_REQUIRE_GPU();
  fit_refusals<float>();
  fit_refusals<double>();
  filled_out_refusal<float>();
  filled_out_refusal<double>();
  apply_table_refusals<float>();
  apply_table_refusals<double>();
  apply_argument_refusals<float>();
  apply_argument_refusals<double>();
}

TEST(Refusals, RefusedParametersAllocateNothing) {
  CUMLRKT_REQUIRE_GPU();
  refused_parameters_allocate_nothing<float>();
  refused_parameters_allocate_nothing<double>();
}

TEST(Refusals, FirstInOrderWins) {
  CUMLRKT_REQUIRE_GPU();
  first_in_order_wins<float>();
  first_in_order_wins<double>();
}

}  // namespace
}  // namespace cumlrkt::test
