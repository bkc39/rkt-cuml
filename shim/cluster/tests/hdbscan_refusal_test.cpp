#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "cluster_fixture.hpp"
#include "cumlrkt/cluster.h"
#include "hdbscan_fixture.hpp"
#include "kmeans_fixture.hpp"
#include "raftrkt/c_api.h"
#include "refusal_fixture.hpp"

namespace cumlrkt::test {
namespace {

constexpr int64_t most_rows = 1073741824;
constexpr int64_t most_min_samples = 1023;
constexpr int64_t short_rows = 9;
constexpr int64_t index_rows = 1073741824;
constexpr int64_t most_index = 2147483647;
constexpr int64_t pair_count = 32769;
constexpr double pair_spacing = 10;
constexpr int64_t pair_side = 182;
constexpr double past_float = 1e39;

void expect_refused(const allocation_meter& meter,
                    const std::vector<const record*>& outs,
                    const std::string& status, const std::string& message,
                    int64_t most_bytes = 0) {
  EXPECT_EQ(status, message);
  EXPECT_EQ(cuml_cluster_last_error_kind(), RR_ERROR_LOGIC) << status;
  EXPECT_LE(meter.peak(), most_bytes) << status;
  EXPECT_EQ(meter.current(), 0) << status;
  for (const record* out : outs) {
    expect_untouched(*out);
  }
}

TEST(HdbscanRefusals, FitNamesTheArgument) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const refusal_case<float> r(res, rows, cols);
  record model(res);
  record out(res);
  const allocation_meter meter;
  const rr_view* const x = &r.x;
  rr_deposit* const m = &model.r;
  rr_deposit* const o = &out.r;
  const rr_view x_double = shaped(r.other, {rows, cols});
  const rr_view x_int = shaped(r.ints, {rows, cols});
  const rr_view x_rank1 = shaped(r.data, {rows});
  const rr_view x_col = shaped(r.data, {rows, cols}, true);
  const rr_view x_one = shaped(r.data, {1, cols});
  const rr_view x_one_double = shaped(r.other, {1, cols});
  const rr_view x_no_rows = shaped(r.data, {0, cols});
  const rr_view x_no_cols = shaped(r.data, {rows, 0});
  const rr_view x_wide = shaped(r.data, {wide_rows, wide_cols});
  const rr_view x_tall = shaped(r.data, {most_rows + 1, 1});
  const rr_view x_elsewhere = on_device(r.x, 1);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const std::string eps_most =
      "cluster-selection-epsilon: expected at most 3.4028234663852886e+38, "
      "got ";
  const std::string alpha_most =
      "alpha: expected at most 3.4028234663852886e+38, got ";
  const std::vector<refusal<hdbscan_call>> cases{
      {{.x = &x_double, .model_out = m, .out = o},
       "X: expected float32, got float64"},
      {{.x = &x_int, .model_out = m, .out = o},
       "X: expected float32, got int32"},
      {{.x = &x_elsewhere, .model_out = m, .out = o},
       "X: on device 1, but the resources are on device 0"},
      {{.x = x, .model_out = nullptr, .out = o}, "model-out is NULL"},
      {{.x = x, .model_out = m, .out = nullptr}, "out is NULL"},
      {{.x = &x_rank1, .model_out = m, .out = o},
       "X: expected rank 2, got rank 1"},
      {{.x = &x_col, .model_out = m, .out = o},
       "X: expected row-major, got col-major 10x2"},
      {{.x = &x_one, .model_out = m, .out = o},
       "X: expected at least 2 rows, got 1"},
      {{.x = &x_no_rows, .model_out = m, .out = o},
       "X: expected at least 2 rows, got 0"},
      {{.x = &x_no_cols, .model_out = m, .out = o},
       "X: expected at least 1 column, got 0"},
      {{.x = &x_wide, .model_out = m, .out = o},
       "X: 65536x32769 elements do not fit the index type"},
      {{.x = &x_tall, .model_out = m, .out = o},
       "X: expected at most 1073741824 rows, got 1073741825"},
      {{.x = x, .model_out = m, .out = o, .min_cluster_size = 1},
       "min-cluster-size: expected at least 2, got 1"},
      {{.x = x, .model_out = m, .out = o, .min_cluster_size = 0},
       "min-cluster-size: expected at least 2, got 0"},
      {{.x = x, .model_out = m, .out = o, .min_cluster_size = -3},
       "min-cluster-size: expected at least 2, got -3"},
      {{.x = x, .model_out = m, .out = o, .min_samples = 0},
       "min-samples: expected at least 1, got 0"},
      {{.x = x, .model_out = m, .out = o, .min_samples = most_min_samples + 1},
       "min-samples: expected at most 1023, got 1024"},
      {{.x = x, .model_out = m, .out = o, .min_samples = rows + 1},
       "X: 10 samples are fewer than min-samples 11"},
      {{.x = x, .model_out = m, .out = o, .cluster_selection_epsilon = -1},
       "cluster-selection-epsilon: expected at least 0, got -1"},
      {{.x = x, .model_out = m, .out = o, .cluster_selection_epsilon = nan},
       "cluster-selection-epsilon: expected at least 0, got nan"},
      {{.x = x, .model_out = m, .out = o, .cluster_selection_epsilon = inf},
       eps_most + "inf"},
      {{.x = x,
        .model_out = m,
        .out = o,
        .cluster_selection_epsilon = past_float},
       eps_most + "1e+39"},
      {{.x = x, .model_out = m, .out = o, .max_cluster_size = -1},
       "max-cluster-size: expected at least 0, got -1"},
      {{.x = x, .model_out = m, .out = o, .alpha = 0},
       "alpha: expected greater than 0, got 0"},
      {{.x = x, .model_out = m, .out = o, .alpha = -1},
       "alpha: expected greater than 0, got -1"},
      {{.x = x, .model_out = m, .out = o, .alpha = nan},
       "alpha: expected greater than 0, got nan"},
      {{.x = x, .model_out = m, .out = o, .alpha = inf}, alpha_most + "inf"},
      {{.x = x, .model_out = m, .out = o, .cluster_selection_method = 2},
       "cluster-selection-method: expected eom or leaf"},
      {{.x = x, .model_out = m, .out = o, .cluster_selection_method = -1},
       "cluster-selection-method: expected eom or leaf"},
      {{.x = &x_one_double, .model_out = m, .out = o},
       "X: expected float32, got float64"},
      {{.x = x,
        .model_out = m,
        .out = o,
        .min_cluster_size = 1,
        .min_samples = 0},
       "min-cluster-size: expected at least 2, got 1"},
      {{.x = &x_tall, .model_out = m, .out = o, .min_cluster_size = 1},
       "X: expected at most 1073741824 rows, got 1073741825"},
      {{.x = x,
        .model_out = m,
        .out = o,
        .min_samples = rows + 1,
        .cluster_selection_epsilon = -1},
       "X: 10 samples are fewer than min-samples 11"},
      {{.x = x,
        .model_out = m,
        .out = o,
        .cluster_selection_epsilon = -1,
        .max_cluster_size = -1},
       "cluster-selection-epsilon: expected at least 0, got -1"},
      {{.x = x,
        .model_out = m,
        .out = o,
        .max_cluster_size = -1,
        .alpha = 0},
       "max-cluster-size: expected at least 0, got -1"},
      {{.x = x,
        .model_out = m,
        .out = o,
        .alpha = 0,
        .cluster_selection_method = 2},
       "alpha: expected greater than 0, got 0"}};
  for (const refusal<hdbscan_call>& c : cases) {
    expect_refused(meter, {&model, &out}, hdbscan_fit(res, c.call), c.message);
  }
}

struct norm_case {
  std::vector<double> row;
  std::string got;
};

TEST(HdbscanRefusals, LargeFitRowsAreRefusedBeforeAnyOutput) {
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
    SCOPED_TRACE(c.got);
    values host = with_row(c.row, bad_row, rows, cols);
    host[later_bad_row * cols] = nan;
    const auto x = upload<float>(res, {rows, cols}, host);
    record model(res);
    record out(res);
    const allocation_meter meter;
    expect_refused(meter, {&model, &out},
                   hdbscan_fit(res, {.x = &x->view,
                                     .model_out = &model.r,
                                     .out = &out.r}),
                   norm_refusal(bad_row, c.got), scratch_bytes);
  }
  values host = with_row({nan, 0}, bad_row, rows, cols);
  const auto x = upload<float>(res, {rows, cols}, host);
  record model(res);
  record out(res);
  const allocation_meter meter;
  expect_refused(meter, {&model, &out},
                 hdbscan_fit(res, {.x = &x->view,
                                   .model_out = &model.r,
                                   .out = &out.r,
                                   .min_samples = rows + 1}),
                 "X: 10 samples are fewer than min-samples 11");
}

TEST(HdbscanRefusals, BoundaryFitRowsFit) {
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
  record model(res);
  record out(res);
  EXPECT_EQ(
      hdbscan_fit(res, {.x = &x->view, .model_out = &model.r, .out = &out.r}),
      "accepted");
  EXPECT_EQ(model.r.count, model_regions);
  EXPECT_EQ(out.r.count, plain_regions);
}

TEST(HdbscanRefusals, PredictionNamesTheArgument) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const fitted_model f(res, three_blobs());
  ASSERT_EQ(f.status, "accepted");
  const refusal_case<float> r(res, 4, cols);
  const device_array foreign(res.r, RR_DTYPE_INT32, {1});
  const device_array floats(res.r, RR_DTYPE_FLOAT32, {1});
  record out(res);
  const allocation_meter meter;
  const rr_view* const model = f.anchor();
  const rr_view* const x = &f.x->view;
  const rr_view p = r.x;
  const rr_view p_double = shaped(r.other, {4, cols});
  const rr_view p_rank1 = shaped(r.data, {4});
  const rr_view p_col = shaped(r.data, {4, cols}, true);
  const rr_view p_empty = shaped(r.data, {0, cols});
  const rr_view p_three = shaped(r.data, {4, 3});
  const rr_view p_index = shaped(r.data, {index_rows, cols});
  const rr_view p_index_three = shaped(r.data, {index_rows, 3});
  const rr_view x_short = shaped(*f.x, {f.n - 1, cols});
  const rr_view x_double = shaped(r.other, {f.n, cols});
  const int64_t k = f.clusters();
  const int64_t entries_rows = ((most_index + 1) / k) + 1;
  const rr_view p_entries = shaped(r.data, {entries_rows, cols});
  const std::string entries_message =
      "points: " + std::to_string(entries_rows) + " rows with " +
      std::to_string(k) + " clusters need " +
      std::to_string(entries_rows * k) + " entries, more than 2147483647";
  struct prediction {
    const rr_view* model;
    const rr_view* x;
    const rr_view* points;
    int64_t batch_size = default_batch_size;
  };
  const std::vector<refusal<prediction>> shared{
      {{nullptr, x, &p}, "model is NULL"},
      {{&foreign.view, x, &p},
       "model: expected a model anchor, got another array"},
      {{&floats.view, x, &p},
       "model: expected a model anchor, got another array"},
      {{&foreign.view, x, &p_double},
       "model: expected a model anchor, got another array"},
      {{model, x, &p_double}, "points: expected float32, got float64"},
      {{model, x, &p_rank1}, "points: expected rank 2, got rank 1"},
      {{model, x, &p_col}, "points: expected row-major, got col-major 4x2"},
      {{model, x, &p_empty}, "points: expected at least 1 row, got 0"},
      {{model, x, &p_three},
       "points: expected 2 columns, as the model was fit on, got 3"},
      {{model, &x_short, &p},
       "X: expected 300x2, as the model was fit on, got 299x2"},
      {{model, &x_double, &p}, "X: expected float32, got float64"},
      {{model, x, &p_index},
       "points: 1073741824x2 elements do not fit the index type"},
      {{model, x, &p_index_three},
       "points: expected 2 columns, as the model was fit on, got 3"}};
  for (const refusal<prediction>& c : shared) {
    expect_refused(
        meter, {&out},
        approximate_predict(res, c.call.model, c.call.x, c.call.points, &out.r),
        c.message);
    expect_refused(meter, {&out},
                   membership_vector(res, c.call.model, c.call.x,
                                     c.call.points, &out.r),
                   c.message);
  }
  expect_refused(meter, {&out}, membership_vector(res, model, x, &p, &out.r, 0),
                 "batch-size: expected at least 1, got 0");
  expect_refused(meter, {&out},
                 membership_vector(res, model, x, &p, &out.r, -4),
                 "batch-size: expected at least 1, got -4");
  expect_refused(meter, {&out},
                 membership_vector(res, model, x, &p_entries, &out.r),
                 entries_message);
  expect_refused(meter, {&out},
                 approximate_predict(res, model, x, &p, nullptr),
                 "out is NULL");
  expect_refused(meter, {&out}, all_points(res, &foreign.view, x, &out.r),
                 "model: expected a model anchor, got another array");
  expect_refused(meter, {&out}, all_points(res, model, &x_short, &out.r),
                 "X: expected 300x2, as the model was fit on, got 299x2");
  expect_refused(meter, {&out}, all_points(res, model, x, &out.r, 0),
                 "batch-size: expected at least 1, got 0");
}

TEST(HdbscanRefusals, LargeQueryRowsAreRefusedBeforeAnyOutput) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const fitted_model f(res, three_blobs());
  ASSERT_EQ(f.status, "accepted");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const std::vector<norm_case> cases{
      {.row = {nan, 1}, .got = "nan"},
      {.row = {inf, 0}, .got = "inf"},
      {.row = {0, -inf}, .got = "inf"},
      {.row = {float_overflow, 0}, .got = "1.0000000400817551e+40"},
      {.row = {past_most, 0}, .got = "8.507059173023462e+37"}};
  for (const norm_case& c : cases) {
    SCOPED_TRACE(c.got);
    values host = with_row(c.row, bad_row, rows, cols);
    host[later_bad_row * cols] = nan;
    const auto p = upload<float>(res, {rows, cols}, host);
    const auto p_three = upload<float>(res, {rows / 2, 4}, host);
    record out(res);
    const std::string message = "points: row " + std::to_string(bad_row) +
                                ": expected a squared norm of at most "
                                "4.253529586511731e+37, got " +
                                c.got;
    const allocation_meter meter;
    expect_refused(meter, {&out},
                   approximate_predict(res, f.anchor(), &f.x->view, &p->view,
                                       &out.r),
                   message, scratch_bytes);
    expect_refused(meter, {&out},
                   membership_vector(res, f.anchor(), &f.x->view, &p->view,
                                     &out.r),
                   message, scratch_bytes);
    const allocation_meter column_meter;
    expect_refused(column_meter, {&out},
                   approximate_predict(res, f.anchor(), &f.x->view,
                                       &p_three->view, &out.r),
                   "points: expected 2 columns, as the model was fit on, got 4");
  }
}

TEST(HdbscanRefusals, BoundaryQueryRowsPredict) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const fitted_model f(res, three_blobs());
  ASSERT_EQ(f.status, "accepted");
  values host;
  for (int64_t i = 0; i < rows * cols; ++i) {
    host.push_back(static_cast<double>(i));
  }
  host[bad_row * cols] = half_most;
  host[(bad_row * cols) + 1] = half_most;
  host[(later_bad_row + 1) * cols] = std::numeric_limits<float>::denorm_min();
  const auto p = upload<float>(res, {rows, cols}, host);
  record out(res);
  EXPECT_EQ(
      approximate_predict(res, f.anchor(), &f.x->view, &p->view, &out.r),
      "accepted");
  EXPECT_EQ(shapes(out),
            (std::vector<int64_t>{2, RR_DTYPE_INT64, rows, RR_DTYPE_FLOAT32,
                                  rows}));
}

TEST(HdbscanRefusals, PredictionNeedsTwiceMinSamplesTrainingRows) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const values host = three_blobs();
  const fitted_model short_fit(
      res, values(host.begin(), host.begin() + (short_rows * cols)));
  ASSERT_EQ(short_fit.status, "accepted");
  const fitted_model enough(
      res, values(host.begin(), host.begin() + ((short_rows + 1) * cols)));
  ASSERT_EQ(enough.status, "accepted");
  const auto p = upload<float>(res, {2, cols}, values(host.begin(),
                                                      host.begin() + 4));
  record out(res);
  const allocation_meter meter;
  const std::string message =
      "model: prediction needs at least 10 training rows (2 × min-samples), "
      "got 9";
  expect_refused(meter, {&out},
                 approximate_predict(res, short_fit.anchor(),
                                     &short_fit.x->view, &p->view, &out.r),
                 message);
  expect_refused(meter, {&out},
                 membership_vector(res, short_fit.anchor(), &short_fit.x->view,
                                   &p->view, &out.r),
                 message);
  record fine(res);
  EXPECT_EQ(approximate_predict(res, enough.anchor(), &enough.x->view,
                                &p->view, &fine.r),
            "accepted");
}

TEST(HdbscanRefusals, AllPointsRefusesAnOversizedMatrix) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  values host;
  uint64_t state = 1;
  const auto jitter = [&state] {
    state = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
    return (static_cast<double>(state >> 11) * 0x1p-52) - 1;
  };
  for (int64_t i = 0; i < pair_count; ++i) {
    const double at = (static_cast<double>(i % pair_side) * pair_spacing) +
                      jitter();
    const double up = (static_cast<double>(i / pair_side) * pair_spacing) +
                      jitter();
    host.insert(host.end(), {at, up, at, up});
  }
  record model(res);
  record out(res);
  const auto x = upload<float>(res, {2 * pair_count, cols}, host);
  ASSERT_EQ(hdbscan_fit(res, {.x = &x->view,
                              .model_out = &model.r,
                              .out = &out.r,
                              .min_cluster_size = 2,
                              .min_samples = 1,
                              .cluster_selection_method =
                                  CUML_CLUSTER_HDBSCAN_LEAF}),
            "accepted");
  const int64_t k = model.r.regions[6].shape[0];
  ASSERT_GT(2 * pair_count * k, most_index);
  record membership(res);
  const allocation_meter meter;
  expect_refused(meter, {&membership},
                 all_points(res, &model.r.regions[0], &x->view, &membership.r),
                 "model: " + std::to_string(2 * pair_count) + " rows with " +
                     std::to_string(k) + " clusters need " +
                     std::to_string(2 * pair_count * k) +
                     " entries, more than 2147483647");
}

}  // namespace
}  // namespace cumlrkt::test
