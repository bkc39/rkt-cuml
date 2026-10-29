#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "cumlrkt/metrics.h"
#include "metrics_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {
namespace {

constexpr int32_t euclidean = CUML_METRICS_METRIC_EUCLIDEAN;
constexpr int32_t cosine = CUML_METRICS_METRIC_COSINE;
constexpr int64_t chunk = 40000;
constexpr int64_t batch = 512;
constexpr int64_t neighbours = 5;
constexpr int64_t pool = 64;
constexpr int64_t n = 5;
constexpr int64_t sil_rows = 30;
constexpr int64_t trust_rows = 100;
constexpr int64_t most_labels = 46340;
constexpr int64_t past_index = int64_t{1} << 31;
constexpr int64_t wide_rows = 65536;
constexpr int64_t wide_cols = 32769;
constexpr int64_t past_chunk = 50000;
constexpr int64_t square_rows = 46341;
constexpr int64_t long_rows = 1000;
constexpr int64_t long_cols = 2147484;
constexpr int64_t bad_row = 4;
constexpr int64_t zero_row = 3;
constexpr std::size_t row_cycle = 7;
constexpr int64_t twenty_rows_k = 10;
constexpr int64_t wide_k = 32767;
constexpr int64_t half_trust_rows = trust_rows / 2;
constexpr int64_t scratch_slack = int64_t{4} << 20;
constexpr int64_t meter_rows = int64_t{1} << 20;
constexpr float tiny_float = 1e-30F;
constexpr double tiny_double = 1e-170;
constexpr float subnormal_float = 1e-20F;
constexpr double subnormal_double = 1e-160;

struct refusal {
  std::function<int()> call;
  std::string message;
};

void expect_refusals(const std::vector<refusal>& cases) {
  for (const refusal& c : cases) {
    const int status = c.call();
    EXPECT_EQ(status, RR_ERROR) << c.message;
    EXPECT_EQ(outcome(status), c.message);
    EXPECT_EQ(cuml_metrics_last_error_kind(), RR_ERROR_LOGIC) << c.message;
  }
}

struct label_run {
  int64_t distinct;
  int64_t repeats;
};

std::vector<int32_t> counting(label_run run) {
  std::vector<int32_t> host(
      static_cast<std::size_t>(run.distinct + run.repeats),
      static_cast<int32_t>(run.distinct - 1));
  for (int64_t i = 0; i < run.distinct; ++i) {
    host[static_cast<std::size_t>(i)] = static_cast<int32_t>(i);
  }
  return host;
}

std::vector<float> row_with(int64_t rows, int64_t cols, int64_t at,
                            float value) {
  std::vector<float> host(static_cast<std::size_t>(rows * cols));
  for (std::size_t i = 0; i < host.size(); ++i) {
    host[i] = static_cast<float>(i % row_cycle) + 1;
  }
  host[static_cast<std::size_t>(at * cols)] = value;
  return host;
}

std::vector<float> distinct_rows(int64_t rows) {
  std::vector<float> host;
  for (int64_t i = 0; i < rows; ++i) {
    host.push_back(static_cast<float>(i * i));
    host.push_back(static_cast<float>((3 * i) + 1));
  }
  return host;
}

template <typename T>
std::vector<T> cosine_rows(T first) {
  std::vector<T> host(static_cast<std::size_t>(sil_rows * 2), T{1});
  host[static_cast<std::size_t>(zero_row * 2)] = first;
  host[static_cast<std::size_t>((zero_row * 2) + 1)] = 0;
  return host;
}

std::string cosine_message(const std::string& least, const std::string& got) {
  return least_norm_message("X", zero_row, least, got);
}

TEST(LabelRefusals, ContingencyAndRandScore) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<int32_t> a(res, {0, 0, 1, 1, 2}, {n});
  const filled<int64_t> b(res, {3, 3, 4, 4, 4}, {n});
  const device_array floats(res.r, RR_DTYPE_FLOAT32, {pool});
  const device_array doubles(res.r, RR_DTYPE_FLOAT64, {pool});
  const device_array ints(res.r, RR_DTYPE_INT32, {pool});
  const rr_view float_labels = shaped(floats.view, {n});
  const rr_view double_labels = shaped(doubles.view, {n});
  const rr_view rank2 = shaped(ints.view, {n, 2});
  rr_view strided = shaped(ints.view, {n});
  strided.strides[0] = 2;
  const rr_view short_labels = shaped(ints.view, {n - 1});
  const rr_view huge = shaped(ints.view, {past_index});
  rr_view elsewhere = *a.view();
  elsewhere.device = 1;
  const filled<int32_t> many(res, counting({most_labels + 1, 0}),
                             {most_labels + 1});
  const filled<int32_t> few(res, std::vector<int32_t>(most_labels + 1, 7),
                            {most_labels + 1});
  record out(res);
  double score = 0;
  using pair_call = std::function<int(const rr_view*, const rr_view*)>;
  const std::vector<pair_call> entries{
      [&](const rr_view* t, const rr_view* p) {
        return cuml_metrics_contingency_matrix(res.handle, t, p, &out.r);
      },
      [&](const rr_view* t, const rr_view* p) {
        return cuml_metrics_adjusted_rand_score(res.handle, t, p, &score);
      }};
  for (const pair_call& call : entries) {
    expect_refusals({
        {[&] { return call(&float_labels, b.view()); },
         "labels-true: expected int32 or int64, got float32"},
        {[&] { return call(a.view(), &double_labels); },
         "labels-pred: expected int32 or int64, got float64"},
        {[&] { return call(&rank2, b.view()); },
         "labels-true: expected rank 1, got rank 2"},
        {[&] { return call(&strided, b.view()); },
         "labels-true: expected contiguous elements, got stride 2"},
        {[&] { return call(a.view(), &short_labels); },
         "labels-pred: expected 5 elements, got 4"},
        {[&] { return call(&huge, &huge); },
         "labels-true: extent 2147483648 does not fit the index type"},
        {[&] { return call(&elsewhere, b.view()); },
         "labels-true: on device 1, but the resources are on device 0"},
        {[&] { return call(many.view(), few.view()); },
         "labels-true: expected at most 46340 distinct labels, got 46341"},
        {[&] { return call(few.view(), many.view()); },
         "labels-pred: expected at most 46340 distinct labels, got 46341"},
        {[&] { return call(&float_labels, &short_labels); },
         "labels-true: expected int32 or int64, got float32"},
    });
  }
  expect_untouched(out);
  expect_refusals({
      {[&] {
         return cuml_metrics_contingency_matrix(res.handle, a.view(), b.view(),
                                                nullptr);
       },
       "out is NULL"},
      {[&] {
         return cuml_metrics_adjusted_rand_score(res.handle, a.view(), b.view(),
                                                 nullptr);
       },
       "score is NULL"},
  });
}

TEST(LabelRefusals, LeaveOnlyTheRelabelScratch) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<int32_t> many(res, counting({most_labels + 1, 0}),
                             {most_labels + 1});
  const filled<int32_t> few(res, std::vector<int32_t>(most_labels + 1, 7),
                            {most_labels + 1});
  const device_array floats(res.r, RR_DTYPE_FLOAT32, {pool});
  const rr_view float_labels = shaped(floats.view, {most_labels + 1});
  record out(res);
  {
    const allocation_meter meter;
    EXPECT_NE(cuml_metrics_contingency_matrix(res.handle, &float_labels,
                                              few.view(), &out.r),
              RR_OK);
    EXPECT_EQ(meter.peak(), 0);
  }
  const allocation_meter meter;
  EXPECT_NE(cuml_metrics_contingency_matrix(res.handle, many.view(), few.view(),
                                            &out.r),
            RR_OK);
  const auto scratch =
      static_cast<int64_t>((most_labels + 1) * (sizeof(int32_t) * 2));
  EXPECT_LE(meter.peak(), scratch + scratch_slack);
  expect_untouched(out);
}

struct silhouette_case {
  explicit silhouette_case(const resources& res)
      : x(res, row_with(sil_rows, 2, 0, 1), {sil_rows, 2}),
        labels(res, std::vector<int32_t>(sil_rows, 0), {sil_rows}),
        two(res, counting({2, sil_rows - 2}), {sil_rows}),
        distinct(res, counting({sil_rows, 0}), {sil_rows}),
        doubles(res.r, RR_DTYPE_FLOAT64, {sil_rows, 2}),
        ints(res.r, RR_DTYPE_INT32, {pool}),
        floats(res.r, RR_DTYPE_FLOAT32, {pool}) {}
  filled<float> x;
  filled<int32_t> labels;
  filled<int32_t> two;
  filled<int32_t> distinct;
  device_array doubles;
  device_array ints;
  device_array floats;
};

using silhouette_call =
    std::function<int(const rr_view*, const rr_view*, int32_t, int64_t)>;

std::vector<silhouette_call> silhouette_entries(const resources& res,
                                                record& out, double& score) {
  return {[&](const rr_view* x, const rr_view* y, int32_t metric,
              int64_t chunk_size) {
            return cuml_metrics_silhouette_score(res.handle, x, y, metric,
                                                 chunk_size, &score);
          },
          [&](const rr_view* x, const rr_view* y, int32_t metric,
              int64_t chunk_size) {
            return cuml_metrics_silhouette_samples(res.handle, x, y, &out.r,
                                                   metric, chunk_size, &score);
          }};
}

TEST(SilhouetteRefusals, NameTheArgument) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const silhouette_case s(res);
  record out(res);
  double score = 0;
  const rr_view* const x = s.x.view();
  const rr_view* const two = s.two.view();
  const rr_view x_int = shaped(s.ints.view, {sil_rows, 2});
  const rr_view x_rank1 = shaped(s.floats.view, {sil_rows});
  const rr_view x_col = shaped(s.x.array.view, {sil_rows, 2}, true);
  const rr_view x_no_cols = shaped(s.floats.view, {sil_rows, 0});
  const rr_view x_wide = shaped(s.floats.view, {wide_rows, wide_cols});
  const rr_view x_two_rows = shaped(s.x.array.view, {2, 2});
  const rr_view x_past_chunk = shaped(s.floats.view, {past_chunk, 1});
  const rr_view y_float = shaped(s.floats.view, {sil_rows});
  const rr_view y_short = shaped(s.ints.view, {sil_rows - 1});
  const rr_view y_wide = shaped(s.ints.view, {wide_rows});
  const rr_view y_two = shaped(s.ints.view, {2});
  const rr_view y_one = shaped(s.ints.view, {1});
  const rr_view y_past_chunk = shaped(s.ints.view, {past_chunk});
  const filled<float> nan_x(res, row_with(sil_rows, 2, bad_row, NAN),
                            {sil_rows, 2});
  const filled<float> zero_x(res, cosine_rows(0.0F), {sil_rows, 2});
  const filled<float> tiny_x(res, cosine_rows(tiny_float), {sil_rows, 2});
  const filled<double> tiny_double_x(res, cosine_rows(tiny_double),
                                     {sil_rows, 2});
  const filled<float> subnormal_x(res, cosine_rows(subnormal_float),
                                  {sil_rows, 2});
  const filled<double> subnormal_double_x(res, cosine_rows(subnormal_double),
                                          {sil_rows, 2});
  std::vector<double> inf_host(static_cast<std::size_t>(sil_rows * 2), 1.0);
  inf_host[static_cast<std::size_t>(bad_row * 2)] = INFINITY;
  const filled<double> inf_x(res, inf_host, {sil_rows, 2});
  std::vector<float> both_host = cosine_rows(0.0F);
  both_host[static_cast<std::size_t>(bad_row * 2)] = NAN;
  const filled<float> both_x(res, both_host, {sil_rows, 2});
  const std::string metric_message =
      "metric: expected euclidean, sqeuclidean, manhattan, cosine or "
      "chebyshev";
  for (const silhouette_call& call : silhouette_entries(res, out, score)) {
    expect_refusals({
        {[&] { return call(x, two, -1, chunk); }, metric_message},
        {[&] { return call(x, two, cosine + 2, chunk); }, metric_message},
        {[&] { return call(x, two, euclidean, 0); },
         "chunk-size: expected at least 1, got 0"},
        {[&] { return call(x, two, euclidean, -1); },
         "chunk-size: expected at least 1, got -1"},
        {[&] { return call(&x_int, two, euclidean, chunk); },
         "X: expected float32 or float64, got int32"},
        {[&] { return call(&x_rank1, two, euclidean, chunk); },
         "X: expected rank 2, got rank 1"},
        {[&] { return call(&x_col, two, euclidean, chunk); },
         "X: expected row-major, got col-major 30x2"},
        {[&] { return call(x, &y_float, euclidean, chunk); },
         "labels: expected int32 or int64, got float32"},
        {[&] { return call(x, &y_short, euclidean, chunk); },
         "labels: expected 30 elements, got 29"},
        {[&] { return call(&x_no_cols, two, euclidean, chunk); },
         "X: expected at least 1 column, got 0"},
        {[&] { return call(&x_wide, &y_wide, euclidean, chunk); },
         "X: 65536x32769 elements do not fit the index type"},
        {[&] { return call(&x_two_rows, &y_two, euclidean, chunk); },
         "X: expected at least 3 rows, got 2"},
        {[&] {
           return call(&x_past_chunk, &y_past_chunk, euclidean, past_chunk);
         },
         "chunk-size: expected at most 46340, got 50000"},
        {[&] { return call(x, s.labels.view(), euclidean, chunk); },
         "labels: expected at least 2 distinct labels, got 1"},
        {[&] { return call(x, s.distinct.view(), euclidean, chunk); },
         "labels: expected at most 29 distinct labels, got 30"},
        {[&] { return call(nan_x.view(), two, euclidean, chunk); },
         norm_message("X", bad_row, float_most, "nan")},
        {[&] { return call(inf_x.view(), two, euclidean, chunk); },
         norm_message("X", bad_row, double_most, "inf")},
        {[&] { return call(zero_x.view(), two, cosine, chunk); },
         cosine_message(float_least, "0")},
        {[&] { return call(tiny_x.view(), two, cosine, chunk); },
         cosine_message(float_least, "0")},
        {[&] { return call(tiny_double_x.view(), two, cosine, chunk); },
         cosine_message(double_least, "0")},
        {[&] { return call(subnormal_x.view(), two, cosine, chunk); },
         cosine_message(float_least, "1e-40")},
        {[&] { return call(subnormal_double_x.view(), two, cosine, chunk); },
         cosine_message(double_least, "1e-320")},
        {[&] { return call(&x_int, two, -1, 0); }, metric_message},
        {[&] { return call(&x_int, two, euclidean, 0); },
         "chunk-size: expected at least 1, got 0"},
        {[&] { return call(&x_two_rows, &y_one, euclidean, chunk); },
         "labels: expected 2 elements, got 1"},
        {[&] { return call(nan_x.view(), s.labels.view(), euclidean, chunk); },
         "labels: expected at least 2 distinct labels, got 1"},
        {[&] { return call(both_x.view(), two, cosine, chunk); },
         norm_message("X", bad_row, float_most, "nan")},
    });
  }
  expect_untouched(out);
  expect_refusals({{[&] {
                      return cuml_metrics_silhouette_samples(res.handle, x, two,
                                                             nullptr, euclidean,
                                                             chunk, &score);
                    },
                    "out is NULL"},
                   {[&] {
                      return cuml_metrics_silhouette_score(
                          res.handle, x, two, euclidean, chunk, nullptr);
                    },
                    "score is NULL"}});
}

TEST(SilhouetteRefusals, OneLabelLeavesOnlyTheRelabelScratch) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, std::vector<float>(meter_rows, 1),
                        {meter_rows, 1});
  const filled<int32_t> labels(res, std::vector<int32_t>(meter_rows, 7),
                               {meter_rows});
  record out(res);
  double score = 0;
  for (const silhouette_call& call : silhouette_entries(res, out, score)) {
    const allocation_meter meter;
    expect_refusals(
        {{[&] { return call(x.view(), labels.view(), euclidean, chunk); },
          "labels: expected at least 2 distinct labels, got 1"}});
    const auto scratch =
        static_cast<int64_t>(meter_rows * (sizeof(int32_t) * 3));
    EXPECT_LE(meter.peak(), scratch + scratch_slack);
  }
  expect_untouched(out);
}

TEST(SilhouetteRefusals, LabelTableOverflow) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const int64_t rows = square_rows + 1;
  const filled<float> x(res, std::vector<float>(rows, 1), {rows, 1});
  const filled<int32_t> labels(res, counting({square_rows, 1}), {rows});
  record out(res);
  double score = 0;
  for (const silhouette_call& call : silhouette_entries(res, out, score)) {
    expect_refusals(
        {{[&] { return call(x.view(), labels.view(), euclidean, chunk); },
          "labels: 46342 rows with 46341 distinct labels need "
          "2147534622 entries, more than 2147483647"}});
  }
  expect_untouched(out);
}

TEST(TrustworthinessRefusals, NameTheArgument) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, row_with(trust_rows, 2, 0, 1), {trust_rows, 2});
  const device_array doubles(res.r, RR_DTYPE_FLOAT64, {trust_rows, 2});
  const device_array floats(res.r, RR_DTYPE_FLOAT32, {pool});
  const rr_view* const e = x.view();
  const rr_view x_rank1 = shaped(floats.view, {trust_rows});
  const rr_view e_col = shaped(x.array.view, {trust_rows, 2}, true);
  const rr_view e_short = shaped(x.array.view, {trust_rows - 1, 2});
  const rr_view x_no_cols = shaped(floats.view, {trust_rows, 0});
  const rr_view x_twenty = shaped(x.array.view, {20, 2});
  const rr_view x_wide = shaped(floats.view, {wide_rows, 1});
  const rr_view x_square = shaped(floats.view, {square_rows, 1});
  const rr_view x_long = shaped(floats.view, {long_rows, long_cols});
  const rr_view e_long = shaped(floats.view, {long_rows, 1});
  const filled<float> nan_x(res, row_with(trust_rows, 2, bad_row, NAN),
                            {trust_rows, 2});
  const filled<float> big_e(res, row_with(trust_rows, 2, bad_row, 1e20F),
                            {trust_rows, 2});
  double score = 0;
  const auto call = [&](const rr_view* a, const rr_view* b, int64_t k,
                        int32_t metric, int64_t batch_size) {
    return cuml_metrics_trustworthiness(res.handle, a, b, k, metric, batch_size,
                                        &score);
  };
  const std::string metric_message = "metric: expected euclidean";
  expect_refusals({
      {[&] { return call(x.view(), e, neighbours, 1, batch); }, metric_message},
      {[&] { return call(x.view(), e, neighbours, -1, batch); },
       metric_message},
      {[&] { return call(x.view(), e, neighbours, euclidean, 0); },
       "batch-size: expected at least 1, got 0"},
      {[&] { return call(x.view(), e, 0, euclidean, batch); },
       "n-neighbors: expected at least 1, got 0"},
      {[&] { return call(&doubles.view, e, neighbours, euclidean, batch); },
       "X: expected float32, got float64"},
      {[&] {
         return call(x.view(), &doubles.view, neighbours, euclidean, batch);
       },
       "X-embedded: expected float32, got float64"},
      {[&] { return call(&x_rank1, e, neighbours, euclidean, batch); },
       "X: expected rank 2, got rank 1"},
      {[&] { return call(x.view(), &e_col, neighbours, euclidean, batch); },
       "X-embedded: expected row-major, got col-major 100x2"},
      {[&] { return call(x.view(), &e_short, neighbours, euclidean, batch); },
       "X-embedded: expected 100 rows, got 99"},
      {[&] { return call(&x_no_cols, e, neighbours, euclidean, batch); },
       "X: expected at least 1 column, got 0"},
      {[&] { return call(x.view(), &x_no_cols, neighbours, euclidean, batch); },
       "X-embedded: expected at least 1 column, got 0"},
      {[&] {
         return call(&x_twenty, &x_twenty, twenty_rows_k, euclidean, batch);
       },
       "n-neighbors: expected less than half the 20 rows, got 10"},
      {[&] { return call(&x_wide, &x_wide, wide_k, euclidean, batch); },
       "n-neighbors: 65536 rows with 32768 neighbours each need 2147483648 "
       "entries, more than 2147483647"},
      {[&] {
         return call(&x_square, &x_square, neighbours, euclidean, square_rows);
       },
       "batch-size: 46341 rows of 46341 distances need 2147488281 entries, "
       "more than 2147483647"},
      {[&] { return call(&x_long, &e_long, 1, euclidean, 1); },
       "X: 1000x2147484 elements do not fit the index type"},
      {[&] { return call(nan_x.view(), e, neighbours, euclidean, batch); },
       norm_message("X", bad_row, float_most, "nan")},
      {[&] {
         return call(x.view(), big_e.view(), neighbours, euclidean, batch);
       },
       norm_message("X-embedded", bad_row, float_most,
                    "1.0000000400817551e+40")},
      {[&] { return call(&doubles.view, e, 0, -1, 0); }, metric_message},
      {[&] { return call(&doubles.view, e, 0, euclidean, 0); },
       "batch-size: expected at least 1, got 0"},
      {[&] { return call(&doubles.view, e, 0, euclidean, batch); },
       "n-neighbors: expected at least 1, got 0"},
      {[&] {
         return call(&x_twenty, &e_short, twenty_rows_k, euclidean, batch);
       },
       "X-embedded: expected 20 rows, got 99"},
      {[&] { return call(nan_x.view(), e, half_trust_rows, euclidean, batch); },
       "n-neighbors: expected less than half the 100 rows, got 50"},
      {[&] {
         return cuml_metrics_trustworthiness(
             res.handle, x.view(), e, neighbours, euclidean, batch, nullptr);
       },
       "score is NULL"},
  });
}

TEST(TrustworthinessRefusals, AfterARefusedNanCallAValidCallScores) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, distinct_rows(trust_rows), {trust_rows, 2});
  const filled<float> nan_x(res, row_with(trust_rows, 2, bad_row, NAN),
                            {trust_rows, 2});
  double score = -1;
  EXPECT_EQ(cuml_metrics_trustworthiness(res.handle, nan_x.view(), x.view(),
                                         neighbours, euclidean, batch, &score),
            RR_ERROR);
  EXPECT_EQ(outcome(cuml_metrics_trustworthiness(res.handle, x.view(), x.view(),
                                                 neighbours, euclidean, batch,
                                                 &score)),
            "accepted");
  EXPECT_EQ(score, 1.0);
}

}  // namespace
}  // namespace cumlrkt::test
