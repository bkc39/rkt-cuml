#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "cumlrkt/preprocessing.h"
#include "preprocessing_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {
namespace {

constexpr int64_t rows = 4;
constexpr int64_t cols = 3;
constexpr int64_t scratch = int64_t{1} << 26;
constexpr int64_t past_index = int64_t{1} << 31;
constexpr int64_t pool = 64;
constexpr int64_t scan_slack = int64_t{4} << 20;
constexpr std::size_t inf_at = (2 * cols) + 1;
constexpr std::size_t nan_at = (3 * cols) + 0;
constexpr std::size_t last = (rows * cols) - 1;

constexpr double inf = std::numeric_limits<double>::infinity();
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

struct refusal {
  std::function<int()> call;
  std::string message;
};

void expect_refusals(const std::vector<refusal>& cases) {
  for (const refusal& c : cases) {
    const int status = c.call();
    EXPECT_EQ(status, RR_ERROR) << c.message;
    EXPECT_EQ(outcome(status), c.message);
    EXPECT_EQ(cuml_preprocessing_last_error_kind(), RR_ERROR_LOGIC)
        << c.message;
  }
}

template <typename T>
std::vector<T> counting(std::size_t count) {
  std::vector<T> host(count);
  for (std::size_t i = 0; i < count; ++i) {
    host[i] = static_cast<T>(i + 1);
  }
  return host;
}

template <typename T>
std::vector<T> with(std::vector<T> host, std::size_t at, double value) {
  host[at] = static_cast<T>(value);
  return host;
}

struct fit_case {
  explicit fit_case(const resources& res)
      : x(res, counting<float>(rows * cols), {rows, cols}),
        ints(res.r, RR_DTYPE_INT32, {pool}),
        floats(res.r, RR_DTYPE_FLOAT32, {pool}),
        out(res) {}
  filled<float> x;
  device_array ints;
  device_array floats;
  record out;
  int64_t n_samples = -1;
};

int fit(const resources& res, fit_case& c, const rr_view* x,
        int64_t scratch_elements = scratch) {
  return cuml_preprocessing_standard_scaler_fit(
      res.handle, x, &c.out.r, 1, 1, scratch_elements, &c.n_samples);
}

TEST(FitRefusals, EachConditionInTableOrder) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  fit_case c(res);
  const rr_view ints = shaped(c.ints.view, {rows, cols});
  const rr_view rank1 = shaped(c.floats.view, {rows});
  const rr_view col_major = shaped(c.x.array.view, {rows, cols}, true);
  const rr_view no_rows = shaped(c.floats.view, {0, cols});
  const rr_view no_cols = shaped(c.floats.view, {rows, 0});
  const rr_view none = shaped(c.floats.view, {0, 0});
  const rr_view wide = shaped(c.floats.view, {1, past_index});
  rr_view elsewhere = *c.x.view();
  elsewhere.device = 1;
  const filled<float> inf_x(
      res, with(with(counting<float>(rows * cols), inf_at, inf), nan_at, nan),
      {rows, cols});
  const filled<double> nan_x(
      res, with(with(counting<double>(rows * cols), 1, nan), last, -inf),
      {rows, cols});
  const filled<double> neg_x(res, with(counting<double>(rows * cols), 2, -inf),
                             {rows, cols});
  expect_refusals({
      {[&] { return fit(res, c, &ints); },
       "X: expected float32 or float64, got int32"},
      {[&] { return fit(res, c, &rank1); }, "X: expected rank 2, got rank 1"},
      {[&] { return fit(res, c, &col_major); },
       "X: expected row-major, got col-major 4x3"},
      {[&] { return fit(res, c, &no_rows); },
       "X: expected at least 1 row, got 0"},
      {[&] { return fit(res, c, &no_cols); },
       "X: expected at least 1 column, got 0"},
      {[&] { return fit(res, c, &wide); },
       "X: extent 2147483648 does not fit the index type"},
      {[&] { return fit(res, c, c.x.view(), 0); },
       "scratch-elements: expected at least 1, got 0"},
      {[&] { return fit(res, c, c.x.view(), -1); },
       "scratch-elements: expected at least 1, got -1"},
      {[&] { return fit(res, c, inf_x.view()); },
       "X: row 2, column 1: expected a finite value, got inf"},
      {[&] { return fit(res, c, nan_x.view()); },
       "X: row 0, column 1: expected a finite value, got nan"},
      {[&] { return fit(res, c, neg_x.view()); },
       "X: row 0, column 2: expected a finite value, got -inf"},
      {[&] { return fit(res, c, &elsewhere); },
       "X: on device 1, but the resources are on device 0"},
      {[&] {
         const rr_view int_rank1 = shaped(c.ints.view, {rows});
         return fit(res, c, &int_rank1);
       },
       "X: expected float32 or float64, got int32"},
      {[&] { return fit(res, c, &none); }, "X: expected at least 1 row, got 0"},
      {[&] { return fit(res, c, &no_cols, 0); },
       "X: expected at least 1 column, got 0"},
      {[&] { return fit(res, c, &wide, 0); },
       "X: extent 2147483648 does not fit the index type"},
      {[&] { return fit(res, c, inf_x.view(), 0); },
       "scratch-elements: expected at least 1, got 0"},
  });
  expect_untouched(c.out);
  EXPECT_EQ(c.n_samples, -1);
  expect_refusals({
      {[&] {
         return cuml_preprocessing_standard_scaler_fit(
             res.handle, c.x.view(), nullptr, 1, 1, scratch, &c.n_samples);
       },
       "out is NULL"},
      {[&] {
         return cuml_preprocessing_standard_scaler_fit(
             res.handle, c.x.view(), &c.out.r, 1, 1, scratch, nullptr);
       },
       "n-samples is NULL"},
  });
}

TEST(FitRefusals, AllocateNothingButTheScan) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  fit_case c(res);
  const filled<float> inf_x(res, with(counting<float>(rows * cols), 1, inf),
                            {rows, cols});
  {
    const allocation_meter meter;
    EXPECT_NE(fit(res, c, c.x.view(), 0), RR_OK);
    EXPECT_EQ(meter.peak(), 0);
  }
  const allocation_meter meter;
  EXPECT_NE(fit(res, c, inf_x.view()), RR_OK);
  EXPECT_LE(meter.peak(), scan_slack);
  expect_untouched(c.out);
}

struct apply_case {
  explicit apply_case(const resources& res)
      : x(res, counting<float>(rows * cols), {rows, cols}),
        mean(res, counting<double>(cols), {cols}),
        scale(res, counting<double>(cols), {cols}),
        ints(res.r, RR_DTYPE_INT32, {pool}),
        floats(res.r, RR_DTYPE_FLOAT32, {pool}),
        doubles(res.r, RR_DTYPE_FLOAT64, {pool}),
        out(res) {}
  filled<float> x;
  filled<double> mean;
  filled<double> scale;
  device_array ints;
  device_array floats;
  device_array doubles;
  record out;
};

using apply_entry = std::function<int(const rr_view*, const rr_view*,
                                      const rr_view*, rr_deposit*, int64_t)>;

void expect_apply_refusals(const resources& res, const apply_entry& call) {
  apply_case c(res);
  const rr_view* const x = c.x.view();
  const rr_view* const m = c.mean.view();
  const rr_view* const s = c.scale.view();
  const rr_view ints = shaped(c.ints.view, {rows, cols});
  const rr_view rank1 = shaped(c.floats.view, {rows});
  const rr_view col_major = shaped(c.x.array.view, {rows, cols}, true);
  const rr_view no_rows = shaped(c.floats.view, {0, cols});
  const rr_view narrow = shaped(c.floats.view, {rows, 2});
  const rr_view float_vector = shaped(c.floats.view, {cols});
  const rr_view short_vector = shaped(c.doubles.view, {2});
  rr_view strided = shaped(c.doubles.view, {cols});
  strided.strides[0] = 2;
  rr_view elsewhere = *s;
  elsewhere.device = 1;
  const filled<float> nan_x(res, with(counting<float>(rows * cols), 2, nan),
                            {rows, cols});
  rr_deposit* const out = &c.out.r;
  expect_refusals({
      {[&] { return call(&ints, m, s, out, cols); },
       "X: expected float32 or float64, got int32"},
      {[&] { return call(&rank1, m, s, out, cols); },
       "X: expected rank 2, got rank 1"},
      {[&] { return call(&col_major, m, s, out, cols); },
       "X: expected row-major, got col-major 4x3"},
      {[&] { return call(&no_rows, m, s, out, cols); },
       "X: expected at least 1 row, got 0"},
      {[&] { return call(&narrow, m, s, out, cols); },
       "X: expected 3 columns, got 2"},
      {[&] { return call(x, &float_vector, s, out, cols); },
       "mean: expected float64, got float32"},
      {[&] { return call(x, &short_vector, s, out, cols); },
       "mean: expected 3 elements, got 2"},
      {[&] { return call(x, &strided, s, out, cols); },
       "mean: expected contiguous elements, got stride 2"},
      {[&] { return call(x, m, &float_vector, out, cols); },
       "scale: expected float64, got float32"},
      {[&] { return call(x, m, &short_vector, out, cols); },
       "scale: expected 3 elements, got 2"},
      {[&] { return call(x, m, &strided, out, cols); },
       "scale: expected contiguous elements, got stride 2"},
      {[&] { return call(nan_x.view(), m, s, out, cols); },
       "X: row 0, column 2: expected a finite value, got nan"},
      {[&] { return call(x, nullptr, &elsewhere, out, cols); },
       "scale: on device 1, but the resources are on device 0"},
      {[&] { return call(&col_major, m, s, out, 2); },
       "X: expected row-major, got col-major 4x3"},
      {[&] { return call(&narrow, &float_vector, s, out, cols); },
       "X: expected 3 columns, got 2"},
      {[&] { return call(x, &strided, &float_vector, out, cols); },
       "mean: expected contiguous elements, got stride 2"},
      {[&] { return call(nan_x.view(), m, &strided, out, cols); },
       "scale: expected contiguous elements, got stride 2"},
      {[&] { return call(x, m, s, nullptr, cols); }, "out is NULL"},
  });
  expect_untouched(c.out);
  {
    const allocation_meter meter;
    EXPECT_NE(call(&narrow, m, s, out, cols), RR_OK);
    EXPECT_EQ(meter.peak(), 0);
  }
  const allocation_meter meter;
  EXPECT_NE(call(nan_x.view(), m, s, out, cols), RR_OK);
  EXPECT_LE(meter.peak(), scan_slack);
  expect_untouched(c.out);
}

TEST(TransformRefusals, EachConditionInTableOrder) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  expect_apply_refusals(res, [&](const rr_view* x, const rr_view* m,
                                 const rr_view* s, rr_deposit* out,
                                 int64_t d) {
    return cuml_preprocessing_standard_scaler_transform(res.handle, x, m, s,
                                                        out, d);
  });
}

TEST(InverseTransformRefusals, EachConditionInTableOrder) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  expect_apply_refusals(res, [&](const rr_view* x, const rr_view* m,
                                 const rr_view* s, rr_deposit* out,
                                 int64_t d) {
    return cuml_preprocessing_standard_scaler_inverse_transform(res.handle, x,
                                                                m, s, out, d);
  });
}

}  // namespace
}  // namespace cumlrkt::test
