#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "cumlrkt/decomposition.h"
#include "decomposition_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {
namespace {

constexpr int64_t rows = 4;
constexpr int64_t cols = 3;
constexpr int64_t kept = 2;
constexpr int64_t past_index = int64_t{1} << 31;
constexpr int64_t past_columns = 46341;
constexpr int64_t pool = 64;
constexpr int64_t scan_slack = int64_t{4} << 20;
constexpr int32_t no_solver = 2;
constexpr double past_float = 1e39;
constexpr double large = 1e19;
constexpr double huge = 1e38;

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
    EXPECT_EQ(cuml_decomposition_last_error_kind(), RR_ERROR_LOGIC)
        << c.message;
  }
}

template <typename T>
std::vector<T> spread(int64_t n, int64_t d) {
  std::vector<T> host(static_cast<std::size_t>(n * d));
  for (std::size_t i = 0; i < host.size(); ++i) {
    host[i] = static_cast<T>((i * i) % 11);
  }
  return host;
}

template <typename T>
std::vector<T> with(std::vector<T> host, std::size_t at, double value) {
  host[at] = static_cast<T>(value);
  return host;
}

std::string magnitude(int64_t row, int64_t col, double most, float got) {
  return std::format(
      "X: row {}, column {}: expected a magnitude of at most {}, got {}", row,
      col, most, got);
}

struct fit_case {
  explicit fit_case(const resources& res)
      : x(res, spread<float>(rows, cols), {rows, cols}),
        ints(res.r, RR_DTYPE_INT32, {pool}),
        floats(res.r, RR_DTYPE_FLOAT32, {pool}),
        out(res) {}
  filled<float> x;
  device_array ints;
  device_array floats;
  pca_fitted out;
};

int fit(const resources& res, fit_case& c, const rr_view* x,
        const pca_fit_options& o = {}) {
  return cuml_decomposition_pca_fit(
      res.handle, x, &c.out.out.r, o.all_components, o.n_components, o.solver,
      o.tol, o.iterations, &c.out.noise_variance, &c.out.n_samples);
}

TEST(PcaFitRefusals, EachConditionInTableOrder) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  fit_case c(res);
  const rr_view* const x = c.x.view();
  const rr_view ints = shaped(c.ints.view, {rows, cols});
  const rr_view rank1 = shaped(c.floats.view, {rows});
  const rr_view col_major = shaped(c.x.array.view, {rows, cols}, true);
  const rr_view one_row = shaped(c.floats.view, {1, cols});
  const rr_view no_rows = shaped(c.floats.view, {0, cols});
  const rr_view one_col = shaped(c.floats.view, {rows, 1});
  const rr_view tall = shaped(c.floats.view, {past_index, 2});
  const rr_view wide = shaped(c.floats.view, {2, past_columns});
  const filled<float> nan_x(
      res, with(with(spread<float>(rows, cols), 10, nan), 0, large),
      {rows, cols});
  const filled<float> large_x(
      res, with(spread<float>(rows, cols), (3 * cols) + 1, large),
      {rows, cols});
  const filled<float> same(res, std::vector<float>(rows * cols, 2.5F),
                           {rows, cols});
  const pca_fit_options k0{.all_components = 0, .n_components = 0};
  expect_refusals({
      {[&] { return fit(res, c, &ints); },
       "X: expected float32 or float64, got int32"},
      {[&] { return fit(res, c, &rank1); }, "X: expected rank 2, got rank 1"},
      {[&] { return fit(res, c, &col_major); },
       "X: expected row-major, got col-major 4x3"},
      {[&] { return fit(res, c, &one_row); },
       "X: expected at least 2 rows, got 1"},
      {[&] { return fit(res, c, &no_rows); },
       "X: expected at least 2 rows, got 0"},
      {[&] { return fit(res, c, &one_col); },
       "X: expected at least 2 columns, got 1"},
      {[&] { return fit(res, c, &tall); },
       "X: 2147483648 rows do not fit the index type"},
      {[&] { return fit(res, c, &wide); },
       "X: 46341 columns: a 46341x46341 covariance does not fit the index "
       "type"},
      {[&] { return fit(res, c, x, {.solver = no_solver}); },
       "solver: expected full or jacobi"},
      {[&] { return fit(res, c, x, {.solver = -1}); },
       "solver: expected full or jacobi"},
      {[&] { return fit(res, c, x, k0); },
       "n-components: expected at least 1, got 0"},
      {[&] {
         return fit(res, c, x, {.all_components = 0, .n_components = cols + 1});
       },
       "n-components: expected at most 3, got 4"},
      {[&] { return fit(res, c, x, {.tol = -1}); },
       "tol: expected a finite value of at least 0, got -1"},
      {[&] { return fit(res, c, x, {.tol = nan}); },
       "tol: expected a finite value of at least 0, got nan"},
      {[&] { return fit(res, c, x, {.tol = inf}); },
       "tol: expected a finite value of at least 0, got inf"},
      {[&] { return fit(res, c, x, {.tol = past_float}); },
       "tol: expected at most 3.4028234663852886e+38, got 1e+39"},
      {[&] { return fit(res, c, x, {.iterations = 0}); },
       "iterations: expected at least 1, got 0"},
      {[&] { return fit(res, c, x, {.iterations = past_index}); },
       "iterations: expected at most 2147483647, got 2147483648"},
      {[&] { return fit(res, c, nan_x.view()); },
       "X: row 3, column 1: expected a finite value, got nan"},
      {[&] { return fit(res, c, large_x.view()); },
       magnitude(3, 1, std::sqrt(FLT_MAX / (4.0 * rows * cols)), 1e19F)},
      {[&] { return fit(res, c, same.view()); },
       "X: expected at least 2 distinct rows, got 1"},
      {[&] { return fit(res, c, &one_col, {.solver = no_solver}); },
       "X: expected at least 2 columns, got 1"},
      {[&] { return fit(res, c, &wide, k0); },
       "X: 46341 columns: a 46341x46341 covariance does not fit the index "
       "type"},
      {[&] { return fit(res, c, x, {.solver = no_solver, .tol = -1}); },
       "solver: expected full or jacobi"},
      {[&] {
         return fit(res, c, nan_x.view(),
                    {.all_components = 0, .n_components = 0});
       },
       "n-components: expected at least 1, got 0"},
      {[&] { return fit(res, c, same.view(), {.iterations = 0}); },
       "iterations: expected at least 1, got 0"},
  });
  expect_untouched(c.out.out);
  EXPECT_EQ(c.out.n_samples, -1);
  EXPECT_EQ(c.out.noise_variance, -1);
  expect_refusals({
      {[&] {
         return cuml_decomposition_pca_fit(res.handle, x, nullptr, 1, 0, 0, 0,
                                           1, &c.out.noise_variance,
                                           &c.out.n_samples);
       },
       "out is NULL"},
      {[&] {
         return cuml_decomposition_pca_fit(res.handle, x, &c.out.out.r, 1, 0,
                                           0, 0, 1, nullptr, &c.out.n_samples);
       },
       "noise-variance is NULL"},
      {[&] {
         return cuml_decomposition_pca_fit(res.handle, x, &c.out.out.r, 1, 0,
                                           0, 0, 1, &c.out.noise_variance,
                                           nullptr);
       },
       "n-samples is NULL"},
  });
}

TEST(PcaFitRefusals, AllocateNothingButTheScans) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  fit_case c(res);
  const filled<float> same(res, std::vector<float>(rows * cols, 2.5F),
                           {rows, cols});
  {
    const allocation_meter meter;
    EXPECT_NE(fit(res, c, c.x.view(), {.iterations = 0}), RR_OK);
    EXPECT_EQ(meter.peak(), 0);
  }
  const allocation_meter meter;
  EXPECT_NE(fit(res, c, same.view()), RR_OK);
  EXPECT_LE(meter.peak(), scan_slack);
  expect_untouched(c.out.out);
}

struct apply_case {
  explicit apply_case(const resources& res)
      : x(res, spread<float>(rows + 1, cols), {rows + 1, cols}), fitted(res) {
    EXPECT_EQ(pca_fit_into(res, x.view(), fitted,
                           {.all_components = 0, .n_components = kept}),
              "accepted");
  }
  filled<float> x;
  pca_fitted fitted;
};

struct apply_shapes {
  int64_t width;
  int64_t wrong_width;
  bool inverse;
};

void expect_apply_refusals(const resources& res, pca_apply entry,
                           const apply_shapes& s) {
  apply_case c(res);
  const pca_views v = fitted_views(c.fitted);
  const std::vector<float> variance = c.fitted.out.read<float>(1);
  const double largest = *std::max_element(variance.begin(), variance.end());
  const double most = s.inverse ? FLT_MAX / (4.0 * kept) : FLT_MAX / (4.0 * cols);
  const double whitened_most =
      s.inverse ? FLT_MAX / (4.0 * kept * std::max(1.0, std::sqrt(largest)))
                : most;
  const auto last = static_cast<std::size_t>(s.width - 1);
  const device_array ints(res.r, RR_DTYPE_INT32, {pool});
  const device_array floats(res.r, RR_DTYPE_FLOAT32, {pool});
  const device_array doubles(res.r, RR_DTYPE_FLOAT64, {pool});
  const filled<float> x(res, spread<float>(rows, s.width), {rows, s.width});
  const filled<double> x64(res, spread<double>(rows, s.width),
                           {rows, s.width});
  const rr_view int_x = shaped(ints.view, {rows, s.width});
  const rr_view rank1 = shaped(floats.view, {rows});
  const rr_view col_major = shaped(x.array.view, {rows, s.width}, true);
  const rr_view no_rows = shaped(floats.view, {0, s.width});
  const rr_view tall = shaped(floats.view, {past_index, s.width});
  const rr_view off_width = shaped(floats.view, {rows, s.wrong_width});
  const rr_view int_components = shaped(ints.view, {kept, cols});
  const rr_view rank1_components = shaped(floats.view, {cols});
  const rr_view col_components = shaped(floats.view, {kept, cols}, true);
  const rr_view no_components = shaped(floats.view, {0, cols});
  const rr_view thin_components = shaped(floats.view, {kept, 1});
  const rr_view double_vector = shaped(doubles.view, {cols});
  const rr_view short_vector = shaped(floats.view, {cols - 1});
  const rr_view long_vector = shaped(floats.view, {kept + 1});
  rr_view strided = shaped(floats.view, {cols});
  strided.strides[0] = 2;
  const filled<float> nan_x(res, with(spread<float>(rows, s.width), last, inf),
                            {rows, s.width});
  const filled<float> huge_x(res,
                             with(spread<float>(rows, s.width), last, huge),
                             {rows, s.width});
  record out(res);
  const auto call = [&](const rr_view* in, const pca_views& with_views,
                        int32_t whiten = 0) {
    return entry(res.handle, in, with_views.components, with_views.mean,
                 with_views.variance, &out.r, whiten);
  };
  const pca_views no_variance{v.components, v.mean, nullptr};
  expect_refusals({
      {[&] { return call(&int_x, v); },
       "X: expected float32 or float64, got int32"},
      {[&] { return call(&rank1, v); }, "X: expected rank 2, got rank 1"},
      {[&] { return call(&col_major, v); },
       std::format("X: expected row-major, got col-major 4x{}", s.width)},
      {[&] { return call(&no_rows, v); }, "X: expected at least 1 row, got 0"},
      {[&] { return call(&tall, v); },
       "X: 2147483648 rows do not fit the index type"},
      {[&] { return call(x.view(), {&int_components, v.mean, v.variance}); },
       "components: expected float32 or float64, got int32"},
      {[&] { return call(x.view(), {&rank1_components, v.mean, v.variance}); },
       "components: expected rank 2, got rank 1"},
      {[&] { return call(x.view(), {&col_components, v.mean, v.variance}); },
       "components: expected row-major, got col-major 2x3"},
      {[&] { return call(x.view(), {&no_components, v.mean, v.variance}); },
       "components: expected at least 1 row, got 0"},
      {[&] { return call(x.view(), {&thin_components, v.mean, v.variance}); },
       "components: expected at least 2 columns, got 1"},
      {[&] { return call(x64.view(), v); }, "X: expected float32, got float64"},
      {[&] { return call(&off_width, v); },
       std::format("X: expected {} columns, got {}", s.width, s.wrong_width)},
      {[&] { return call(x.view(), {v.components, &double_vector, nullptr}); },
       "mean: expected float32, got float64"},
      {[&] { return call(x.view(), {v.components, &short_vector, nullptr}); },
       "mean: expected 3 elements, got 2"},
      {[&] { return call(x.view(), {v.components, &strided, nullptr}); },
       "mean: expected contiguous elements, got stride 2"},
      {[&] { return call(x.view(), no_variance, 1); },
       "explained-variance: expected a vector, got none"},
      {[&] { return call(x.view(), {v.components, v.mean, &double_vector}, 1); },
       "explained-variance: expected float32, got float64"},
      {[&] { return call(x.view(), {v.components, v.mean, &long_vector}, 1); },
       "explained-variance: expected 2 elements, got 3"},
      {[&] { return call(nan_x.view(), v); },
       std::format("X: row 0, column {}: expected a finite value, got inf",
                   last)},
      {[&] { return call(huge_x.view(), v); },
       magnitude(0, static_cast<int64_t>(last), most, 1e38F)},
      {[&] { return call(huge_x.view(), v, 1); },
       magnitude(0, static_cast<int64_t>(last), whitened_most, 1e38F)},
      {[&] { return call(&col_major, {&int_components, v.mean, nullptr}); },
       std::format("X: expected row-major, got col-major 4x{}", s.width)},
      {[&] { return call(x64.view(), {v.components, &strided, nullptr}); },
       "X: expected float32, got float64"},
      {[&] { return call(nan_x.view(), {v.components, &strided, nullptr}); },
       "mean: expected contiguous elements, got stride 2"},
  });
  expect_untouched(out);
  {
    const allocation_meter meter;
    EXPECT_NE(call(&off_width, v), RR_OK);
    EXPECT_EQ(meter.peak(), 0);
  }
  const allocation_meter meter;
  EXPECT_NE(call(nan_x.view(), v), RR_OK);
  EXPECT_LE(meter.peak(), scan_slack);
  expect_untouched(out);
  EXPECT_EQ(outcome(entry(res.handle, x.view(), v.components, v.mean, nullptr,
                          nullptr, 0)),
            "out is NULL");
}

TEST(PcaTransformRefusals, EachConditionInTableOrder) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  expect_apply_refusals(res, cuml_decomposition_pca_transform,
                        {.width = cols, .wrong_width = kept, .inverse = false});
}

TEST(PcaInverseTransformRefusals, EachConditionInTableOrder) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  expect_apply_refusals(res, cuml_decomposition_pca_inverse_transform,
                        {.width = kept, .wrong_width = cols, .inverse = true});
}

}  // namespace
}  // namespace cumlrkt::test
