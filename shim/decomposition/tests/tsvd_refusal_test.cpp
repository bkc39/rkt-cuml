#include <gtest/gtest.h>

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
#include "tsvd_fixture.hpp"

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
constexpr double past_norm = 0x1p64;
constexpr double past_norm_double = 0x1p511;
constexpr std::size_t spread_modulus = 11;

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
    host[i] = static_cast<T>((i * i) % spread_modulus);
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
      : x(res, spread<float>(rows, cols), {rows, cols}),
        ints(res.r, RR_DTYPE_INT32, {pool}),
        floats(res.r, RR_DTYPE_FLOAT32, {pool}),
        out(res) {}
  filled<float> x;
  device_array ints;
  device_array floats;
  record out;
};

TEST(TsvdFitRefusals, EachConditionInTableOrder) {
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
      res, with(with(spread<float>(rows, cols), 7, nan), 0, past_norm),
      {rows, cols});
  const filled<float> large_x(
      res, with(spread<float>(rows, cols), 4, past_norm), {rows, cols});
  const filled<double> large_x64(
      res, with(spread<double>(rows, cols), 4, past_norm_double), {rows, cols});
  record& out = c.out;
  expect_refusals({
      {[&] { return tsvd_fit(res, &ints, out); },
       "X: expected float32 or float64, got int32"},
      {[&] { return tsvd_fit(res, &rank1, out); },
       "X: expected rank 2, got rank 1"},
      {[&] { return tsvd_fit(res, &col_major, out); },
       "X: expected row-major, got col-major 4x3"},
      {[&] { return tsvd_fit(res, &one_row, out); },
       "X: expected at least 2 rows, got 1"},
      {[&] { return tsvd_fit(res, &no_rows, out); },
       "X: expected at least 2 rows, got 0"},
      {[&] { return tsvd_fit(res, &one_col, out); },
       "X: expected at least 2 columns, got 1"},
      {[&] { return tsvd_fit(res, &tall, out); },
       "X: 2147483648 rows do not fit the index type"},
      {[&] { return tsvd_fit(res, &wide, out); },
       "X: 46341 columns: a 46341x46341 Gram matrix does not fit the index "
       "type"},
      {[&] { return tsvd_fit(res, x, out, {.n_components = 0}); },
       "n-components: expected at least 1, got 0"},
      {[&] { return tsvd_fit(res, x, out, {.n_components = cols + 1}); },
       "n-components: expected at most 3, got 4"},
      {[&] { return tsvd_fit(res, x, out, {.solver = no_solver}); },
       "solver: expected full or jacobi"},
      {[&] { return tsvd_fit(res, x, out, {.solver = -1}); },
       "solver: expected full or jacobi"},
      {[&] { return tsvd_fit(res, x, out, {.tol = -1}); },
       "tol: expected a finite value of at least 0, got -1"},
      {[&] { return tsvd_fit(res, x, out, {.tol = nan}); },
       "tol: expected a finite value of at least 0, got nan"},
      {[&] { return tsvd_fit(res, x, out, {.tol = inf}); },
       "tol: expected a finite value of at least 0, got inf"},
      {[&] { return tsvd_fit(res, x, out, {.tol = past_float}); },
       "tol: expected at most 3.4028234663852886e+38, got 1e+39"},
      {[&] { return tsvd_fit(res, x, out, {.iterations = 0}); },
       "iterations: expected at least 1, got 0"},
      {[&] { return tsvd_fit(res, x, out, {.iterations = past_index}); },
       "iterations: expected at most 2147483647, got 2147483648"},
      {[&] { return tsvd_fit(res, nan_x.view(), out); },
       "X: row 2, column 1: expected a finite value, got nan"},
      {[&] { return tsvd_fit(res, large_x.view(), out); },
       "X: expected a squared Frobenius norm of at most 4.253529586511731e+37, "
       "got 3.402823669209385e+38"},
      {[&] { return tsvd_fit(res, large_x64.view(), out); },
       "X: expected a squared Frobenius norm of at most "
       "2.247116418577895e+307, got 4.49423283715579e+307"},
      {[&] { return tsvd_fit(res, &one_row, out, {.n_components = 3}); },
       "X: expected at least 2 rows, got 1"},
      {[&] { return tsvd_fit(res, &one_col, out, {.solver = no_solver}); },
       "X: expected at least 2 columns, got 1"},
      {[&] {
         return tsvd_fit(res, x, out, {.n_components = 0, .solver = no_solver});
       },
       "n-components: expected at least 1, got 0"},
      {[&] { return tsvd_fit(res, nan_x.view(), out, {.iterations = 0}); },
       "iterations: expected at least 1, got 0"},
  });
  expect_untouched(out);
  expect_refusals({
      {[&] {
         return cuml_decomposition_tsvd_fit(res.handle, x, nullptr, 1, 0, 0, 1,
                                            0);
       },
       "out is NULL"},
  });
}

TEST(TsvdFitRefusals, AllocateNothingButTheScans) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  fit_case c(res);
  const filled<float> large(res, with(spread<float>(rows, cols), 4, past_norm),
                            {rows, cols});
  {
    const allocation_meter meter;
    EXPECT_NE(tsvd_fit(res, c.x.view(), c.out, {.iterations = 0}), RR_OK);
    EXPECT_EQ(meter.peak(), 0);
  }
  const allocation_meter meter;
  EXPECT_NE(tsvd_fit(res, large.view(), c.out), RR_OK);
  EXPECT_LE(meter.peak(), scan_slack);
  expect_untouched(c.out);
}

struct apply_shapes {
  int64_t width;
  int64_t wrong_width;
  const char* most;
};

void expect_apply_refusals(const resources& res, tsvd_apply entry,
                           const apply_shapes& s) {
  const filled<float> fit_x(res, spread<float>(rows + 1, cols),
                            {rows + 1, cols});
  record fitted(res);
  ASSERT_EQ(tsvd_fit_into(res, fit_x.view(), fitted, {.n_components = kept}),
            "accepted");
  const rr_view* const c = &fitted.r.regions[0];
  const device_array ints(res.r, RR_DTYPE_INT32, {pool});
  const device_array floats(res.r, RR_DTYPE_FLOAT32, {pool});
  const device_array doubles(res.r, RR_DTYPE_FLOAT64, {pool});
  const filled<float> x(res, spread<float>(rows, s.width), {rows, s.width});
  const rr_view int_x = shaped(ints.view, {rows, s.width});
  const rr_view rank1 = shaped(floats.view, {rows});
  const rr_view col_major = shaped(x.array.view, {rows, s.width}, true);
  const rr_view no_rows = shaped(floats.view, {0, s.width});
  const rr_view tall = shaped(floats.view, {past_index, s.width});
  const rr_view x64 = shaped(doubles.view, {rows, s.width});
  const rr_view off_width = shaped(floats.view, {rows, s.wrong_width});
  const rr_view int_components = shaped(ints.view, {kept, cols});
  const rr_view rank1_components = shaped(floats.view, {cols});
  const rr_view col_components = shaped(floats.view, {kept, cols}, true);
  const rr_view no_components = shaped(floats.view, {0, cols});
  const rr_view narrow_components = shaped(floats.view, {kept, 1});
  const filled<float> nan_x(
      res, with(spread<float>(rows, s.width), s.width - 1, nan),
      {rows, s.width});
  const filled<float> huge_x(res, with(spread<float>(rows, s.width), 1, 1e38),
                             {rows, s.width});
  const filled<float> huge_nan_x(
      res, with(with(spread<float>(rows, s.width), 0, 1e38), 1, nan),
      {rows, s.width});
  record out(res);
  expect_refusals({
      {[&] { return entry(res.handle, &int_x, c, &out.r); },
       "X: expected float32 or float64, got int32"},
      {[&] { return entry(res.handle, &rank1, c, &out.r); },
       "X: expected rank 2, got rank 1"},
      {[&] { return entry(res.handle, &col_major, c, &out.r); },
       std::format("X: expected row-major, got col-major 4x{}", s.width)},
      {[&] { return entry(res.handle, &no_rows, c, &out.r); },
       "X: expected at least 1 row, got 0"},
      {[&] { return entry(res.handle, &tall, c, &out.r); },
       "X: 2147483648 rows do not fit the index type"},
      {[&] { return entry(res.handle, x.view(), &int_components, &out.r); },
       "components: expected float32 or float64, got int32"},
      {[&] { return entry(res.handle, x.view(), &rank1_components, &out.r); },
       "components: expected rank 2, got rank 1"},
      {[&] { return entry(res.handle, x.view(), &col_components, &out.r); },
       "components: expected row-major, got col-major 2x3"},
      {[&] { return entry(res.handle, x.view(), &no_components, &out.r); },
       "components: expected at least 1 row, got 0"},
      {[&] { return entry(res.handle, x.view(), &narrow_components, &out.r); },
       "components: expected at least 2 columns, got 1"},
      {[&] { return entry(res.handle, &x64, c, &out.r); },
       "X: expected float32, got float64"},
      {[&] { return entry(res.handle, &off_width, c, &out.r); },
       std::format("X: expected {} columns, got {}", s.width, s.wrong_width)},
      {[&] { return entry(res.handle, nan_x.view(), c, &out.r); },
       std::format("X: row 0, column {}: expected a finite value, got nan",
                   s.width - 1)},
      {[&] { return entry(res.handle, huge_nan_x.view(), c, &out.r); },
       "X: row 0, column 1: expected a finite value, got nan"},
      {[&] { return entry(res.handle, huge_x.view(), c, &out.r); },
       std::format(
           "X: row 0, column 1: expected a magnitude of at most {}, got 1e+38",
           s.most)},
      {[&] { return entry(res.handle, &no_rows, &int_components, &out.r); },
       "X: expected at least 1 row, got 0"},
      {[&] { return entry(res.handle, x.view(), c, nullptr); }, "out is NULL"},
  });
  expect_untouched(out);
}

TEST(TsvdTransformRefusals, EachConditionInTableOrder) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  expect_apply_refusals(res, cuml_decomposition_tsvd_transform,
                        {.width = cols,
                         .wrong_width = kept,
                         .most = "2.8356862219877405e+37"});
}

TEST(TsvdInverseTransformRefusals, EachConditionInTableOrder) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  expect_apply_refusals(res, cuml_decomposition_tsvd_inverse_transform,
                        {.width = kept,
                         .wrong_width = cols,
                         .most = "4.2535293329816107e+37"});
}

}  // namespace
}  // namespace cumlrkt::test
