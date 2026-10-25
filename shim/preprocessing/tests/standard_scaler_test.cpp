#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "cumlrkt/preprocessing.h"
#include "preprocessing_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {
namespace {

constexpr int64_t rows = 4;
constexpr int64_t cols = 2;
constexpr int64_t scratch = int64_t{1} << 26;
constexpr int64_t wide_rows = 257;
constexpr int64_t wide_cols = 9;
constexpr int64_t odd_chunk = 3;
constexpr std::size_t cycle = 13;
constexpr std::size_t offsets = 5;
constexpr float spread = 7;

const std::vector<double> grid{1, 10, 3, 10, 1, 14, 3, 14};
const std::vector<double> unit{-1, -1, 1, -1, -1, 1, 1, 1};

struct fitted {
  explicit fitted(const resources& res) : out(res) {}
  record out;
  int64_t n_samples = -1;
  std::string status;
};

void fit_into(const resources& res, const rr_view* x, fitted& f,
              int32_t with_mean = 1, int32_t with_std = 1,
              int64_t scratch_elements = scratch) {
  f.status = outcome(cuml_preprocessing_standard_scaler_fit(
      res.handle, x, &f.out.r, with_mean, with_std, scratch_elements,
      &f.n_samples));
}

template <typename T>
std::vector<T> applied(const resources& res, bool inverse, const rr_view* x,
                       const rr_view* mean, const rr_view* scale,
                       int64_t d = cols) {
  record out(res);
  const int status =
      inverse ? cuml_preprocessing_standard_scaler_inverse_transform(
                    res.handle, x, mean, scale, &out.r, d)
              : cuml_preprocessing_standard_scaler_transform(
                    res.handle, x, mean, scale, &out.r, d);
  EXPECT_EQ(outcome(status), "accepted");
  EXPECT_EQ(out.r.count, 1);
  EXPECT_EQ(out.r.regions[0].dtype, dtype_of<T>);
  EXPECT_EQ(out.r.regions[0].rank, 2);
  return out.read<T>(0);
}

TEST(StandardScaler, FitDepositsMeanVarianceAndScale) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> x(res, grid, {rows, cols});
  fitted f(res);
  fit_into(res, x.view(), f);
  ASSERT_EQ(f.status, "accepted");
  EXPECT_EQ(f.n_samples, rows);
  ASSERT_EQ(f.out.r.count, 3);
  EXPECT_EQ(f.out.r.regions[0].dtype, RR_DTYPE_FLOAT64);
  EXPECT_EQ(f.out.read<double>(0), (std::vector<double>{2, 12}));
  EXPECT_EQ(f.out.read<double>(1), (std::vector<double>{1, 4}));
  EXPECT_EQ(f.out.read<double>(2), (std::vector<double>{1, 2}));
}

TEST(StandardScaler, WithoutStdDepositsTheMeanOnly) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, {1, 10, 3, 10, 1, 14, 3, 14}, {rows, cols});
  for (const int32_t with_mean : {0, 1}) {
    fitted f(res);
    fit_into(res, x.view(), f, with_mean, 0);
    ASSERT_EQ(f.status, "accepted");
    ASSERT_EQ(f.out.r.count, 1);
    EXPECT_EQ(f.out.r.regions[0].dtype, RR_DTYPE_FLOAT64);
    EXPECT_EQ(f.out.read<double>(0), (std::vector<double>{2, 12}));
  }
}

TEST(StandardScaler, AConstantColumnScalesByOne) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, {5, 1, 5, 3}, {2, cols});
  fitted f(res);
  fit_into(res, x.view(), f);
  ASSERT_EQ(f.status, "accepted");
  EXPECT_EQ(f.out.read<double>(1), (std::vector<double>{0, 1}));
  EXPECT_EQ(f.out.read<double>(2), (std::vector<double>{1, 1}));
}

TEST(StandardScaler, ATinySpreadOnALargeMeanKeepsItsVariance) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  constexpr int64_t samples = 1000;
  constexpr double level = 1e6;
  constexpr double bump = 1e-4;
  constexpr double exact_var = 9.989989319111282e-12;
  constexpr double relative = 1e-11;
  std::vector<double> values(samples, level);
  values[0] = level + bump;
  const filled<double> x(res, values, {samples, 1});
  fitted f(res);
  fit_into(res, x.view(), f);
  ASSERT_EQ(f.status, "accepted");
  EXPECT_NEAR(f.out.read<double>(1)[0], exact_var, relative * exact_var);
}

TEST(StandardScaler, TransformAndInverseOfTheGrid) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> x(res, grid, {rows, cols});
  const filled<double> t(res, unit, {rows, cols});
  const filled<double> mean(res, {2, 12}, {cols});
  const filled<double> scale(res, {1, 2}, {cols});
  EXPECT_EQ(applied<double>(res, false, x.view(), mean.view(), scale.view()),
            unit);
  EXPECT_EQ(applied<double>(res, true, t.view(), mean.view(), scale.view()),
            grid);
  EXPECT_EQ(applied<double>(res, false, x.view(), nullptr, nullptr), grid);
  EXPECT_EQ(applied<double>(res, true, x.view(), nullptr, nullptr), grid);
  EXPECT_EQ(applied<double>(res, false, x.view(), mean.view(), nullptr),
            (std::vector<double>{-1, -2, 1, -2, -1, 2, 1, 2}));
  const filled<float> narrow(res, {1, 10, 3, 10, 1, 14, 3, 14}, {rows, cols});
  EXPECT_EQ(applied<float>(res, false, narrow.view(), mean.view(), scale.view()),
            (std::vector<float>{-1, -1, 1, -1, -1, 1, 1, 1}));
}

TEST(StandardScaler, ChunkedFitsAreBitwiseEqual) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  std::vector<float> host(static_cast<std::size_t>(wide_rows * wide_cols));
  for (std::size_t i = 0; i < host.size(); ++i) {
    host[i] = (static_cast<float>(i % cycle) / spread) +
              static_cast<float>(i % offsets);
  }
  const filled<float> x(res, host, {wide_rows, wide_cols});
  fitted whole(res);
  fit_into(res, x.view(), whole);
  ASSERT_EQ(whole.status, "accepted");
  for (const int64_t elements : {int64_t{1}, wide_rows, odd_chunk * wide_rows,
                                 (wide_rows * wide_cols) - 1}) {
    fitted chunked(res);
    fit_into(res, x.view(), chunked, 1, 1, elements);
    ASSERT_EQ(chunked.status, "accepted");
    for (const int32_t i : {0, 1, 2}) {
      const std::vector<double> a = whole.out.read<double>(i);
      const std::vector<double> b = chunked.out.read<double>(i);
      EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(double)), 0)
          << "region " << i << " at " << elements;
    }
  }
}

TEST(StandardScaler, ViewsOnAnotherDeviceAreRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> x(res, grid, {rows, cols});
  const filled<double> mean(res, {2, 12}, {cols});
  rr_view x_elsewhere = *x.view();
  x_elsewhere.device = 1;
  rr_view mean_elsewhere = *mean.view();
  mean_elsewhere.device = 1;
  fitted f(res);
  fit_into(res, &x_elsewhere, f);
  EXPECT_EQ(f.status, "X: on device 1, but the resources are on device 0");
  record out(res);
  EXPECT_EQ(outcome(cuml_preprocessing_standard_scaler_transform(
                res.handle, x.view(), &mean_elsewhere, nullptr, &out.r, cols)),
            "mean: on device 1, but the resources are on device 0");
  EXPECT_EQ(outcome(cuml_preprocessing_standard_scaler_inverse_transform(
                res.handle, x.view(), nullptr, &mean_elsewhere, &out.r, cols)),
            "scale: on device 1, but the resources are on device 0");
  expect_untouched(out);
}

}  // namespace
}  // namespace cumlrkt::test
