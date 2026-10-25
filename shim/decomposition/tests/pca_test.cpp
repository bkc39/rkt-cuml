#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "cumlrkt/decomposition.h"
#include "decomposition_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {
namespace {

constexpr int64_t rows = 9;
constexpr int64_t cols = 3;
constexpr int32_t regions = 5;
constexpr double first_variance = 8;
constexpr double second_variance = 1;
constexpr double third_variance = 0.25;
constexpr double total_variance =
    first_variance + second_variance + third_variance;

template <typename T>
std::vector<T> axis_rows() {
  return {0, 4, 0, 0,  -4, 0, 0, 4,  0, 0, -4, 0, 0, 0,
          2, 0, 0, -2, 1,  0, 0, -1, 0, 0, 0,  0, 0};
}

template <typename T>
std::vector<T> permuted(const std::vector<T>& x) {
  std::vector<T> out;
  for (std::size_t i = 0; i < x.size(); i += cols) {
    out.insert(out.end(), {x[i + 1], x[i + 2], x[i]});
  }
  return out;
}

template <typename T>
std::vector<T> variances() {
  return {static_cast<T>(first_variance), static_cast<T>(second_variance),
          static_cast<T>(third_variance)};
}

template <typename T>
void expect_axis_spectrum(const pca_fitted& f) {
  std::vector<T> ratio;
  std::vector<T> singular;
  for (const T v : variances<T>()) {
    ratio.push_back(static_cast<T>(static_cast<double>(v) / total_variance));
    singular.push_back(
        static_cast<T>(std::sqrt(static_cast<double>(v) * (rows - 1))));
  }
  EXPECT_EQ(f.out.read<T>(1), variances<T>());
  EXPECT_EQ(f.out.read<T>(2), ratio);
  EXPECT_EQ(f.out.read<T>(3), singular);
}

template <typename T>
void expect_unchanged(const filled<T>& x, const std::vector<T>& host) {
  std::vector<T> after(host.size());
  EXPECT_EQ(cudaMemcpy(after.data(), x.view()->data, after.size() * sizeof(T),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(after, host);
}

template <typename T>
void expect_axis_fit(int32_t solver) {
  const resources res;
  const std::vector<T> host = axis_rows<T>();
  const filled<T> x(res, host, {rows, cols});
  pca_fitted f(res);
  ASSERT_EQ(pca_fit_into(res, x.view(), f, {.solver = solver}), "accepted");
  ASSERT_EQ(f.out.r.count, regions);
  EXPECT_EQ(f.out.read<T>(0), (std::vector<T>{0, 1, 0, 0, 0, 1, 1, 0, 0}));
  expect_axis_spectrum<T>(f);
  EXPECT_EQ(f.out.read<T>(4), (std::vector<T>{0, 0, 0}));
  EXPECT_EQ(f.noise_variance, 0.0);
  EXPECT_EQ(f.n_samples, rows);
  expect_unchanged(x, host);
}

TEST(Pca, AxisFixtureFitsExactly) {
  CUMLRKT_REQUIRE_GPU();
  for (const int32_t solver :
       {CUML_DECOMPOSITION_SOLVER_FULL, CUML_DECOMPOSITION_SOLVER_JACOBI}) {
    expect_axis_fit<float>(solver);
    expect_axis_fit<double>(solver);
  }
}

TEST(Pca, NoiseVarianceAveragesTheComponentsLeftOut) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> x(res, axis_rows<double>(), {rows, cols});
  pca_fitted f(res);
  ASSERT_EQ(
      pca_fit_into(res, x.view(), f, {.all_components = 0, .n_components = 1}),
      "accepted");
  EXPECT_EQ(f.out.r.regions[0].shape[0], 1);
  EXPECT_EQ(f.out.read<double>(0), (std::vector<double>{0, 1, 0}));
  EXPECT_EQ(f.noise_variance, (second_variance + third_variance) / 2);
}

template <typename T>
void expect_axis_transforms() {
  const resources res;
  const std::vector<T> host = axis_rows<T>();
  const filled<T> x(res, host, {rows, cols});
  pca_fitted f(res);
  ASSERT_EQ(pca_fit_into(res, x.view(), f), "accepted");
  const pca_views v = fitted_views(f);
  record t(res);
  ASSERT_EQ(
      pca_apply_into(res, cuml_decomposition_pca_transform, x.view(), v, t),
      "accepted");
  EXPECT_EQ(t.read<T>(0), permuted(host));
  record back(res);
  ASSERT_EQ(pca_apply_into(res, cuml_decomposition_pca_inverse_transform,
                           &t.r.regions[0], v, back),
            "accepted");
  EXPECT_EQ(back.read<T>(0), host);
}

template <typename T>
void expect_axis_whitening() {
  const resources res;
  const filled<T> x(res, axis_rows<T>(), {rows, cols});
  pca_fitted f(res);
  ASSERT_EQ(pca_fit_into(res, x.view(), f), "accepted");
  const pca_views v = fitted_views(f);
  const filled<T> one(res, {0, 4, 0}, {1, cols});
  record whitened(res);
  ASSERT_EQ(pca_apply_into(res, cuml_decomposition_pca_transform, one.view(), v,
                           whitened, 1),
            "accepted");
  const T scale = std::sqrt(static_cast<T>(first_variance));
  EXPECT_EQ(whitened.read<T>(0),
            (std::vector<T>{static_cast<T>(4) / scale, static_cast<T>(0),
                            static_cast<T>(0)}));
  record unwhitened(res);
  ASSERT_EQ(pca_apply_into(res, cuml_decomposition_pca_inverse_transform,
                           &whitened.r.regions[0], v, unwhitened, 1),
            "accepted");
  const T whitened_first = static_cast<T>(4) / scale;
  EXPECT_EQ(unwhitened.read<T>(0),
            (std::vector<T>{0, whitened_first * scale, 0}));
}

TEST(Pca, AxisFixtureTransformsExactlyWhateverTheRowCount) {
  CUMLRKT_REQUIRE_GPU();
  expect_axis_transforms<float>();
  expect_axis_transforms<double>();
  expect_axis_whitening<float>();
  expect_axis_whitening<double>();
}

void expect_apply_elsewhere(const resources& res, pca_apply entry,
                            const rr_view* x, const rr_view& x_elsewhere,
                            const pca_views& v) {
  rr_view components = *v.components;
  components.device = 1;
  rr_view mean = *v.mean;
  mean.device = 1;
  rr_view variance = *v.variance;
  variance.device = 1;
  record out(res);
  EXPECT_EQ(pca_apply_into(res, entry, &x_elsewhere, v, out),
            "X: on device 1, but the resources are on device 0");
  EXPECT_EQ(
      pca_apply_into(res, entry, x, {&components, v.mean, v.variance}, out),
      "components: on device 1, but the resources are on device 0");
  EXPECT_EQ(pca_apply_into(res, entry, x, {v.components, &mean, nullptr}, out),
            "mean: on device 1, but the resources are on device 0");
  EXPECT_EQ(
      pca_apply_into(res, entry, x, {v.components, v.mean, &variance}, out, 1),
      "explained-variance: on device 1, but the resources are on "
      "device 0");
  expect_untouched(out);
}

TEST(Pca, ViewsOnAnotherDeviceAreRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> x(res, axis_rows<double>(), {rows, cols});
  pca_fitted f(res);
  ASSERT_EQ(pca_fit_into(res, x.view(), f), "accepted");
  rr_view x_elsewhere = *x.view();
  x_elsewhere.device = 1;
  pca_fitted g(res);
  EXPECT_EQ(pca_fit_into(res, &x_elsewhere, g),
            "X: on device 1, but the resources are on device 0");
  expect_untouched(g.out);
  const pca_views v = fitted_views(f);
  for (const pca_apply entry : {cuml_decomposition_pca_transform,
                                cuml_decomposition_pca_inverse_transform}) {
    expect_apply_elsewhere(res, entry, x.view(), x_elsewhere, v);
  }
}

TEST(Pca, OutputsAreFiniteJustUnderTheOverflowBound) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  constexpr int64_t n = 50;
  constexpr int64_t d = 4;
  constexpr std::size_t period = 7;
  constexpr float under_bound = 6e17F;
  std::vector<float> host(static_cast<std::size_t>(n * d));
  for (std::size_t i = 0; i < host.size(); ++i) {
    host[i] = static_cast<float>(i % period) - static_cast<float>(i % 3);
  }
  host[(3 * d) + 1] = under_bound;
  const filled<float> x(res, host, {n, d});
  pca_fitted f(res);
  ASSERT_EQ(pca_fit_into(res, x.view(), f), "accepted");
  for (int32_t i = 0; i < regions; ++i) {
    for (const float v : f.out.read<float>(i)) {
      EXPECT_TRUE(std::isfinite(v)) << "region " << i;
    }
  }
  EXPECT_TRUE(std::isfinite(f.noise_variance));
}

}  // namespace
}  // namespace cumlrkt::test
