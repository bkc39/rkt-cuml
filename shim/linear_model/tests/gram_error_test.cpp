#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <vector>

#include "block_gram.hpp"
#include "cumlrkt/support.hpp"
#include "glm_staging_device.hpp"
#include "gram_condition.hpp"
#include "linear_model_fixture.hpp"
#include "raftrkt/view.hpp"
#include "ritz.hpp"

namespace {

using cumlrkt::stream_of;
using cumlrkt::glm::block_rows_of;
using cumlrkt::glm::condition_estimate;
using cumlrkt::glm::equilibrated_condition;
using cumlrkt::glm::narrow_block_grams;
using cumlrkt::glm::to_device;
using cumlrkt::glm::to_host;
using cumlrkt::glm::wide_block_grams;
using cumlrkt::glm::ritz::bound_of;
using cumlrkt::glm::ritz::gram_batch;
using cumlrkt::glm::ritz::gram_bound;
using cumlrkt::glm::ritz::gram_error;
using cumlrkt::glm::ritz::gram_path;
using cumlrkt::glm::ritz::gram_run_rows;
using cumlrkt::glm::ritz::mean_error;
using cumlrkt::glm::ritz::wide_batch;
using cumlrkt::test::filled;
using cumlrkt::test::resources;

constexpr double refusal_kappa = 0x1p32;
constexpr int64_t sign_columns = 4;
constexpr int64_t sign_patterns = 6;
constexpr std::array<std::array<double, sign_columns>, sign_patterns> signs{
    {{1, 1, -1, -1},
     {1, -1, 1, -1},
     {1, -1, -1, 1},
     {-1, 1, 1, -1},
     {-1, 1, -1, 1},
     {-1, -1, 1, 1}}};
constexpr int64_t hadamard_rows = 1024;
constexpr double hadamard_loading = 0x1p16;
constexpr int hadamard_spread_bits = 10;
constexpr int64_t wide_columns = 100;
constexpr int64_t kernel_blocks = 2;
constexpr double rounding_lean = 0x1.fcp-18;
constexpr double wide_lean = 0x1.fcp-16;
constexpr int64_t wide_kernel_columns = 17;
constexpr int64_t mean_rows = 1000000;
constexpr double mean_offset = 0x1p20;

struct design {
  std::vector<double> x;
  int64_t n;
  int64_t d;
};

condition_estimate estimate_of(const resources& res, const design& c,
                               bool fit_intercept) {
  const std::vector<float> host(c.x.begin(), c.x.end());
  const filled<float> x(res, host, {c.n * c.d});
  return equilibrated_condition<float>(
      raftrkt::handle_of(res.handle),
      static_cast<const float*>(x.array.view.data), nullptr, c.n, c.d, 0.0F,
      fit_intercept, refusal_kappa);
}

void expect_refused(const resources& res, const design& c, bool fit_intercept) {
  const condition_estimate e = estimate_of(res, c, fit_intercept);
  EXPECT_NE(e.path, gram_path::certified) << c.n << " " << c.d;
  EXPECT_NE(e.path, gram_path::enclosed) << c.n << " " << c.d;
  EXPECT_GE(e.kappa, refusal_kappa) << c.n << " " << c.d;
}

design dummies_beside_ones(int64_t n) {
  design c{std::vector<double>(3 * n, 0.0), n, 3};
  for (int64_t i = 0; i < n; ++i) {
    c.x[((i % 2) * n) + i] = 1.0;
    c.x[(2 * n) + i] = 1.0;
  }
  return c;
}

design signs_summing_to_zero(int64_t n) {
  design c{std::vector<double>(sign_columns * n), n, sign_columns};
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t k = 0; k < sign_columns; ++k) {
      c.x[(k * n) + i] = signs.at(i % sign_patterns).at(k);
    }
  }
  return c;
}

double hadamard(int64_t i, int64_t j) {
  return std::popcount(static_cast<uint64_t>(i & j)) % 2 == 0 ? 1.0 : -1.0;
}

design hadamard_pair(int spread_bits) {
  const int64_t n = hadamard_rows;
  design c{std::vector<double>(2 * n), n, 2};
  for (int64_t j = 0; j < 2; ++j) {
    for (int64_t i = 0; i < n; ++i) {
      c.x[(j * n) + i] =
          (hadamard(i, j + 1) + (hadamard_loading * hadamard(i, n - 1))) *
          std::ldexp(1.0, spread_bits * static_cast<int>(j));
    }
  }
  return c;
}

TEST(GramError, DummiesBesideAOnesColumnAreRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (const int64_t n : {23379, 57115}) {
    expect_refused(res, dummies_beside_ones(n), false);
  }
}

TEST(GramError, SignColumnsSummingToZeroAreRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (const int64_t n : {4596, 4818}) {
    for (const bool fit_intercept : {false, true}) {
      expect_refused(res, signs_summing_to_zero(n), fit_intercept);
    }
  }
}

TEST(GramError, AHadamardPairPastTheThresholdIsRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (const int spread_bits : {0, hadamard_spread_bits}) {
    for (const bool fit_intercept : {false, true}) {
      expect_refused(res, hadamard_pair(spread_bits), fit_intercept);
    }
  }
}

TEST(GramError, TheNarrowKernelStaysWithinItsRunBound) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const raft::handle_t& handle = raftrkt::handle_of(res.handle);
  const int64_t rows = kernel_blocks * cumlrkt::glm::ritz::gram_block_rows;
  const auto c = static_cast<float>(std::sqrt(1.0 + rounding_lean));
  const double square = static_cast<double>(c) * static_cast<double>(c);
  const auto x = to_device(std::vector<float>(rows, c), stream_of(handle));
  const auto zero = to_device(std::vector<double>{0.0}, stream_of(handle));
  const auto one = to_device(std::vector<double>{1.0}, stream_of(handle));
  auto parts =
      to_device(std::vector<double>(kernel_blocks, 0.0), stream_of(handle));
  narrow_block_grams<float>(
      stream_of(handle),
      block_rows_of<float>{x.data(), nullptr, zero.data(), one.data(), rows, 1},
      0, kernel_blocks, parts.data());
  const double exact =
      square * static_cast<double>(cumlrkt::glm::ritz::gram_block_rows);
  const double bound = gram_error(
      {.runs = cumlrkt::glm::ritz::gram_block_rows / gram_run_rows(1),
       .run_rows = gram_run_rows(1),
       .trace = exact,
       .diagonal = 0.0});
  for (const double part : to_host(parts, stream_of(handle))) {
    EXPECT_LE(std::abs(part - exact), bound);
  }
}

TEST(GramError, TheWideKernelStaysWithinItsRunBound) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const raft::handle_t& handle = raftrkt::handle_of(res.handle);
  const int64_t d = wide_kernel_columns;
  const int64_t m = gram_run_rows(d);
  const gram_batch batch =
      wide_batch({kernel_blocks * cumlrkt::glm::ritz::gram_block_rows, d});
  const int64_t rows = batch.blocks * batch.run;
  const auto c = static_cast<float>(std::sqrt(1.0 + wide_lean));
  const double square = static_cast<double>(c) * static_cast<double>(c);
  const auto x = to_device(std::vector<float>(rows * d, c), stream_of(handle));
  auto parts = to_device(std::vector<float>(batch.blocks * d * d, 0.0F),
                         stream_of(handle));
  cublasHandle_t cublas = raft::resource::get_cublas_handle(handle);
  ASSERT_EQ(cublasSetStream(cublas, stream_of(handle).value()),
            CUBLAS_STATUS_SUCCESS);
  wide_block_grams<float>(cublas, x.data(), rows, d, batch, parts.data());
  const double exact = square * static_cast<double>(m);
  const double bound =
      gram_error({.runs = 1, .run_rows = m, .trace = exact, .diagonal = 0.0});
  for (const float part : to_host(parts, stream_of(handle))) {
    EXPECT_LE(std::abs(static_cast<double>(part) - exact), bound);
  }
}

TEST(GramError, TheMeansRoundingEntersTheBound) {
  const std::vector<double> spread{static_cast<double>(mean_rows)};
  const std::vector<double> scale{1.0 / std::sqrt(spread[0])};
  const double centred = mean_error(mean_rows, spread, scale,
                                    {{mean_offset}, double{mean_rows}, {}});
  const double reach =
      2.0 * static_cast<double>(mean_rows) * 0x1p-53 * mean_offset;
  EXPECT_GT(centred, double{mean_rows} * reach * reach * scale[0] * scale[0]);
  EXPECT_EQ(mean_error(mean_rows, spread, scale, {{}, 0.0, {}}), 0.0);
  EXPECT_LT(mean_error(mean_rows, spread, scale,
                       {{mean_offset}, double{mean_rows}, {0.0}}),
            centred / mean_offset);
  gram_bound b = bound_of(mean_rows, spread, scale, 0.0);
  const double without = gram_error(b);
  b.mean = centred;
  EXPECT_GE(gram_error(b), without + centred);
}

TEST(GramError, TheBoundGrowsWithTheRunAndTheTrace) {
  const int64_t m = gram_run_rows(wide_columns);
  const gram_bound unit{
      .runs = 1, .run_rows = m, .trace = 1.0, .diagonal = 1.0};
  const double one = gram_error(unit);
  EXPECT_GT(one, static_cast<double>(m) * 0x1p-24);
  EXPECT_LT(one, static_cast<double>(m + 6) * 0x1p-24);
  EXPECT_GT(gram_error({.runs = 1,
                        .run_rows = m,
                        .trace = static_cast<double>(wide_columns),
                        .diagonal = 1.0}),
            static_cast<double>(wide_columns) * one * (1 - 0x1p-4));
  EXPECT_GT(
      gram_error({.runs = 1, .run_rows = 2 * m, .trace = 1.0, .diagonal = 1.0}),
      2 * static_cast<double>(m) * 0x1p-24);
}

TEST(GramError, NarrowAndWideKernelsNameTheirRuns) {
  EXPECT_EQ(gram_run_rows(1), cumlrkt::glm::ritz::narrow_run_rows);
  EXPECT_EQ(gram_run_rows(cumlrkt::glm::ritz::narrow_gram_columns),
            cumlrkt::glm::ritz::narrow_run_rows);
  EXPECT_LE(gram_run_rows(wide_columns), cumlrkt::glm::ritz::gram_block_rows);
}

}  // namespace
