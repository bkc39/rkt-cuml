#include "glm_staging.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <string>
#include <vector>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/rows.hpp"
#include "cumlrkt/support.hpp"
#include "fit_checks.hpp"
#include "glm_staging_device.hpp"
#include "linear_model_fixture.hpp"
#include "raftrkt/view.hpp"
#include "residual_check.hpp"

namespace {

using cumlrkt::matrix;
using cumlrkt::glm::column_profile;
using cumlrkt::glm::column_profiles;
using cumlrkt::glm::constant_vector;
using cumlrkt::glm::cosine_of;
using cumlrkt::glm::moment;
using cumlrkt::glm::ols_workspace;
using cumlrkt::glm::plan_staging;
using cumlrkt::glm::profiles;
using cumlrkt::glm::require_finite_solution;
using cumlrkt::glm::row_selection;
using cumlrkt::glm::scaled_sum;
using cumlrkt::glm::spread_exponent;
using cumlrkt::glm::staged_weights;
using cumlrkt::glm::staging;
using cumlrkt::glm::stationarity_of;
using cumlrkt::glm::to_host;
using cumlrkt::glm::uniform;
using cumlrkt::glm::unstage_coef;
using cumlrkt::test::filled;
using cumlrkt::test::resources;

void require_workspace(int algorithm, int64_t n, int64_t d, int64_t lwork) {
  cumlrkt::glm::require_workspace({cumlrkt::glm::glm_family::ols, algorithm},
                                  {n, d}, lwork);
}

std::string refusal_of(void (*f)(int, int64_t, int64_t, int64_t), int a,
                       int64_t n, int64_t d, int64_t lwork) {
  try {
    f(a, n, d, lwork);
  } catch (const cumlrkt::logic_error& e) {
    return e.what();
  }
  return "accepted";
}

const raft::handle_t& handle_of(const resources& res) {
  return raftrkt::handle_of(res.handle);
}

template <typename T>
matrix<T, int64_t> matrix_of(const filled<T>& a, int64_t n, int64_t d) {
  return {static_cast<T*>(a.array.view.data), n, d};
}

template <typename T>
const T* data_of(const filled<T>& a) {
  return static_cast<const T*>(a.array.view.data);
}

TEST(OlsWorkspace, EigIsThreeSquaresAndTwoColumns) {
  EXPECT_EQ(ols_workspace(CUML_LINEAR_MODEL_OLS_EIG, {10, 4}, 0), (3 * 16) + 8);
  EXPECT_EQ(ols_workspace(CUML_LINEAR_MODEL_OLS_EIG, {30000, 26755}, 0),
            2147543585);
  EXPECT_EQ(
      refusal_of(require_workspace, CUML_LINEAR_MODEL_OLS_EIG, 30000, 26754, 0),
      "accepted");
  EXPECT_EQ(
      refusal_of(require_workspace, CUML_LINEAR_MODEL_OLS_EIG, 30000, 26755, 0),
      "X: 26755 columns need a 2147543585-element eig workspace, more "
      "than 2147483647 elements");
}

TEST(OlsWorkspace, SvdCountsTheQueriedWorkspaceAndBothFactors) {
  EXPECT_EQ(ols_workspace(CUML_LINEAR_MODEL_OLS_SVD, {10, 4}, 7),
            7 + 40 + 16 + 8 + 1);
  EXPECT_EQ(ols_workspace(CUML_LINEAR_MODEL_OLS_SVD_QR, {10, 4}, 7),
            7 + 40 + 16 + 8 + 1);
  EXPECT_EQ(ols_workspace(CUML_LINEAR_MODEL_OLS_SVD_QR, {4, 10}, 7),
            7 + 16 + 100 + 8 + 1);
  EXPECT_EQ(refusal_of(require_workspace, CUML_LINEAR_MODEL_OLS_SVD, 2,
                       1073741800, 64),
            "X: 1073741800 columns need a 2147483673-element svd workspace, "
            "more than 2147483647 elements");
  EXPECT_EQ(refusal_of(require_workspace, CUML_LINEAR_MODEL_OLS_SVD_QR, 50000,
                       30000, 0),
            "X: 30000 columns need a 2400060001-element svd-qr workspace, "
            "more than 2147483647 elements");
}

TEST(OlsWorkspace, CusolverWorkspacesAtTheSvdBoundaries) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const raft::handle_t& h = handle_of(res);
  const int64_t n = 40000;
  for (const int64_t d : {int64_t{20000}, int64_t{26000}}) {
    const int64_t lwork = cumlrkt::glm::cusolver_lwork_of<float>(
        h, {cumlrkt::glm::glm_family::ols, CUML_LINEAR_MODEL_OLS_SVD}, {n, d});
    EXPECT_GE(lwork, 0) << d;
    const int64_t size =
        ols_workspace(CUML_LINEAR_MODEL_OLS_SVD, {n, d}, lwork);
    EXPECT_EQ(refusal_of(require_workspace, CUML_LINEAR_MODEL_OLS_SVD, n, d,
                         lwork) == "accepted",
              size <= cumlrkt::most_index)
        << d << " " << lwork;
  }
}

struct staging_case {
  resources res;
  filled<double> x{res, {0.75, 1.0, 1.5, 3.0, 1e30, 1e30, 0.5, 9.0}, {4, 2}};
  filled<double> y{res, {1.0, 5.0, 1e30, 3.0}, {4}};

  staging planned(const double* w) const {
    return plan_staging(handle_of(res), matrix_of(x, 4, 2), data_of(y), w,
                        row_selection{nullptr, 4}, true);
  }
};

TEST(PlanStaging, ColumnAndTargetExponentsIgnoreZeroWeightRows) {
  CUMLRKT_REQUIRE_GPU();
  const staging_case c;
  const filled<double> w(c.res, {0.375, 3.0, 0.0, 1.0}, {4});
  const staging s = c.planned(data_of(w));
  EXPECT_EQ(s.column_exponent, (std::vector<int>{1, -2}));
  EXPECT_EQ(s.y_exponent, -2);
}

TEST(PlanStaging, ColumnExponentFollowsTheWeightedSpread) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const double low = std::ldexp(1.0, -40);
  const filled<double> x(
      res, {1e6, 0.0, 1e6 + 1, 0.0, 1e6 + 2, 0.0, 1e6 + 3, 1.5}, {4, 2});
  const filled<double> y(res, {1.0, 2.0, 3.0, 4.0}, {4});
  const filled<double> w(res, {1.0, 1.0, 1.0, low}, {4});
  const auto planned = [&](const double* weights, bool fit_intercept) {
    return plan_staging(handle_of(res), matrix_of(x, 4, 2), data_of(y), weights,
                        row_selection{nullptr, 4}, fit_intercept)
        .column_exponent;
  };
  EXPECT_EQ(planned(data_of(w), true), (std::vector<int>{0, 20}));
  EXPECT_EQ(planned(nullptr, false), (std::vector<int>{-20, 0}));
}

TEST(SpreadExponent, KeepsTheColumnSumFiniteAndFallsBackOnAZeroSpread) {
  EXPECT_EQ(spread_exponent<float>(-10, std::ldexp(1.0, -200), 4), 113);
  EXPECT_EQ(spread_exponent<double>(-10, std::ldexp(1.0, -200), 4), 190);
  EXPECT_EQ(spread_exponent<float>(-10, 0.0, 4), -10);
  EXPECT_EQ(spread_exponent<double>(3, 6.0, 4), 1);
}

struct staged_terms {
  int exponent;
  int64_t count;
};

template <typename T>
bool weighted_column_sum_finite(staged_terms terms) {
  const T below_two = std::nextafter(T{2}, T{0});
  const T largest = std::ldexp(below_two, terms.exponent);
  T sum{0};
  for (int64_t i = 0; i < terms.count; ++i) {
    sum += below_two * largest;
  }
  return std::isfinite(sum);
}

template <typename T>
void expect_cap_at_the_boundary() {
  constexpr int64_t n = 7;
  const int cap = spread_exponent<T>(0, std::ldexp(1.0, -1070), n);
  EXPECT_TRUE(weighted_column_sum_finite<T>({cap, n})) << cap;
  EXPECT_FALSE(weighted_column_sum_finite<T>({cap + 1, n})) << cap;
}

TEST(SpreadExponent, CapKeepsNStagedWeightTimesColumnTermsFinite) {
  expect_cap_at_the_boundary<float>();
  expect_cap_at_the_boundary<double>();
}

template <typename T>
std::vector<T> weights_after_staging(const resources& res, const filled<T>& w) {
  const raft::handle_t& h = handle_of(res);
  const auto stream = raft::resource::get_cuda_stream(h);
  return to_host(
      staged_weights(h, data_of(w), 4,
                     cumlrkt::largest_magnitude(data_of(w), 4, stream)),
      stream);
}

TEST(StagedWeights, TakeAnEvenExponent) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> w(res, {0.375, 3.0, 0.0, 1.0}, {4});
  const std::vector<double> quartered{0.09375, 0.75, 0.0, 0.25};
  EXPECT_EQ(weights_after_staging(res, w), quartered);
  const filled<double> w2(res, {0.375, 0.75, 0.0, 1.0}, {4});
  const std::vector<double> unchanged{0.375, 0.75, 0.0, 1.0};
  EXPECT_EQ(weights_after_staging(res, w2), unchanged);
}

TEST(StagedWeights, AtMostTheCutoffBecomeZero) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const double cutoff = std::ldexp(1.0, -1073);
  const double least = std::numeric_limits<double>::denorm_min();
  const filled<double> w(res, {3.0, cutoff, cutoff + least, 1.0}, {4});
  const std::vector<double> staged{0.75, 0.0, least, 0.25};
  EXPECT_EQ(weights_after_staging(res, w), staged);
}

TEST(StagedWeights, AtMostTheFloat32CutoffBecomeZero) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const float cutoff = std::ldexp(1.0F, -148);
  const float least = std::numeric_limits<float>::denorm_min();
  const filled<float> w(res, {3.0F, cutoff, cutoff + least, 1.0F}, {4});
  const std::vector<float> staged{0.75F, 0.0F, least, 0.25F};
  EXPECT_EQ(weights_after_staging(res, w), staged);
}

TEST(PlanStaging, UniformExponentIsSharedByEveryColumn) {
  CUMLRKT_REQUIRE_GPU();
  const staging_case c;
  const staging u = uniform(c.planned(static_cast<const double*>(nullptr)));
  EXPECT_EQ(u.column_exponent, (std::vector<int>{-99, -99}));
}

std::vector<char> constant_columns(const resources& res, const filled<float>& x,
                                   const float* w, bool fit_intercept) {
  const raft::handle_t& h = handle_of(res);
  const profiles p =
      column_profiles(matrix_of(x, 4, 3), w, row_selection{nullptr, 4},
                      fit_intercept, raft::resource::get_cuda_stream(h));
  std::vector<char> constant(p.columns.size());
  std::ranges::transform(p.columns, constant.begin(), [](column_profile c) {
    return static_cast<char>(!c.varies);
  });
  return constant;
}

TEST(ColumnProfiles, ConstantColumnsFollowTheInterceptRule) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, {2, 0, 1, 2, 0, 2, 2, 0, 3, 7, 5, 4}, {4, 3});
  const filled<float> w(res, {1, 1, 1, 0}, {4});
  EXPECT_EQ(constant_columns(res, x, nullptr, true),
            (std::vector<char>{0, 0, 0}));
  EXPECT_EQ(constant_columns(res, x, data_of(w), true),
            (std::vector<char>{1, 1, 0}));
  EXPECT_EQ(constant_columns(res, x, data_of(w), false),
            (std::vector<char>{0, 1, 0}));
  const filled<float> later(res, {0, 0, 0, 1}, {4});
  EXPECT_EQ(constant_columns(res, x, data_of(later), true),
            (std::vector<char>{1, 1, 1}));
}

TEST(ConstantVector, FollowsTheInterceptRule) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const raft::handle_t& h = handle_of(res);
  const filled<float> v(res, {2, 2, 2, 7}, {4});
  const filled<float> w(res, {1, 1, 1, 0}, {4});
  EXPECT_FALSE(constant_vector(h, data_of(v), 4,
                               static_cast<const float*>(nullptr), true));
  EXPECT_TRUE(constant_vector(h, data_of(v), 4, data_of(w), true));
  EXPECT_FALSE(constant_vector(h, data_of(v), 4, data_of(w), false));
}

constexpr float poison = 9.0F;
constexpr int past_float_range = 130;

TEST(UnstageCoef, ZerosDroppedColumnsAndReportsTheFirstOverflow) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const raft::handle_t& h = handle_of(res);
  staging s{{3, 0, -2, 1}, 1, {0, 2, 3}, {}, {}, {}, {}};
  const filled<float> staged(res, {1.5F, -2.0F, 4.0F}, {3});
  filled<float> coef(res, {poison, poison, poison, poison}, {4});
  EXPECT_EQ(unstage_coef(h, data_of(staged), s, 4,
                         static_cast<float*>(coef.array.view.data)),
            -1);
  EXPECT_EQ(coef.array.read<float>(),
            (std::vector<float>{6.0F, 0.0F, -0.25F, 4.0F}));
  s.column_exponent = {0, 0, past_float_range, past_float_range - 1};
  EXPECT_EQ(unstage_coef(h, data_of(staged), s, 4,
                         static_cast<float*>(coef.array.view.data)),
            2);
}

constexpr double finite_intercept = 0.5;

std::string finite_refusal(const resources& res, const filled<double>& coef,
                           double intercept, int algorithm) {
  try {
    require_finite_solution(data_of(coef), 2, intercept, algorithm,
                            raft::resource::get_cuda_stream(handle_of(res)));
  } catch (const cumlrkt::logic_error& e) {
    return e.what();
  }
  return "accepted";
}

TEST(FiniteSolution, AFiniteSolutionIsAccepted) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> good(res, {1.0, 2.0}, {2});
  EXPECT_EQ(
      finite_refusal(res, good, finite_intercept, CUML_LINEAR_MODEL_OLS_QR),
      "accepted");
}

TEST(FiniteSolution, AStagedNanIsRefusedNamingTheSolve) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> bad(res, {1.0, std::nan("")}, {2});
  EXPECT_EQ(
      finite_refusal(res, bad, finite_intercept, CUML_LINEAR_MODEL_OLS_QR),
      "X: the qr solve gave a non-finite coefficient");
}

TEST(FiniteSolution, ANonFiniteInterceptIsRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<double> good(res, {1.0, 2.0}, {2});
  EXPECT_EQ(finite_refusal(res, good, std::numeric_limits<double>::infinity(),
                           CUML_LINEAR_MODEL_OLS_EIG),
            "X: the eig solve gave a non-finite coefficient");
}

struct stationarity_row {
  moment m;
  scaled_sum y;
  double alpha;
  double b;
  double cosine;
};

TEST(Stationarity, WithoutAPenaltyItIsTheResidualCosineBitwise) {
  const std::vector<moment> moments{
      {{0x1.4p-1, 4}, {0x1.8p-1, 3}},
      {{0x1.fffffp-1, -1021}, {-0x1.3p-1, -40}},
      {{0x1.1p-1, 900}, {0x1.7p-1, 950}},
  };
  const std::vector<scaled_sum> ys{{0x1.2p-1, 5}, {0x1.9p-1, -3}};
  const std::vector<double> coefficients{0.0, 0x1.5p-30, -0x1.5p300};
  for (const moment& m : moments) {
    for (const scaled_sum& y : ys) {
      for (const double b : coefficients) {
        EXPECT_EQ(stationarity_of(m, y, 0.0, b), cosine_of(m, y)) << b;
        EXPECT_EQ(stationarity_of(m, y, -0.0, b), cosine_of(m, y)) << b;
      }
    }
  }
}

TEST(Stationarity, ACoefficientTheSolveZeroedUnderALargePenaltyIsSeen) {
  const std::vector<stationarity_row> rows{
      {{{1.0, 0}, {0.5, 0}}, {1.0, 0}, 0x1p40, 0.0, 0.5},
      {{{1.0, 0}, {0.5, 0}}, {1.0, 0}, 0x1p40, 0x1p-41, 0.0},
      {{{1.0, 2000}, {0.5, 1050}}, {1.0, 100}, 0x1p1000, 0.0, 0.5},
      {{{1.0, 0}, {0.5, 0}}, {1.0, 0}, 1.0, 0.25, 0.2},
  };
  for (const stationarity_row& r : rows) {
    EXPECT_EQ(stationarity_of(r.m, r.y, r.alpha, r.b), r.cosine) << r.alpha;
  }
}

TEST(Stationarity, AnOddSpreadExponentSplitsBetweenTheTwoNorms) {
  const double root_half = std::sqrt(0.5);
  const std::vector<stationarity_row> rows{
      {{{1.0, 3}, {0.5, 2}}, {1.0, 0}, 0x1p-60, 0.0, root_half},
      {{{1.0, -3}, {0.5, -1}}, {1.0, 0}, 0x1p-60, 0.0, root_half},
  };
  for (const stationarity_row& r : rows) {
    EXPECT_DOUBLE_EQ(stationarity_of(r.m, r.y, r.alpha, r.b), r.cosine)
        << r.m.square.exponent;
  }
}

}  // namespace
