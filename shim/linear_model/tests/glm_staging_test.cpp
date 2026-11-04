#include "glm_staging.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <string>
#include <vector>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "fit_checks.hpp"
#include "linear_model_fixture.hpp"
#include "raftrkt/view.hpp"

namespace {

using cumlrkt::matrix;
using cumlrkt::glm::constant_columns;
using cumlrkt::glm::exponents;
using cumlrkt::glm::ols_workspace;
using cumlrkt::glm::plan_staging;
using cumlrkt::glm::require_finite_solution;
using cumlrkt::glm::require_workspace;
using cumlrkt::glm::staging;
using cumlrkt::glm::unstage_coef;
using cumlrkt::test::filled;
using cumlrkt::test::resources;

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
        h, CUML_LINEAR_MODEL_OLS_SVD, {n, d});
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

  staging planned(const double* w, exponents rule) const {
    return plan_staging(handle_of(res), matrix_of(x, 4, 2), data_of(y), w, true,
                        rule);
  }
};

TEST(PlanStaging, ColumnAndTargetExponentsIgnoreZeroWeightRows) {
  CUMLRKT_REQUIRE_GPU();
  const staging_case c;
  const filled<double> w(c.res, {0.375, 3.0, 0.0, 1.0}, {4});
  const staging s = c.planned(data_of(w), exponents::per_column);
  EXPECT_EQ(s.column_exponent, (std::vector<int>{0, -3}));
  EXPECT_EQ(s.y_exponent, -2);
}

TEST(PlanStaging, WeightExponentIsEven) {
  CUMLRKT_REQUIRE_GPU();
  const staging_case c;
  const filled<double> w(c.res, {0.375, 3.0, 0.0, 1.0}, {4});
  EXPECT_EQ(c.planned(data_of(w), exponents::per_column).weight_exponent, -2);
  const filled<double> w2(c.res, {0.375, 0.75, 0.0, 1.0}, {4});
  EXPECT_EQ(c.planned(data_of(w2), exponents::per_column).weight_exponent, 0);
}

TEST(PlanStaging, UniformExponentIsSharedByEveryColumn) {
  CUMLRKT_REQUIRE_GPU();
  const staging_case c;
  const staging u =
      c.planned(static_cast<const double*>(nullptr), exponents::uniform);
  EXPECT_EQ(u.column_exponent, (std::vector<int>{-99, -99}));
  EXPECT_EQ(u.weight_exponent, 0);
}

TEST(ConstantColumns, FollowTheInterceptRule) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const raft::handle_t& h = handle_of(res);
  const filled<float> x(res, {2, 0, 1, 2, 0, 2, 2, 0, 3, 7, 5, 4}, {4, 3});
  const filled<float> w(res, {1, 1, 1, 0}, {4});
  EXPECT_EQ(constant_columns(h, matrix_of(x, 4, 3),
                             static_cast<const float*>(nullptr), true),
            (std::vector<char>{0, 0, 0}));
  EXPECT_EQ(constant_columns(h, matrix_of(x, 4, 3), data_of(w), true),
            (std::vector<char>{1, 1, 0}));
  EXPECT_EQ(constant_columns(h, matrix_of(x, 4, 3), data_of(w), false),
            (std::vector<char>{0, 1, 0}));
  const filled<float> later(res, {0, 0, 0, 1}, {4});
  EXPECT_EQ(constant_columns(h, matrix_of(x, 4, 3), data_of(later), true),
            (std::vector<char>{1, 1, 1}));
}

constexpr float poison = 9.0F;
constexpr int past_float_range = 130;

TEST(UnstageCoef, ZerosDroppedColumnsAndReportsTheFirstOverflow) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const raft::handle_t& h = handle_of(res);
  staging s{{3, 0, -2, 1}, 1, 0, {0, 2, 3}};
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

}  // namespace
