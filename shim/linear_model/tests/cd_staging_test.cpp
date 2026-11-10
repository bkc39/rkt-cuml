#include "cd_staging.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <string>
#include <utility>
#include <vector>

#include "cd_device.hpp"
#include "cumlrkt/support.hpp"
#include "glm_staging.hpp"
#include "glm_staging_device.hpp"
#include "linear_model_fixture.hpp"
#include "raftrkt/view.hpp"
#include "residual_check.hpp"

namespace {

using cumlrkt::matrix;
using cumlrkt::glm::absolute_stop;
using cumlrkt::glm::cd_centres;
using cumlrkt::glm::cd_epochs;
using cumlrkt::glm::cd_scale_of;
using cumlrkt::glm::cd_shared_cap;
using cumlrkt::glm::centre;
using cumlrkt::glm::kkt_of;
using cumlrkt::glm::kkt_terms;
using cumlrkt::glm::moment;
using cumlrkt::glm::require_cd_rows;
using cumlrkt::glm::require_kkt;
using cumlrkt::glm::require_resolved;
using cumlrkt::glm::residual_terms;
using cumlrkt::glm::row_selection;
using cumlrkt::glm::scaled_sum;
using cumlrkt::glm::stage_weighted;
using cumlrkt::glm::staging;
using cumlrkt::glm::to_host;
using cumlrkt::test::filled;
using cumlrkt::test::resources;

constexpr int64_t rows = 100;

struct planned_case {
  std::vector<int> exponent;
  std::vector<double> largest;
  int y_exponent;
};

staging planned(const planned_case& c) {
  staging s;
  s.column_exponent = c.exponent;
  s.largest = c.largest;
  s.y_exponent = c.y_exponent;
  for (std::size_t j = 0; j < s.largest.size(); ++j) {
    s.kept.push_back(static_cast<int64_t>(j));
  }
  return s;
}

struct cap_case {
  const char* binding;
  planned_case plan;
  int cap;
};

const std::array<cap_case, 3> float_caps{{
    {"(a) the columns' squared spread", {{0}, {1.5}, 10}, 57},
    {"(b) the cross products", {{0}, {1.5}, -10}, 52},
    {"(c) m times y", {{0}, {0x1p-40}, -100}, 17},
}};

TEST(CdStaging, EachOverflowCapBindsInTurn) {
  for (const cap_case& c : float_caps) {
    EXPECT_EQ(cd_shared_cap<float>(planned(c.plan), rows), c.cap) << c.binding;
  }
}

TEST(CdStaging, TheColumnCapIsNeverBelowCapA) {
  const std::array<double, 3> largest{0x1p-60, 1.5, 0x1p60};
  for (const double l : largest) {
    const staging s = planned({{0}, {l}, 0});
    const int columns =
        cumlrkt::unit_exponent(l) + cumlrkt::glm::column_reach<float>(rows);
    EXPECT_GT(columns, cd_shared_cap<float>(s, rows)) << l;
  }
}

struct scale_case {
  planned_case plan;
  double alpha;
  int shared;
};

const std::array<scale_case, 3> float_scales{{
    {{{2, 5}, {1.5, 1.5}, 10}, 0.75, 5},
    {{{19, 20}, {0x1p-40, 0x1p-40}, -100}, 0.75, 17},
    {{{16, 21}, {0x1p-40, 0x1p-40}, -100}, 0.75, 17},
}};

TEST(CdStaging, OneExponentPutsTheNarrowestColumnAtUnitSpreadUnderTheCaps) {
  for (const scale_case& c : float_scales) {
    const auto r = cd_scale_of<float>(planned(c.plan), rows, c.alpha);
    EXPECT_EQ(r.shared, c.shared);
    EXPECT_EQ(r.alpha, static_cast<float>(std::ldexp(c.alpha, 2 * c.shared)));
  }
}

template <typename F>
std::string refusal_of(F f) {
  try {
    f();
  } catch (const cumlrkt::logic_error& e) {
    return e.what();
  }
  return "accepted";
}

TEST(CdStaging, AColumnMoreThanFourBitsTooNarrowIsRefused) {
  const staging s = planned({{19, 22}, {0x1p-40, 0x1p-40}, -100});
  EXPECT_EQ(refusal_of([&] { cd_scale_of<float>(s, rows, 1.0); }),
            "X: column 1 is too narrow beside the other columns and y for "
            "float32 coordinate descent");
}

constexpr double overflowing_alpha = 1e38;
constexpr double fitting_alpha = 1e35;

TEST(CdStaging, AnAlphaWhoseRowMultipleOverflowsIsRefused) {
  const staging s = planned({{0}, {1.5}, 0});
  const std::string got =
      refusal_of([&] { cd_scale_of<float>(s, rows, overflowing_alpha); });
  EXPECT_EQ(got.rfind("alpha: expected at most ", 0), 0U) << got;
  EXPECT_NE(got.find(" for this X in float32, got 1e+38"), std::string::npos)
      << got;
  EXPECT_EQ(refusal_of([&] { cd_scale_of<float>(s, rows, fitting_alpha); }),
            "accepted");
}

constexpr double past_half_range = 1.71e36;
constexpr double under_half_range = 1.70e36;

TEST(CdStaging, AnAlphaWhoseRowMultiplePassesHalfTheRangeIsRefused) {
  const staging s = planned({{0}, {1.5}, 0});
  EXPECT_EQ(
      refusal_of([&] { cd_scale_of<float>(s, rows, past_half_range); }),
      "alpha: expected at most 1.7014118346046924e+36 for this X in float32, "
      "got "
      "1.71e+36");
  EXPECT_EQ(refusal_of([&] { cd_scale_of<float>(s, rows, under_half_range); }),
            "accepted");
}

template <typename T>
double printed_largest_alpha(const staging& s) {
  const std::string got = refusal_of(
      [&] { cd_scale_of<T>(s, rows, std::numeric_limits<double>::max()); });
  const std::string lead = "alpha: expected at most ";
  return std::stod(got.substr(lead.size(), got.find(" for ") - lead.size()));
}

TEST(CdStaging, ThePrintedLargestAlphaIsAccepted) {
  const staging s = planned({{0}, {1.5}, 0});
  EXPECT_EQ(printed_largest_alpha<double>(s),
            std::nextafter(std::ldexp(1.0 / rows, 1023), 0.0));
  EXPECT_EQ(refusal_of([&] {
              cd_scale_of<double>(s, rows, printed_largest_alpha<double>(s));
            }),
            "accepted");
  EXPECT_EQ(refusal_of([&] {
              cd_scale_of<float>(s, rows, printed_largest_alpha<float>(s));
            }),
            "accepted");
}

scaled_sum sum(double value, int exponent) {
  return {value, exponent};
}

constexpr double first_cross = 1.53125;
constexpr double second_cross = 1.25;
constexpr double dropped_cross = 1.5625;
constexpr int dropped_exponent = 6;

residual_terms hand_built_epoch() {
  return {{moment{sum(1, 2), sum(first_cross, 1)},
           moment{sum(1, 0), sum(second_cross, 1)},
           moment{sum(0, 0), sum(dropped_cross, dropped_exponent)}},
          {1, 1, 0},
          sum(1, 4)};
}

const std::vector<double> epoch_coef{1, 0, 0};
constexpr double epoch_weight = 4;
constexpr double epoch_tol = 0.1;
constexpr double epoch_l1 = 0.5;
constexpr double light_l2 = 0.25;
constexpr double heavy_l2 = 1;
constexpr int epoch_iters = 4;
constexpr double epoch_largest = 0.0625;
constexpr int64_t epoch_rows = 9;
constexpr double float_roundoff = 0x1p-24;
constexpr double light_residual = 1.0 / 6;
constexpr double heavy_residual = 0.3671875;
constexpr double missing = std::numeric_limits<double>::quiet_NaN();
constexpr double light_allowance = float_roundoff * 2 * (2 + 3 * 2 * (2.0 / 3));
constexpr double heavy_allowance = float_roundoff * 2 * (2 + 3 * 2 * 0.5);

TEST(CdCheck, KBAndAOnAHandBuiltEpochUnderALightL2Penalty) {
  const cd_epochs epochs{epoch_iters, epoch_rows, epoch_tol, float_roundoff};
  const kkt_terms k = kkt_of(hand_built_epoch(), epoch_coef, epoch_weight,
                             {epoch_l1, light_l2}, epochs);
  EXPECT_DOUBLE_EQ(k.residual.value_or(missing), light_residual);
  EXPECT_DOUBLE_EQ(k.bound, epoch_tol);
  EXPECT_DOUBLE_EQ(k.allowance, light_allowance);
}

TEST(CdCheck, KBAndAOnAHandBuiltEpochUnderAHeavyL2Penalty) {
  const cd_epochs epochs{epoch_iters, epoch_rows, epoch_tol, float_roundoff};
  const kkt_terms k = kkt_of(hand_built_epoch(), epoch_coef, epoch_weight,
                             {epoch_l1, heavy_l2}, epochs);
  EXPECT_DOUBLE_EQ(k.residual.value_or(missing), heavy_residual);
  EXPECT_DOUBLE_EQ(k.bound, epoch_tol);
  EXPECT_DOUBLE_EQ(k.allowance, heavy_allowance);
  const std::string got =
      refusal_of([&] { require_kkt(k, epochs.n_iter, epochs.tol); });
  EXPECT_EQ(got.rfind("X: coordinate descent stopped after 4 epochs with every "
                      "coefficient under tol 0.1 and KKT residual 0.3671875 > ",
                      0),
            0U)
      << got;
  EXPECT_NE(got.find("; tol is absolute in the coefficients' units, so scale "
                     "y up or X down"),
            std::string::npos);
}

TEST(CdCheck, AResidualWithinBPlusAIsAccepted) {
  EXPECT_EQ(refusal_of([&] {
              require_kkt({epoch_tol, epoch_tol, 0}, epoch_iters, epoch_tol);
            }),
            "accepted");
  EXPECT_NE(refusal_of([&] {
              require_kkt({std::nullopt, epoch_tol, 1}, epoch_iters, epoch_tol);
            }),
            "accepted");
}

TEST(CdCheck, AnAllowanceAboveBIsRefusedAsUncheckable) {
  const kkt_terms unresolved{epoch_tol / 2, epoch_tol, 2 * epoch_tol};
  EXPECT_EQ(refusal_of([&] {
              require_resolved<float>(unresolved, epoch_iters, epoch_tol,
                                      epoch_largest);
            }),
            "X: coordinate descent stopped after 4 epochs with every "
            "coefficient under tol 0.1, and float32 cannot check that stop; "
            "lower tol below the largest coefficient (0.0625) or fit in "
            "float64");
  EXPECT_EQ(refusal_of([&] {
              require_resolved<double>(unresolved, epoch_iters, epoch_tol,
                                       epoch_largest);
            }),
            "X: coordinate descent stopped after 4 epochs with every "
            "coefficient under tol 0.1, and float64 cannot check that stop; "
            "lower tol below the largest coefficient (0.0625)");
  EXPECT_EQ(refusal_of([&] {
              require_resolved<float>({epoch_tol / 2, epoch_tol, epoch_tol},
                                      epoch_iters, epoch_tol, epoch_largest);
            }),
            "accepted");
}

struct stop_case {
  int n_iter;
  int max_iter;
  double largest;
  bool checked;
};

const std::array<stop_case, 5> stops{{
    {3, 10, 1e-4, true},
    {10, 10, 1e-4, false},
    {3, 10, 0.0, false},
    {3, 10, 1e-3, false},
    {3, 10, 1.0, false},
}};

TEST(CdCheck, OnlyStopsTheAbsoluteTestCouldMakeAreChecked) {
  for (const stop_case& c : stops) {
    EXPECT_EQ(absolute_stop(c.n_iter, c.max_iter, c.largest, 1e-3), c.checked)
        << c.n_iter << " " << c.largest;
  }
}

constexpr int64_t most_rows = std::numeric_limits<int32_t>::max();

TEST(CdStaging, RowsPastInt32AreRefused) {
  EXPECT_EQ(refusal_of([] { require_cd_rows(most_rows + 1); }),
            "X: 2147483648 rows of positive weight exceed coordinate "
            "descent's 2147483647");
  EXPECT_EQ(refusal_of([] { require_cd_rows(most_rows); }), "accepted");
}

constexpr int64_t dyadic_rows = 64;
constexpr std::array<std::pair<float, int64_t>, 3> dyadic_weights{
    {{1.0F, 32}, {4.0F, 24}, {16.0F, 8}}};
constexpr std::array<float, 3> dyadic_roots{0.5F, 1.0F, 2.0F};

TEST(CdWeighting, RootWeightsAreExactOnTheDyadicCase) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  std::vector<float> w;
  std::vector<float> expected;
  for (std::size_t g = 0; g < dyadic_weights.size(); ++g) {
    for (int64_t i = 0; i < dyadic_weights[g].second; ++i) {
      w.push_back(dyadic_weights[g].first);
      expected.push_back(dyadic_roots[g]);
    }
  }
  const std::vector<float> ones(dyadic_rows, 1.0F);
  const filled<float> x(res, ones, {dyadic_rows, 1});
  const filled<float> y(res, ones, {dyadic_rows});
  const filled<float> weights(res, w, {dyadic_rows});
  const raft::handle_t& h = raftrkt::handle_of(res.handle);
  const auto stream = raft::resource::get_cuda_stream(h);
  const staging s = planned({{0}, {1.0}, 0});
  const cd_centres none{{0.0}, {0.0}, centre{0.0, 0.0}};
  const auto p = stage_weighted(
      h,
      matrix<float, int64_t>{static_cast<float*>(x.array.view.data),
                             dyadic_rows, 1},
      static_cast<const float*>(y.array.view.data),
      static_cast<const float*>(weights.array.view.data),
      row_selection{nullptr, dyadic_rows}, s, none,
      cumlrkt::glm::weight_total(
          static_cast<const float*>(weights.array.view.data),
          row_selection{nullptr, dyadic_rows}, stream));
  EXPECT_EQ(to_host(p.x, stream), expected);
  EXPECT_EQ(to_host(p.y, stream), expected);
}

}  // namespace
