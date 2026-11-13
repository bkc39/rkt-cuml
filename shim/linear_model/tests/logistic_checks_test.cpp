#include "logistic_checks.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <string>
#include <vector>

#include "cumlrkt/support.hpp"
#include "linear_model_fixture.hpp"
#include "raftrkt/view.hpp"

namespace {

using cumlrkt::test::filled;
using cumlrkt::test::resources;

constexpr double nan = std::numeric_limits<double>::quiet_NaN();
constexpr double inf = std::numeric_limits<double>::infinity();

template <typename T>
std::string refusal_of(const resources& res, const std::vector<T>& values,
                       void (*check)(const T*, int64_t,
                                     rmm::cuda_stream_view)) {
  const filled<T> v(res, values, {static_cast<int64_t>(values.size())});
  const auto stream =
      raft::resource::get_cuda_stream(raftrkt::handle_of(res.handle));
  try {
    check(static_cast<const T*>(v.array.view.data),
          static_cast<int64_t>(values.size()), stream);
  } catch (const cumlrkt::logic_error& e) {
    return e.what();
  }
  return "accepted";
}

TEST(LogisticSolution, NonFiniteParametersAreRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  struct row {
    std::vector<double> values;
    std::string message;
  };
  const std::vector<row> rows{
      {{1, -2, 0.5}, "accepted"},
      {{1, nan, 0.5}, "X: the qn solve gave a non-finite coefficient"},
      {{1, -2, -inf}, "X: the qn solve gave a non-finite coefficient"},
  };
  for (const row& r : rows) {
    EXPECT_EQ(
        refusal_of<double>(res, r.values,
                           cumlrkt::glm::logistic::require_finite_solution),
        r.message);
  }
}

TEST(LogisticSolution, CoefficientsOutsideTheDtypeAreRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const std::vector<float> finite{1, -2};
  const std::vector<float> overflowed{1,
                                      std::numeric_limits<float>::infinity()};
  EXPECT_EQ(refusal_of<float>(res, finite,
                              cumlrkt::glm::logistic::require_coef_in_range),
            "accepted");
  EXPECT_EQ(refusal_of<float>(res, overflowed,
                              cumlrkt::glm::logistic::require_coef_in_range),
            "X: a coefficient exceeds the float32 range");
}

}  // namespace
