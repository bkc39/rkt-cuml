#include "gamma.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <raft/core/handle.hpp>
#include <string>
#include <vector>

#include "cumlrkt/support.hpp"
#include "svm_fixture.hpp"

namespace cumlrkt::test {
namespace {

using cumlrkt::svm::element_variance;
using cumlrkt::svm::scale_gamma;

constexpr int64_t rows = 64;
constexpr int64_t cols = 4;

template <typename T>
std::vector<T> shifted_columns() {
  std::vector<T> x(static_cast<std::size_t>(rows * cols));
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      x[(i * cols) + j] = static_cast<T>((i % 8) + (16 * j));
    }
  }
  return x;
}

double exact_variance(const std::vector<double>& x) {
  double sum = 0.0;
  for (const double v : x) {
    sum += v;
  }
  const double mean = sum / static_cast<double>(x.size());
  double squares = 0.0;
  for (const double v : x) {
    squares += (v - mean) * (v - mean);
  }
  return squares / static_cast<double>(x.size());
}

template <typename T>
void expect_scale() {
  const raft::handle_t h;
  const std::vector<T> host = shifted_columns<T>();
  auto x = on_device(host, h);
  const double var =
      exact_variance(std::vector<double>(host.begin(), host.end()));
  EXPECT_EQ(static_cast<double>(
                element_variance(x.data(), rows * cols, stream_of(h))),
            var);
  EXPECT_EQ(scale_gamma(x.data(), rows, cols, stream_of(h)),
            1.0 / (static_cast<double>(cols) * var));
}

TEST(gamma, G1_scale_is_over_every_element_float) {
  CUMLRKT_REQUIRE_GPU();
  expect_scale<float>();
}

TEST(gamma, G1_scale_is_over_every_element_double) {
  CUMLRKT_REQUIRE_GPU();
  expect_scale<double>();
}

TEST(gamma, a_constant_X_is_refused) {
  CUMLRKT_REQUIRE_GPU();
  const raft::handle_t h;
  auto x = on_device(std::vector<float>(rows * cols, 2.5F), h);
  try {
    scale_gamma(x.data(), rows, cols, stream_of(h));
    FAIL() << "accepted";
  } catch (const logic_error& e) {
    EXPECT_EQ(std::string(e.what()),
              "gamma 'scale: X has zero variance; pass a numeric #:gamma");
  }
}

}  // namespace
}  // namespace cumlrkt::test
