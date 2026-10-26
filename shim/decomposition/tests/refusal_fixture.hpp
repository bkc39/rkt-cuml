#pragma once

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "cumlrkt/decomposition.h"
#include "decomposition_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {

constexpr std::size_t spread_modulus = 11;

constexpr double inf = std::numeric_limits<double>::infinity();
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

struct refusal {
  std::function<int()> call;
  std::string message;
};

inline void expect_refusals(const std::vector<refusal>& cases) {
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

}  // namespace cumlrkt::test
