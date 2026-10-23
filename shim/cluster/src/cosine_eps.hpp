#pragma once

#include <bit>
#include <cstdint>
#include <type_traits>

namespace cumlrkt::cluster {

constexpr double max_cosine_distance = 2;
constexpr double squared_past_every_unit_pair = 8;
constexpr double bisection_upper_bound = 3;
static_assert(bisection_upper_bound * bisection_upper_bound >=
              squared_past_every_unit_pair);

template <typename T>
T cosine_target(double eps) {
  return static_cast<T>(
      eps >= max_cosine_distance ? squared_past_every_unit_pair : 2 * eps);
}

template <typename T>
T cosine_eps(double eps) {
  using U = std::conditional_t<sizeof(T) == sizeof(std::uint32_t),
                               std::uint32_t, std::uint64_t>;
  const T target = cosine_target<T>(eps);
  U lo = 0;
  U hi = std::bit_cast<U>(static_cast<T>(bisection_upper_bound));
  while (lo < hi) {
    const U mid = lo + (hi - lo) / 2;
    const T m = std::bit_cast<T>(mid);
    if (m * m >= target) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return std::bit_cast<T>(lo);
}

}  // namespace cumlrkt::cluster
