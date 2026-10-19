#pragma once

#include <bit>
#include <cstdint>
#include <type_traits>

namespace cumlrkt::cluster::detail {

constexpr double widest_cosine = 2;
constexpr double widest_target = 8;
constexpr double eps_ceiling = 3;
static_assert(eps_ceiling * eps_ceiling >= widest_target);

template <typename T>
T cosine_target(double eps) {
  return static_cast<T>(eps >= widest_cosine ? widest_target : 2 * eps);
}

template <typename T>
T cosine_eps(double eps) {
  using U = std::conditional_t<sizeof(T) == sizeof(std::uint32_t),
                               std::uint32_t, std::uint64_t>;
  const T target = cosine_target<T>(eps);
  U lo = 0;
  U hi = std::bit_cast<U>(static_cast<T>(eps_ceiling));
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

}  // namespace cumlrkt::cluster::detail
