#pragma once

#include <cmath>
#include <limits>

namespace cumlrkt {

template <typename T>
inline constexpr int least_unscaled_exponent =
    ((std::numeric_limits<T>::min_exponent - 1) / 2) +
    std::numeric_limits<T>::digits - 1;

template <typename T>
int scale_exponent(double largest) {
  return !(largest > 0) ||
                 largest >= std::ldexp(1.0, least_unscaled_exponent<T>)
             ? 0
             : -std::ilogb(largest);
}

}  // namespace cumlrkt
