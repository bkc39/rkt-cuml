#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace cumlrkt::cluster::detail {

constexpr double widest_cosine = 4;

template <typename T>
T cosine_target(double eps) {
  return static_cast<T>(2 * std::min(eps, widest_cosine));
}

template <typename T>
T cosine_eps(double eps) {
  const T target = cosine_target<T>(eps);
  T e = static_cast<T>(std::sqrt(static_cast<double>(target)));
  while (e * e < target) {
    e = std::nextafter(e, std::numeric_limits<T>::infinity());
  }
  for (T p = std::nextafter(e, T{0}); p * p >= target;
       p = std::nextafter(p, T{0})) {
    e = p;
  }
  return e;
}

}  // namespace cumlrkt::cluster::detail
