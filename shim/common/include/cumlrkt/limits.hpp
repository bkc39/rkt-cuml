#pragma once

#include <cstdint>
#include <limits>

namespace cumlrkt {

inline constexpr int64_t most_index = std::numeric_limits<int32_t>::max();
inline constexpr double most_squared_norm = 0x1p125;
inline constexpr double most_squared_norm_double = 0x1p1021;

template <typename T>
inline constexpr double most_value = std::numeric_limits<T>::max();

inline constexpr double most_float = most_value<float>;

template <typename T>
inline constexpr double squared_norm_bound =
    sizeof(T) == sizeof(float) ? most_squared_norm : most_squared_norm_double;

}  // namespace cumlrkt
