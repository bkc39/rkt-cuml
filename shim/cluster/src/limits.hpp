#pragma once

#include <cfloat>
#include <cstdint>
#include <limits>

namespace cumlrkt::cluster {

inline constexpr int64_t most_index = std::numeric_limits<int32_t>::max();
inline constexpr double most_squared_norm = 0x1p125;
inline constexpr double most_float = FLT_MAX;

inline constexpr int64_t hdbscan_least_rows = 2;
inline constexpr int64_t hdbscan_least_cluster_size = 2;
inline constexpr int64_t hdbscan_most_rows = int64_t{1} << 30;
inline constexpr int64_t hdbscan_most_min_samples = 1023;

}  // namespace cumlrkt::cluster
