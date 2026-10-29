#pragma once

#include <cstdint>

#include "cumlrkt/limits.hpp"

namespace cumlrkt::cluster {

using cumlrkt::most_index;

inline constexpr int64_t hdbscan_least_rows = 2;
inline constexpr int64_t hdbscan_least_cluster_size = 2;
inline constexpr int64_t hdbscan_most_rows = int64_t{1} << 30;
inline constexpr int64_t hdbscan_most_min_samples = 1023;

}  // namespace cumlrkt::cluster
