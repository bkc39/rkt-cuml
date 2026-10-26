#pragma once

#include <cstdint>
#include <raft/core/handle.hpp>

namespace cumlrkt::decomposition {

template <typename T>
struct fit_arrays {
  T* components;
  T* variance;
  T* ratio;
  T* singular;
  T* mean;
};

template <typename T>
void post_fit(const raft::handle_t& handle, const T* components, T* variance,
              int64_t m, int64_t k, int64_t d, int64_t n,
              const fit_arrays<T>& out, double* totals, int exponent);

struct variance_bounds {
  double smallest_divisor;
  double largest_root;
};

template <typename T>
variance_bounds bounds_of(const raft::handle_t& handle, const T* variance,
                          int64_t k);

}  // namespace cumlrkt::decomposition
