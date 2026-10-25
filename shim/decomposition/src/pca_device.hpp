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
};

template <typename T>
void to_columns(const raft::handle_t& handle, const T* x, int64_t n, int64_t d,
                const T* variance, T* out);

template <typename T>
void to_rows(const raft::handle_t& handle, const T* x, int64_t n, int64_t d,
             const T* variance, T* out);

template <typename T>
void post_fit(const raft::handle_t& handle, const T* components, T* variance,
              int64_t m, int64_t k, int64_t d, int64_t n,
              const fit_arrays<T>& out, double* totals);

}  // namespace cumlrkt::decomposition
