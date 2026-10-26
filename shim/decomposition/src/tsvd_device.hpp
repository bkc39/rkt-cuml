#pragma once

#include <cstdint>
#include <raft/core/handle.hpp>

namespace cumlrkt::decomposition {

template <typename T>
struct tsvd_arrays {
  T* components;
  T* singular;
  T* variance;
  T* ratio;
  T* transform;
};

struct tsvd_extents {
  int64_t n;
  int64_t d;
  int64_t k;
  bool constant;
  int exponent;
};

template <typename T>
void tsvd_post_fit(const raft::handle_t& handle, const tsvd_arrays<T>& fitted,
                   const tsvd_extents& e, bool* flipped,
                   const tsvd_arrays<T>& out);

template <typename T>
double squared_frobenius(const raft::handle_t& handle, const T* x,
                         int64_t count);

}  // namespace cumlrkt::decomposition
