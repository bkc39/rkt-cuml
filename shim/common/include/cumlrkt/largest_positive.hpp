#pragma once

#include <cstdint>
#include <raft/core/handle.hpp>

namespace cumlrkt {

template <typename T>
void make_largest_positive(const raft::handle_t& handle, T* x, int64_t n,
                           int64_t k);

}  // namespace cumlrkt
