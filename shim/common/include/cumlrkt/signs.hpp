#pragma once

#include <cstdint>
#include <raft/core/handle.hpp>

namespace cumlrkt {

template <typename T>
void columns_sign_rule(const raft::handle_t& handle, T* x, int64_t n,
                       int64_t k);

}  // namespace cumlrkt
