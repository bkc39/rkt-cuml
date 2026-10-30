#pragma once

#include <cstdint>
#include <raft/core/handle.hpp>

namespace cumlrkt {

template <typename T>
void to_columns(const raft::handle_t& handle, const T* x, int64_t n, int64_t d,
                const T* variance, T* out, int exponent);

template <typename T>
void to_rows(const raft::handle_t& handle, const T* x, int64_t n, int64_t d,
             const T* variance, T* out);

template <typename T>
void scaled_copy(const raft::handle_t& handle, const T* x, int64_t count,
                 T* out, int exponent);

}  // namespace cumlrkt
