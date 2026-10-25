#pragma once

#include <cstdint>
#include <raft/core/handle.hpp>

namespace cumlrkt::preprocessing {

template <typename T>
int64_t first_non_finite(const T* x, int64_t count,
                         const raft::handle_t& handle);

template <typename T>
void column_statistics(const raft::handle_t& handle, const T* x, int64_t n,
                       int64_t d, int64_t chunk, double* scratch,
                       double* residual, double* mean, double* var);

void scales_of(const raft::handle_t& handle, const double* mean,
               const double* var, double* scale, int64_t n, int64_t d);

template <typename T>
void standardized(const raft::handle_t& handle, const T* x, const double* mean,
                  const double* scale, T* out, int64_t n, int64_t d);

template <typename T>
void unstandardized(const raft::handle_t& handle, const T* x,
                    const double* mean, const double* scale, T* out, int64_t n,
                    int64_t d);

}  // namespace cumlrkt::preprocessing
