#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <raft/core/cublas_macros.hpp>
#include <raft/core/cusolver_macros.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cusolver_dn_handle.hpp>

#include "cumlrkt/support.hpp"

namespace cumlrkt::glm {

inline void blas(cublasStatus_t result) {
  RAFT_CUBLAS_TRY(result);
}

inline void solved(cusolverStatus_t result) {
  RAFT_CUSOLVER_TRY(result);
}

inline cublasHandle_t cublas_on(const raft::handle_t& handle) {
  cublasHandle_t cublas = raft::resource::get_cublas_handle(handle);
  blas(cublasSetStream(cublas, stream_of(handle).value()));
  return cublas;
}

inline cusolverDnHandle_t cusolver_on(const raft::handle_t& handle) {
  cusolverDnHandle_t solver = raft::resource::get_cusolver_dn_handle(handle);
  solved(cusolverDnSetStream(solver, stream_of(handle).value()));
  return solver;
}

}  // namespace cumlrkt::glm
