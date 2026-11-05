#include "fit_checks.hpp"

#include <cstdint>
#include <format>
#include <functional>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cusolver_dn_handle.hpp>
#include <raft/linalg/detail/cusolver_wrappers.hpp>
#include <rmm/cuda_stream_view.hpp>

#include "cumlrkt/limits.hpp"
#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "cumlrkt/vectors.hpp"
#include "glm_staging.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt::glm {

void require_fit_intercept(int32_t fit_intercept) {
  if (fit_intercept != 0 && fit_intercept != 1) {
    throw logic_error("fit-intercept: expected #t or #f");
  }
}

void require_ols_codes(const fit_request& request) {
  if (request.algorithm < CUML_LINEAR_MODEL_OLS_SVD ||
      request.algorithm > CUML_LINEAR_MODEL_OLS_SVD_QR) {
    throw logic_error(
        "algorithm: expected auto, eig, svd, svd-jacobi, qr or svd-qr");
  }
  require_fit_intercept(request.fit_intercept);
}

int64_t workspace_of(glm_solver solver, ols_shape shape, int64_t lwork) {
  if (solver.family == glm_family::ols) {
    return ols_workspace(solver.algorithm, shape, lwork);
  }
  const auto [m, d] = shape;
  const int64_t square =
      solver.algorithm == CUML_LINEAR_MODEL_RIDGE_SVD ? d * d : 2 * d * d;
  return (m * d) + square + (3 * d) + lwork;
}

void require_workspace(glm_solver solver, ols_shape shape, int64_t lwork) {
  const int64_t size = workspace_of(solver, shape, lwork);
  if (size > most_index) {
    throw logic_error(std::format(
        "X: {} columns need a {}-element {} workspace, more than {} elements",
        shape.d, size, algorithm_name(solver.algorithm), most_index));
  }
}

template <typename T>
checked_fit<T> checked_inputs(const raft::handle_t& handle,
                              const fit_request& request,
                              const std::function<void()>& require_codes) {
  const matrix<T, int64_t> x = rows_of<T, int64_t>(
      request.x, "X", raftrkt::any_extent, raftrkt::any_extent);
  const T* y = raftrkt::vector_data<T, int32_t>(request.y, "y", x.n);
  const T* w = request.w == nullptr ? nullptr
                                    : raftrkt::vector_data<T, int32_t>(
                                          request.w, "sample-weight", x.n);
  require_some("X", "row", x.n, 2);
  require_some("X", "column", x.d);
  require_index("X", x.n, x.d);
  require_codes();
  const rmm::cuda_stream_view stream = stream_of(handle);
  require_finite("X", x, stream);
  require_finite_elements("y", y, x.n, stream);
  const double largest_weight =
      w == nullptr ? 0.0 : require_sample_weights(w, x.n, stream);
  return {{x, y, w}, largest_weight};
}

namespace {

enum class lwork_query : std::uint8_t { none, gesvdj, gesvd, syevd };

lwork_query query_of(glm_solver solver) {
  if (solver.family == glm_family::ridge) {
    return solver.algorithm == CUML_LINEAR_MODEL_RIDGE_SVD ? lwork_query::gesvd
                                                           : lwork_query::syevd;
  }
  switch (solver.algorithm) {
    case CUML_LINEAR_MODEL_OLS_SVD:
      return lwork_query::gesvdj;
    case CUML_LINEAR_MODEL_OLS_SVD_QR:
      return lwork_query::gesvd;
    default:
      return lwork_query::none;
  }
}

template <typename T>
cusolverStatus_t gesvdj_lwork(cusolverDnHandle_t solver, ols_shape shape,
                              int* lwork) {
  const auto [n, d] = shape;
  gesvdjInfo_t params = nullptr;
  if (cusolverDnCreateGesvdjInfo(&params) != CUSOLVER_STATUS_SUCCESS) {
    throw logic_error("X: cusolverDnCreateGesvdjInfo failed");
  }
  const auto status = raft::linalg::detail::cusolverDngesvdj_bufferSize<T>(
      solver, CUSOLVER_EIG_MODE_VECTOR, 1, static_cast<int>(n),
      static_cast<int>(d), nullptr, static_cast<int>(n), nullptr, nullptr,
      static_cast<int>(n), nullptr, static_cast<int>(d), lwork, params);
  cusolverDnDestroyGesvdjInfo(params);
  return status;
}

}  // namespace

template <typename T>
int64_t cusolver_lwork_of(const raft::handle_t& handle, glm_solver solver,
                          ols_shape shape) {
  const auto [n, d] = shape;
  cusolverDnHandle_t dn = raft::resource::get_cusolver_dn_handle(handle);
  int lwork = 0;
  cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
  switch (query_of(solver)) {
    case lwork_query::gesvdj:
      status = gesvdj_lwork<T>(dn, shape, &lwork);
      break;
    case lwork_query::gesvd:
      status = raft::linalg::detail::cusolverDngesvd_bufferSize<T>(
          dn, static_cast<int>(n), static_cast<int>(d), &lwork);
      break;
    case lwork_query::syevd:
      status = raft::linalg::detail::cusolverDnsyevd_bufferSize<T>(
          dn, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
          static_cast<int>(d), nullptr, static_cast<int>(d), nullptr, &lwork);
      break;
    case lwork_query::none:
      break;
  }
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw logic_error(
        std::format("X: {}x{} elements are past cuSOLVER's {} workspace query",
                    n, d, algorithm_name(solver.algorithm)));
  }
  return lwork;
}

template int64_t cusolver_lwork_of<float>(const raft::handle_t&, glm_solver,
                                          ols_shape);
template int64_t cusolver_lwork_of<double>(const raft::handle_t&, glm_solver,
                                           ols_shape);
template checked_fit<float> checked_inputs<float>(const raft::handle_t&,
                                                  const fit_request&,
                                                  const std::function<void()>&);
template checked_fit<double> checked_inputs<double>(
    const raft::handle_t&, const fit_request&, const std::function<void()>&);

}  // namespace cumlrkt::glm
