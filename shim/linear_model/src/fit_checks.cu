#include <cstdint>
#include <format>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cusolver_dn_handle.hpp>
#include <raft/linalg/detail/cusolver_wrappers.hpp>
#include <rmm/cuda_stream_view.hpp>

#include "cumlrkt/limits.hpp"
#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "cumlrkt/vectors.hpp"
#include "fit_checks.hpp"
#include "glm_staging.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt::glm {

void require_workspace(int algorithm, int64_t n, int64_t d, int64_t lwork) {
  const int64_t size = ols_workspace(algorithm, n, d, lwork);
  if (size > most_index) {
    throw logic_error(std::format(
        "X: {} columns need a {}-element {} workspace, more than {} elements",
        d, size, algorithm_name(algorithm), most_index));
  }
}

void require_codes(int32_t fit_intercept, int32_t algorithm) {
  if (algorithm < CUML_LINEAR_MODEL_OLS_SVD ||
      algorithm > CUML_LINEAR_MODEL_OLS_SVD_QR) {
    throw logic_error(
        "algorithm: expected auto, eig, svd, svd-jacobi, qr or svd-qr");
  }
  if (fit_intercept != 0 && fit_intercept != 1) {
    throw logic_error("fit-intercept: expected #t or #f");
  }
}

template <typename T>
fit_inputs<T> checked_inputs(const raft::handle_t& handle,
                             const rr_view* x_view, const rr_view* y_view,
                             const rr_view* w_view, int32_t fit_intercept,
                             int32_t algorithm) {
  const matrix<T, int64_t> x = rows_of<T, int64_t>(
      x_view, "X", raftrkt::any_extent, raftrkt::any_extent);
  const T* y = raftrkt::vector_data<T, int32_t>(y_view, "y", x.n);
  const T* w = w_view == nullptr ? nullptr
                                 : raftrkt::vector_data<T, int32_t>(
                                       w_view, "sample-weight", x.n);
  require_some("X", "row", x.n, 2);
  require_some("X", "column", x.d);
  require_index("X", x.n, x.d);
  require_codes(fit_intercept, algorithm);
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  require_finite("X", x, stream);
  require_finite_elements("y", y, x.n, stream);
  if (w != nullptr) {
    require_sample_weights(w, x.n, stream);
  }
  return {x, y, w};
}

template <typename T>
int64_t cusolver_lwork_of(const raft::handle_t& handle, int algorithm,
                          int64_t n, int64_t d) {
  cusolverDnHandle_t solver = raft::resource::get_cusolver_dn_handle(handle);
  int lwork = 0;
  if (algorithm == CUML_LINEAR_MODEL_OLS_SVD) {
    gesvdjInfo_t params = nullptr;
    if (cusolverDnCreateGesvdjInfo(&params) != CUSOLVER_STATUS_SUCCESS) {
      throw logic_error("X: cusolverDnCreateGesvdjInfo failed");
    }
    const auto status = raft::linalg::detail::cusolverDngesvdj_bufferSize<T>(
        solver, CUSOLVER_EIG_MODE_VECTOR, 1, static_cast<int>(n),
        static_cast<int>(d), nullptr, static_cast<int>(n), nullptr, nullptr,
        static_cast<int>(n), nullptr, static_cast<int>(d), &lwork, params);
    cusolverDnDestroyGesvdjInfo(params);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      throw logic_error(std::format(
          "X: {}x{} elements are past cuSOLVER's svd workspace query", n, d));
    }
  } else if (algorithm == CUML_LINEAR_MODEL_OLS_SVD_QR) {
    if (raft::linalg::detail::cusolverDngesvd_bufferSize<T>(
            solver, static_cast<int>(n), static_cast<int>(d), &lwork) !=
        CUSOLVER_STATUS_SUCCESS) {
      throw logic_error(std::format(
          "X: {}x{} elements are past cuSOLVER's svd-qr workspace query", n,
          d));
    }
  }
  return lwork;
}

template int64_t cusolver_lwork_of<float>(const raft::handle_t&, int, int64_t,
                                          int64_t);
template int64_t cusolver_lwork_of<double>(const raft::handle_t&, int, int64_t,
                                           int64_t);
template fit_inputs<float> checked_inputs<float>(const raft::handle_t&,
                                                 const rr_view*, const rr_view*,
                                                 const rr_view*, int32_t,
                                                 int32_t);
template fit_inputs<double> checked_inputs<double>(const raft::handle_t&,
                                                   const rr_view*,
                                                   const rr_view*,
                                                   const rr_view*, int32_t,
                                                   int32_t);

}  // namespace cumlrkt::glm
