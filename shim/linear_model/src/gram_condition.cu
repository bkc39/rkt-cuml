#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <raft/core/cublas_macros.hpp>
#include <raft/core/cusolver_macros.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cusolver_dn_handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <type_traits>
#include <vector>

#include "cumlrkt/device.cuh"
#include "cumlrkt/linear_model.h"
#include "cumlrkt/scale.hpp"
#include "cumlrkt/support.hpp"
#include "glm_staging.hpp"
#include "glm_staging_device.hpp"
#include "gram_condition.hpp"

namespace cumlrkt::glm {

namespace {

constexpr int64_t gram_rows = int64_t{1} << 15;

template <typename T>
struct weight_of {
  const T* w;

  __device__ double operator()(int64_t i) const {
    return w == nullptr ? 1.0 : static_cast<double>(w[i]);
  }
};

template <typename T>
struct weighted_entry {
  const T* x;
  const T* w;
  int64_t n;

  __device__ double operator()(int64_t o) const {
    const double v = static_cast<double>(x[o]);
    return w == nullptr ? v : static_cast<double>(w[o % n]) * v;
  }
};

template <typename T>
struct centred_rows {
  const T* x;
  const T* w;
  const double* mean;
  int64_t n;
  int64_t start;
  int64_t count;
  double* out;

  __device__ void operator()(int64_t o) const {
    const int64_t k = o / count;
    const int64_t i = start + (o - (k * count));
    const double root = w == nullptr ? 1.0 : sqrt(static_cast<double>(w[i]));
    out[o] = root * (static_cast<double>(x[(k * n) + i]) - mean[k]);
  }
};

template <typename T>
std::vector<double> column_means(const T* x, const T* w, int64_t n, int64_t d,
                                 rmm::cuda_stream_view stream) {
  const thrust::counting_iterator<int64_t> rows(0);
  const double total =
      thrust::transform_reduce(rmm::exec_policy(stream), rows, rows + n,
                               weight_of<T>{w}, 0.0, thrust::plus<double>{});
  std::vector<double> mean(d);
  for (int64_t k = 0; k < d; ++k) {
    mean[k] = thrust::transform_reduce(rmm::exec_policy(stream), rows, rows + n,
                                       weighted_entry<T>{x + (k * n), w, n},
                                       0.0, thrust::plus<double>{}) /
              total;
  }
  return mean;
}

template <typename T>
buffer<double> centred_gram(const raft::handle_t& handle, const T* x,
                            const T* w, int64_t n, int64_t d,
                            bool fit_intercept) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const std::vector<double> mean = fit_intercept
                                       ? column_means(x, w, n, d, stream)
                                       : std::vector<double>(d, 0.0);
  const buffer<double> means = to_device(mean, stream);
  buffer<double> gram = scratch<double>(d * d, stream);
  const int64_t chunk = std::min(n, gram_rows);
  buffer<double> block = scratch<double>(chunk * d, stream);
  cublasHandle_t cublas = raft::resource::get_cublas_handle(handle);
  RAFT_CUBLAS_TRY(cublasSetStream(cublas, stream.value()));
  for (int64_t start = 0; start < n; start += chunk) {
    const int64_t count = std::min(chunk, n - start);
    each(handle, count * d,
         centred_rows<T>{x, w, means.data(), n, start, count, block.data()});
    const double one = 1.0;
    const double keep = start == 0 ? 0.0 : 1.0;
    RAFT_CUBLAS_TRY(cublasDsyrk(cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                static_cast<int>(d), static_cast<int>(count),
                                &one, block.data(), static_cast<int>(count),
                                &keep, gram.data(), static_cast<int>(d)));
  }
  return gram;
}

double smallest_over_largest(const raft::handle_t& handle,
                             std::vector<double> m, int64_t d) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  buffer<double> a = to_device(m, stream);
  buffer<double> values = scratch<double>(d, stream);
  cusolverDnHandle_t solver = raft::resource::get_cusolver_dn_handle(handle);
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(solver, stream.value()));
  int lwork = 0;
  RAFT_CUSOLVER_TRY(cusolverDnDsyevd_bufferSize(
      solver, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
      static_cast<int>(d), a.data(), static_cast<int>(d), values.data(),
      &lwork));
  buffer<double> work = scratch<double>(lwork, stream);
  buffer<int> info = scratch<int>(1, stream);
  RAFT_CUSOLVER_TRY(cusolverDnDsyevd(
      solver, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
      static_cast<int>(d), a.data(), static_cast<int>(d), values.data(),
      work.data(), lwork, info.data()));
  const std::vector<double> lambda = to_host(values, stream);
  const std::vector<int> status = to_host(info, stream);
  if (status[0] != 0 || !(lambda.front() > 0)) {
    return 0.0;
  }
  return lambda.front() / lambda.back();
}

}  // namespace

template <typename T>
double equilibrated_condition(const raft::handle_t& handle, const T* x,
                              const T* w, int64_t n, int64_t d, T alpha,
                              bool fit_intercept) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const std::vector<double> gram =
      to_host(centred_gram(handle, x, w, n, d, fit_intercept), stream);
  const auto a = static_cast<double>(alpha);
  std::vector<double> scale(d);
  for (int64_t k = 0; k < d; ++k) {
    scale[k] = 1.0 / std::sqrt(gram[(k * d) + k] + a);
  }
  std::vector<double> m(d * d, 0.0);
  for (int64_t k = 0; k < d; ++k) {
    for (int64_t j = k; j < d; ++j) {
      const double g = gram[(k * d) + j] + (j == k ? a : 0.0);
      m[(k * d) + j] = g * scale[j] * scale[k];
    }
  }
  const double ratio = smallest_over_largest(handle, std::move(m), d);
  return ratio > 0 ? 1.0 / ratio : std::numeric_limits<double>::infinity();
}

template <typename T>
bool checks_conditioning(glm_solver solver, bool alpha_window) {
  if (solver.family == glm_family::ols) {
    return std::is_same_v<T, float> &&
           solver.algorithm == CUML_LINEAR_MODEL_OLS_SVD_QR;
  }
  return solver.family == glm_family::ridge && alpha_window &&
         solver.algorithm == CUML_LINEAR_MODEL_RIDGE_SVD;
}

template <typename T>
void require_conditioned(double kappa, glm_solver solver) {
  const bool ols = solver.family == glm_family::ols;
  const double share = unit_roundoff<T> * (ols ? std::sqrt(kappa) : kappa);
  if (share < std::ldexp(1.0, -tolerance_bits<T>)) {
    return;
  }
  throw logic_error(std::format(
      "X: the {} solve's {}Gram matrix has u*kappa {} >= 2^-{}; {}",
      solver_name(solver), ols ? "" : "regularised ", printed(share),
      tolerance_bits<T>,
      ols ? "X may have collinear columns"
      : std::is_same_v<T, float>
          ? "X is ill-conditioned for this alpha (float64 or a larger "
            "alpha may fit)"
          : "X is ill-conditioned for this alpha (a larger alpha may "
            "fit)"));
}

template bool checks_conditioning<float>(glm_solver, bool);
template bool checks_conditioning<double>(glm_solver, bool);
template void require_conditioned<float>(double, glm_solver);
template void require_conditioned<double>(double, glm_solver);
template double equilibrated_condition<float>(const raft::handle_t&,
                                              const float*, const float*,
                                              int64_t, int64_t, float, bool);
template double equilibrated_condition<double>(const raft::handle_t&,
                                               const double*, const double*,
                                               int64_t, int64_t, double, bool);

}  // namespace cumlrkt::glm
