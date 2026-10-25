#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/map.cuh>
#include <raft/stats/meanvar.cuh>
#include <rmm/exec_policy.hpp>

#include "standard_scaler_device.hpp"

namespace cumlrkt::preprocessing {

namespace {

template <typename T>
struct non_finite {
  const T* x;

  __device__ bool operator()(int64_t i) const {
    return !isfinite(x[i]);
  }
};

template <typename T>
struct widened_column {
  const T* x;
  int64_t n;
  int64_t d;
  int64_t first;

  __device__ double operator()(int64_t k) const {
    const int64_t column = k / n;
    const int64_t row = k - (column * n);
    return static_cast<double>(x[(row * d) + first + column]);
  }
};

struct centred {
  const double* scratch;
  const double* mean;
  int64_t n;

  __device__ double operator()(int64_t k) const {
    return scratch[k] - mean[k / n];
  }
};

struct scale_of {
  const double* mean;
  const double* var;
  double* scale;
  double n;

  __device__ void operator()(int64_t j) const {
    const double eps = __DBL_EPSILON__;
    const double v = var[j];
    const double spread = n * mean[j] * eps;
    const double bound =
        __dadd_rn(__dmul_rn(__dmul_rn(n, eps), v), __dmul_rn(spread, spread));
    scale[j] = v <= bound ? 1.0 : sqrt(v);
  }
};

__device__ float narrow(float, double v) {
  return __double2float_rn(v);
}
__device__ double narrow(double, double v) {
  return v;
}
__device__ float sub(float a, float b) {
  return __fsub_rn(a, b);
}
__device__ double sub(double a, double b) {
  return __dsub_rn(a, b);
}
__device__ float div(float a, float b) {
  return __fdiv_rn(a, b);
}
__device__ double div(double a, double b) {
  return __ddiv_rn(a, b);
}
__device__ float mul(float a, float b) {
  return __fmul_rn(a, b);
}
__device__ double mul(double a, double b) {
  return __dmul_rn(a, b);
}
__device__ float add(float a, float b) {
  return __fadd_rn(a, b);
}
__device__ double add(double a, double b) {
  return __dadd_rn(a, b);
}

template <typename T, bool Inverse>
struct elementwise {
  const T* x;
  const double* mean;
  const double* scale;
  T* out;
  int64_t d;

  __device__ void operator()(int64_t i) const {
    const int64_t j = i % d;
    T v = x[i];
    if constexpr (Inverse) {
      if (scale != nullptr) {
        v = mul(v, narrow(T{}, scale[j]));
      }
      if (mean != nullptr) {
        v = add(v, narrow(T{}, mean[j]));
      }
    } else {
      if (mean != nullptr) {
        v = sub(v, narrow(T{}, mean[j]));
      }
      if (scale != nullptr) {
        v = div(v, narrow(T{}, scale[j]));
      }
    }
    out[i] = v;
  }
};

template <typename T, bool Inverse>
void apply(const raft::handle_t& handle, const T* x, const double* mean,
           const double* scale, T* out, int64_t n, int64_t d) {
  const auto stream = raft::resource::get_cuda_stream(handle);
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::counting_iterator<int64_t>(0), n * d,
                     elementwise<T, Inverse>{x, mean, scale, out, d});
}

}  // namespace

template <typename T>
int64_t first_non_finite(const T* x, int64_t count,
                         const raft::handle_t& handle) {
  const auto stream = raft::resource::get_cuda_stream(handle);
  const thrust::counting_iterator<int64_t> elements(0);
  const int64_t i = *thrust::find_if(rmm::exec_policy(stream), elements,
                                     elements + count, non_finite<T>{x});
  return i == count ? -1 : i;
}

template <typename T>
void column_statistics(const raft::handle_t& handle, const T* x, int64_t n,
                       int64_t d, int64_t chunk, double* scratch,
                       double* residual, double* mean, double* var) {
  for (int64_t first = 0; first < d; first += chunk) {
    const int64_t count = std::min(chunk, d - first);
    const auto columns =
        raft::make_device_vector_view<double, int64_t>(scratch, n * count);
    const auto block =
        raft::make_device_matrix_view<const double, int64_t, raft::col_major>(
            scratch, n, count);
    raft::linalg::map_offset(handle, columns,
                             widened_column<T>{x, n, d, first});
    raft::stats::meanvar(
        handle, block,
        raft::make_device_vector_view<double, int64_t>(mean + first, count),
        raft::make_device_vector_view<double, int64_t>(var + first, count),
        false);
    raft::linalg::map_offset(handle, columns,
                             centred{scratch, mean + first, n});
    raft::stats::meanvar(
        handle, block,
        raft::make_device_vector_view<double, int64_t>(residual, count),
        raft::make_device_vector_view<double, int64_t>(var + first, count),
        false);
  }
}

void scales_of(const raft::handle_t& handle, const double* mean,
               const double* var, double* scale, int64_t n, int64_t d) {
  const auto stream = raft::resource::get_cuda_stream(handle);
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::counting_iterator<int64_t>(0), d,
                     scale_of{mean, var, scale, static_cast<double>(n)});
}

template <typename T>
void standardized(const raft::handle_t& handle, const T* x, const double* mean,
                  const double* scale, T* out, int64_t n, int64_t d) {
  apply<T, false>(handle, x, mean, scale, out, n, d);
}

template <typename T>
void unstandardized(const raft::handle_t& handle, const T* x,
                    const double* mean, const double* scale, T* out, int64_t n,
                    int64_t d) {
  apply<T, true>(handle, x, mean, scale, out, n, d);
}

template int64_t first_non_finite<float>(const float*, int64_t,
                                         const raft::handle_t&);
template int64_t first_non_finite<double>(const double*, int64_t,
                                          const raft::handle_t&);
template void column_statistics<float>(const raft::handle_t&, const float*,
                                       int64_t, int64_t, int64_t, double*,
                                       double*, double*, double*);
template void column_statistics<double>(const raft::handle_t&, const double*,
                                        int64_t, int64_t, int64_t, double*,
                                        double*, double*, double*);
template void standardized<float>(const raft::handle_t&, const float*,
                                  const double*, const double*, float*, int64_t,
                                  int64_t);
template void standardized<double>(const raft::handle_t&, const double*,
                                   const double*, const double*, double*,
                                   int64_t, int64_t);
template void unstandardized<float>(const raft::handle_t&, const float*,
                                    const double*, const double*, float*,
                                    int64_t, int64_t);
template void unstandardized<double>(const raft::handle_t&, const double*,
                                     const double*, const double*, double*,
                                     int64_t, int64_t);

}  // namespace cumlrkt::preprocessing
