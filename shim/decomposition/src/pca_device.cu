#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <cfloat>
#include <cstdint>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/exec_policy.hpp>

#include "pca_device.hpp"

namespace cumlrkt::decomposition {

namespace {

__device__ float root(float v) {
  return __fsqrt_rn(v);
}
__device__ double root(double v) {
  return __dsqrt_rn(v);
}
__device__ float times(float a, float b) {
  return __fmul_rn(a, b);
}
__device__ double times(double a, double b) {
  return __dmul_rn(a, b);
}
__device__ float over(float a, float b) {
  return __fdiv_rn(a, b);
}
__device__ double over(double a, double b) {
  return __ddiv_rn(a, b);
}
__device__ float narrow(float, double v) {
  return __double2float_rn(v);
}
__device__ double narrow(double, double v) {
  return v;
}
__device__ float epsilon(float) {
  return FLT_EPSILON;
}
__device__ double epsilon(double) {
  return DBL_EPSILON;
}

template <typename T>
__device__ T divisor(T variance) {
  const T scale = root(variance);
  return scale < epsilon(T{}) ? epsilon(T{}) : scale;
}

template <typename T>
struct column_major {
  const T* x;
  const T* variance;
  T* out;
  int64_t n;
  int64_t d;

  __device__ void operator()(int64_t o) const {
    const int64_t j = o / n;
    const int64_t i = o - (j * n);
    const T v = x[(i * d) + j];
    out[o] = variance == nullptr ? v : times(v, root(variance[j]));
  }
};

template <typename T>
struct row_major {
  const T* x;
  const T* variance;
  T* out;
  int64_t n;
  int64_t d;

  __device__ void operator()(int64_t o) const {
    const int64_t i = o / d;
    const int64_t j = o - (i * d);
    const T v = x[i + (j * n)];
    out[o] = variance == nullptr ? v : over(v, divisor(variance[j]));
  }
};

template <typename T>
struct spectrum_totals {
  T* variance;
  double* totals;
  int64_t m;
  int64_t k;

  __device__ void operator()(int64_t) const {
    double sum = 0;
    double rest = 0;
    for (int64_t i = 0; i < m; ++i) {
      if (variance[i] < 0) {
        variance[i] = 0;
      }
      const auto v = static_cast<double>(variance[i]);
      sum = __dadd_rn(sum, v);
      if (i >= k) {
        rest = __dadd_rn(rest, v);
      }
    }
    totals[0] = sum;
    totals[1] = k < m ? __ddiv_rn(rest, static_cast<double>(m - k)) : 0.0;
  }
};

template <typename T>
struct kept_spectrum {
  const T* variance;
  const double* totals;
  fit_arrays<T> out;
  double dof;

  __device__ void operator()(int64_t i) const {
    const T v = variance[i];
    const auto wide = static_cast<double>(v);
    out.variance[i] = v;
    out.ratio[i] =
        totals[0] == 0 ? T{0} : narrow(T{}, __ddiv_rn(wide, totals[0]));
    out.singular[i] = narrow(T{}, __dsqrt_rn(__dmul_rn(wide, dof)));
  }
};

template <typename T>
struct signed_components {
  const T* components;
  T* out;
  int64_t m;
  int64_t d;

  __device__ void operator()(int64_t i) const {
    T best = components[i];
    for (int64_t j = 1; j < d; ++j) {
      const T c = components[i + (j * m)];
      if (fabs(c) > fabs(best)) {
        best = c;
      }
    }
    const bool flip = best < 0;
    for (int64_t j = 0; j < d; ++j) {
      const T c = components[i + (j * m)];
      out[(i * d) + j] = flip ? -c : c;
    }
  }
};

template <typename T>
struct bounds_one {
  __device__ variance_bounds operator()(T v) const {
    return {static_cast<double>(divisor(v)), root(static_cast<double>(v))};
  }
};

struct bounds_both {
  __host__ __device__ variance_bounds
  operator()(const variance_bounds& a, const variance_bounds& b) const {
    return {thrust::minimum<double>{}(a.smallest_divisor, b.smallest_divisor),
            thrust::maximum<double>{}(a.largest_root, b.largest_root)};
  }
};

template <typename F>
void each(const raft::handle_t& handle, int64_t count, F f) {
  const auto stream = raft::resource::get_cuda_stream(handle);
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::counting_iterator<int64_t>(0), count, f);
}

}  // namespace

template <typename T>
void to_columns(const raft::handle_t& handle, const T* x, int64_t n, int64_t d,
                const T* variance, T* out) {
  each(handle, n * d, column_major<T>{x, variance, out, n, d});
}

template <typename T>
void to_rows(const raft::handle_t& handle, const T* x, int64_t n, int64_t d,
             const T* variance, T* out) {
  each(handle, n * d, row_major<T>{x, variance, out, n, d});
}

template <typename T>
void post_fit(const raft::handle_t& handle, const T* components, T* variance,
              int64_t m, int64_t k, int64_t d, int64_t n,
              const fit_arrays<T>& out, double* totals) {
  each(handle, 1, spectrum_totals<T>{variance, totals, m, k});
  each(handle, k,
       kept_spectrum<T>{variance, totals, out, static_cast<double>(n - 1)});
  each(handle, k, signed_components<T>{components, out.components, m, d});
}

template <typename T>
variance_bounds bounds_of(const raft::handle_t& handle, const T* variance,
                          int64_t k) {
  const auto stream = raft::resource::get_cuda_stream(handle);
  return thrust::transform_reduce(rmm::exec_policy(stream), variance,
                                  variance + k, bounds_one<T>{},
                                  variance_bounds{1.0, 1.0}, bounds_both{});
}

template void to_columns<float>(const raft::handle_t&, const float*, int64_t,
                                int64_t, const float*, float*);
template void to_columns<double>(const raft::handle_t&, const double*, int64_t,
                                 int64_t, const double*, double*);
template void to_rows<float>(const raft::handle_t&, const float*, int64_t,
                             int64_t, const float*, float*);
template void to_rows<double>(const raft::handle_t&, const double*, int64_t,
                              int64_t, const double*, double*);
template void post_fit<float>(const raft::handle_t&, const float*, float*,
                              int64_t, int64_t, int64_t, int64_t,
                              const fit_arrays<float>&, double*);
template void post_fit<double>(const raft::handle_t&, const double*, double*,
                               int64_t, int64_t, int64_t, int64_t,
                               const fit_arrays<double>&, double*);
template variance_bounds bounds_of<float>(const raft::handle_t&, const float*,
                                          int64_t);
template variance_bounds bounds_of<double>(const raft::handle_t&, const double*,
                                           int64_t);

}  // namespace cumlrkt::decomposition
