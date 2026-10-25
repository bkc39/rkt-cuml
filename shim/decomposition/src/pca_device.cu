#include <cfloat>
#include <cstdint>
#include <raft/core/handle.hpp>

#include "pca_device.hpp"
#include "signs.cuh"

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
    if (variance == nullptr) {
      out[o] = v;
    } else {
      const T scale = root(variance[j]);
      out[o] = over(v, scale < epsilon(T{}) ? epsilon(T{}) : scale);
    }
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
  each(handle, k, signed_components<T>{components, out.components, m, d, nullptr});
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

}  // namespace cumlrkt::decomposition
