#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <cstdint>
#include <raft/core/handle.hpp>

#include "pca_device.hpp"
#include "signs.cuh"

namespace cumlrkt::decomposition {

namespace {

__device__ float narrow(float, double v) {
  return __double2float_rn(v);
}
__device__ double narrow(double, double v) {
  return v;
}

template <typename T>
struct spectrum_totals {
  T* variance;
  double* totals;
  int64_t m;
  int64_t k;
  int exponent;

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
    totals[1] = k < m ? scaled(__ddiv_rn(rest, static_cast<double>(m - k)),
                               -2 * exponent)
                      : 0.0;
  }
};

template <typename T>
struct kept_spectrum {
  const T* variance;
  const double* totals;
  fit_arrays<T> out;
  double dof;
  int exponent;

  __device__ void operator()(int64_t i) const {
    const T v = variance[i];
    const auto wide = static_cast<double>(v);
    out.variance[i] = scaled(v, -2 * exponent);
    out.ratio[i] =
        totals[0] == 0 ? T{0} : narrow(T{}, __ddiv_rn(wide, totals[0]));
    out.singular[i] =
        narrow(T{}, scaled(__dsqrt_rn(__dmul_rn(wide, dof)), -exponent));
  }
};

template <typename T>
struct unscaled_mean {
  T* mean;
  int exponent;

  __device__ void operator()(int64_t j) const {
    mean[j] = scaled(mean[j], -exponent);
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

}  // namespace

template <typename T>
void post_fit(const raft::handle_t& handle, const T* components, T* variance,
              int64_t m, int64_t k, int64_t d, int64_t n,
              const fit_arrays<T>& out, double* totals, int exponent) {
  each(handle, 1, spectrum_totals<T>{variance, totals, m, k, exponent});
  each(handle, k,
       kept_spectrum<T>{variance, totals, out, static_cast<double>(n - 1),
                        exponent});
  each(handle, k,
       signed_components<T>{components, out.components, m, d, nullptr});
  each(handle, d, unscaled_mean<T>{out.mean, exponent});
}

template <typename T>
variance_bounds bounds_of(const raft::handle_t& handle, const T* variance,
                          int64_t k) {
  const auto stream = raft::resource::get_cuda_stream(handle);
  return thrust::transform_reduce(rmm::exec_policy(stream), variance,
                                  variance + k, bounds_one<T>{},
                                  variance_bounds{1.0, 1.0}, bounds_both{});
}

template void post_fit<float>(const raft::handle_t&, const float*, float*,
                              int64_t, int64_t, int64_t, int64_t,
                              const fit_arrays<float>&, double*, int);
template void post_fit<double>(const raft::handle_t&, const double*, double*,
                               int64_t, int64_t, int64_t, int64_t,
                               const fit_arrays<double>&, double*, int);
template variance_bounds bounds_of<float>(const raft::handle_t&, const float*,
                                          int64_t);
template variance_bounds bounds_of<double>(const raft::handle_t&, const double*,
                                           int64_t);

}  // namespace cumlrkt::decomposition
