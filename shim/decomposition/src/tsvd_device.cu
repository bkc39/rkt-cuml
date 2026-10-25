#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <cstdint>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/exec_policy.hpp>

#include "signs.cuh"
#include "tsvd_device.hpp"

namespace cumlrkt::decomposition {

namespace {

template <typename T>
struct kept_spectrum {
  tsvd_arrays<T> fitted;
  tsvd_arrays<T> out;
  bool constant;

  __device__ void operator()(int64_t i) const {
    const T s = fitted.singular[i];
    const T r = fitted.ratio[i];
    out.singular[i] = s > T{0} ? s : T{0};
    out.variance[i] = constant ? T{0} : fitted.variance[i];
    out.ratio[i] = constant || !isfinite(r) ? T{0} : r;
  }
};

template <typename T>
struct signed_rows {
  const T* t;
  const bool* flipped;
  T* out;
  int64_t n;
  int64_t k;

  __device__ void operator()(int64_t o) const {
    const int64_t i = o / k;
    const int64_t j = o - (i * k);
    const T v = t[i + (j * n)];
    out[o] = flipped[j] ? -v : v;
  }
};

template <typename T>
struct squared {
  __device__ double operator()(T x) const {
    const auto wide = static_cast<double>(x);
    return wide * wide;
  }
};

}  // namespace

template <typename T>
void tsvd_post_fit(const raft::handle_t& handle, const tsvd_arrays<T>& fitted,
                   const tsvd_extents& e, bool* flipped,
                   const tsvd_arrays<T>& out) {
  each(handle, e.k,
       signed_components<T>{fitted.components, out.components, e.k, e.d,
                            flipped});
  each(handle, e.k, kept_spectrum<T>{fitted, out, e.constant});
  if (out.transform != nullptr) {
    each(handle, e.n * e.k,
         signed_rows<T>{fitted.transform, flipped, out.transform, e.n, e.k});
  }
}

template <typename T>
double squared_frobenius(const raft::handle_t& handle, const T* x,
                         int64_t count) {
  const auto stream = raft::resource::get_cuda_stream(handle);
  return thrust::transform_reduce(rmm::exec_policy(stream), x, x + count,
                                  squared<T>{}, 0.0, thrust::plus<double>());
}

template void tsvd_post_fit<float>(const raft::handle_t&,
                                   const tsvd_arrays<float>&,
                                   const tsvd_extents&, bool*,
                                   const tsvd_arrays<float>&);
template void tsvd_post_fit<double>(const raft::handle_t&,
                                    const tsvd_arrays<double>&,
                                    const tsvd_extents&, bool*,
                                    const tsvd_arrays<double>&);
template double squared_frobenius<float>(const raft::handle_t&, const float*,
                                         int64_t);
template double squared_frobenius<double>(const raft::handle_t&, const double*,
                                          int64_t);

}  // namespace cumlrkt::decomposition
