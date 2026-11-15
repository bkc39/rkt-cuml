#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <cmath>
#include <cstdint>
#include <raft/core/handle.hpp>
#include <rmm/exec_policy.hpp>

#include "cumlrkt/device.cuh"
#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "mbsgd_device.hpp"

namespace cumlrkt::glm::mbsgd {

template <typename T>
void columns_of(const raft::handle_t& handle, const T* x, int64_t n, int64_t d,
                const int64_t* order, bool ones, T* out) {
  const int64_t cols = d + (ones ? 1 : 0);
  each(handle, n * cols, [=] __device__(int64_t k) {
    const int64_t j = k / n;
    const int64_t i = k % n;
    const int64_t row = order == nullptr ? i : order[i];
    out[k] = j < d ? x[(row * d) + j] : T{1};
  });
}

template <typename T>
void reordered(const raft::handle_t& handle, const T* y, int64_t n,
               const int64_t* order, T* out) {
  each(handle, n, [=] __device__(int64_t i) {
    out[i] = y[order == nullptr ? i : order[i]];
  });
}

template <typename T>
void code_targets(const raft::handle_t& handle, const int32_t* codes, int64_t n,
                  bool signs, T* out) {
  each(handle, n, [=] __device__(int64_t i) {
    const T c = codes[i] == 1 ? T{1} : T{0};
    out[i] = signs ? (2 * c) - 1 : c;
  });
}

template <typename T>
double mean_of(const raft::handle_t& handle, const T* y, int64_t n) {
  const auto begin = thrust::counting_iterator<int64_t>(0);
  const double sum = thrust::transform_reduce(
      rmm::exec_policy(stream_of(handle)), begin, begin + n,
      [=] __device__(int64_t i) -> double { return static_cast<double>(y[i]); },
      0.0, thrust::plus<double>());
  return sum / static_cast<double>(n);
}

inline __device__ double row_loss(int32_t loss, double y, double z) {
  switch (loss) {
    case CUML_LINEAR_MODEL_SGD_LOG:
      return log1p(exp(-fabs(z))) + fmax(z, 0.0) - (y * z);
    case CUML_LINEAR_MODEL_SGD_HINGE:
      return fmax(0.0, 1.0 - (y * z));
    default: {
      const double r = y - z;
      return r * r;
    }
  }
}

template <typename T>
double mean_loss(const raft::handle_t& handle, const T* x, const T* y,
                 const T* coef, double intercept, const objective_shape& s) {
  const auto begin = thrust::counting_iterator<int64_t>(0);
  const int64_t d = s.d;
  const bool ones = s.ones;
  const int32_t loss = s.loss;
  const double sum = thrust::transform_reduce(
      rmm::exec_policy(stream_of(handle)), begin, begin + s.n,
      [=] __device__(int64_t i) -> double {
        double z = ones ? static_cast<double>(coef[d]) : intercept;
        for (int64_t j = 0; j < d; ++j) {
          z += static_cast<double>(x[(i * d) + j]) *
               static_cast<double>(coef[j]);
        }
        return row_loss(loss, static_cast<double>(y[i]), z);
      },
      0.0, thrust::plus<double>());
  return sum / static_cast<double>(s.n);
}

template <typename T, typename L>
void threshold_labels(const raft::handle_t& handle, const T* decision,
                      int64_t n, double threshold, const L* classes, L* out) {
  each(handle, n, [=] __device__(int64_t i) {
    const bool second = static_cast<double>(decision[i]) > threshold;
    if (classes == nullptr) {
      out[i] = second ? L{1} : L{0};
    } else {
      out[i] = classes[second ? 1 : 0];
    }
  });
}

#define CUMLRKT_MBSGD_FLOAT(T)                                                 \
  template void columns_of<T>(const raft::handle_t&, const T*, int64_t,        \
                              int64_t, const int64_t*, bool, T*);              \
  template void reordered<T>(const raft::handle_t&, const T*, int64_t,         \
                             const int64_t*, T*);                              \
  template void code_targets<T>(const raft::handle_t&, const int32_t*,         \
                                int64_t, bool, T*);                            \
  template double mean_of<T>(const raft::handle_t&, const T*, int64_t);        \
  template double mean_loss<T>(const raft::handle_t&, const T*, const T*,      \
                               const T*, double, const objective_shape&);      \
  template void threshold_labels<T, int32_t>(const raft::handle_t&, const T*,  \
                                             int64_t, double, const int32_t*,  \
                                             int32_t*);                        \
  template void threshold_labels<T, int64_t>(const raft::handle_t&, const T*,  \
                                             int64_t, double, const int64_t*,  \
                                             int64_t*);                        \
  template void threshold_labels<T, float>(                                    \
      const raft::handle_t&, const T*, int64_t, double, const float*, float*); \
  template void threshold_labels<T, double>(const raft::handle_t&, const T*,   \
                                            int64_t, double, const double*,    \
                                            double*);

CUMLRKT_MBSGD_FLOAT(float)
CUMLRKT_MBSGD_FLOAT(double)

}  // namespace cumlrkt::glm::mbsgd
