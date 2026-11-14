#include <cmath>
#include <cstdint>
#include <raft/core/handle.hpp>

#include "cumlrkt/device.cuh"
#include "cumlrkt/support.hpp"
#include "logistic_device.cuh"
#include "logistic_device.hpp"

namespace cumlrkt::glm::logistic {

namespace {

template <typename T>
struct packed_params {
  const T* coef;
  const T* intercept;
  int64_t d;
  int64_t k;
  T* params;

  __device__ void operator()(int64_t o) const {
    const int64_t j = o / k;
    const int64_t c = o - (j * k);
    params[o] = j < d ? coef[(c * d) + j] : intercept[c];
  }
};

template <typename T>
struct probabilities {
  const T* scores;
  int64_t k;
  int64_t classes;
  bool logarithm;
  T* out;

  __device__ void operator()(int64_t i) const {
    if (k == 1) {
      const double z = static_cast<double>(scores[i]);
      if (logarithm) {
        out[(i * 2) + 0] = static_cast<T>(-softplus(z));
        out[(i * 2) + 1] = static_cast<T>(-softplus(-z));
      } else {
        const auto p1 = static_cast<T>(1.0 / (1.0 + exp(-z)));
        out[(i * 2) + 1] = p1;
        out[(i * 2) + 0] = static_cast<T>(1.0 - static_cast<double>(p1));
      }
      return;
    }
    double top = -INFINITY;
    for (int64_t c = 0; c < classes; ++c) {
      top = fmax(top, static_cast<double>(scores[(i * k) + c]));
    }
    double total = 0;
    for (int64_t c = 0; c < classes; ++c) {
      total += exp(static_cast<double>(scores[(i * k) + c]) - top);
    }
    const double log_total = log(total);
    for (int64_t c = 0; c < classes; ++c) {
      const double shifted = static_cast<double>(scores[(i * k) + c]) - top;
      out[(i * k) + c] = static_cast<T>(logarithm ? shifted - log_total
                                                  : exp(shifted) / total);
    }
  }
};

template <typename T>
__device__ int64_t argmax_of(const T* scores, int64_t k, int64_t i) {
  if (k == 1) {
    return scores[i] > T{0} ? 1 : 0;
  }
  int64_t best = 0;
  for (int64_t c = 1; c < k; ++c) {
    if (scores[(i * k) + c] > scores[(i * k) + best]) {
      best = c;
    }
  }
  return best;
}

template <typename T>
struct argmax_code {
  const T* scores;
  int64_t k;
  int32_t* out;

  __device__ void operator()(int64_t i) const {
    out[i] = static_cast<int32_t>(argmax_of(scores, k, i));
  }
};

template <typename T, typename L>
struct argmax_label {
  const T* scores;
  int64_t k;
  const L* classes;
  L* out;

  __device__ void operator()(int64_t i) const {
    out[i] = classes[argmax_of(scores, k, i)];
  }
};

}  // namespace

template <typename T>
void coef_to_params(const raft::handle_t& handle, const T* coef,
                    const T* intercept, int64_t d, int64_t k, T* params) {
  each(handle, (d + 1) * k, packed_params<T>{coef, intercept, d, k, params});
}

template <typename T>
void scores_to(const raft::handle_t& handle, const T* scores, int64_t n,
               int64_t k, bool logarithm, T* out) {
  const int64_t classes = k == 1 ? 2 : k;
  each(handle, n, probabilities<T>{scores, k, classes, logarithm, out});
}

template <typename T>
void codes_of(const raft::handle_t& handle, const T* scores, int64_t n,
              int64_t k, int32_t* out) {
  each(handle, n, argmax_code<T>{scores, k, out});
}

template <typename T, typename L>
void labels_of(const raft::handle_t& handle, const T* scores, int64_t n,
               int64_t k, const L* classes, L* out) {
  each(handle, n, argmax_label<T, L>{scores, k, classes, out});
}

#define CUMLRKT_LOGISTIC_APPLY_FLOAT(T)                                        \
  template void coef_to_params<T>(const raft::handle_t&, const T*, const T*,   \
                                  int64_t, int64_t, T*);                       \
  template void scores_to<T>(const raft::handle_t&, const T*, int64_t,         \
                             int64_t, bool, T*);                               \
  template void codes_of<T>(const raft::handle_t&, const T*, int64_t, int64_t, \
                            int32_t*);

#define CUMLRKT_LOGISTIC_APPLY_LABEL(L)                                    \
  template void labels_of<float, L>(const raft::handle_t&, const float*,   \
                                    int64_t, int64_t, const L*, L*);       \
  template void labels_of<double, L>(const raft::handle_t&, const double*, \
                                     int64_t, int64_t, const L*, L*);

CUMLRKT_LOGISTIC_APPLY_FLOAT(float)
CUMLRKT_LOGISTIC_APPLY_FLOAT(double)
CUMLRKT_LOGISTIC_APPLY_LABEL(int32_t)
CUMLRKT_LOGISTIC_APPLY_LABEL(int64_t)
CUMLRKT_LOGISTIC_APPLY_LABEL(float)
CUMLRKT_LOGISTIC_APPLY_LABEL(double)

}  // namespace cumlrkt::glm::logistic
