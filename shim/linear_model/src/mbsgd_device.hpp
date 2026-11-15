#pragma once

#include <cstdint>
#include <raft/core/handle.hpp>

namespace cumlrkt::glm::mbsgd {

struct objective_shape {
  int64_t n;
  int64_t d;
  bool ones;
  int32_t loss;
};

template <typename T>
void columns_of(const raft::handle_t& handle, const T* x, int64_t n, int64_t d,
                const int64_t* order, bool ones, T* out);

template <typename T>
void reordered(const raft::handle_t& handle, const T* y, int64_t n,
               const int64_t* order, T* out);

template <typename T>
void code_targets(const raft::handle_t& handle, const int32_t* codes, int64_t n,
                  bool signs, T* out);

template <typename T>
double mean_of(const raft::handle_t& handle, const T* y, int64_t n);

template <typename T>
double mean_loss(const raft::handle_t& handle, const T* x, const T* y,
                 const T* coef, double intercept, const objective_shape& s);

template <typename T, typename L>
void threshold_labels(const raft::handle_t& handle, const T* decision,
                      int64_t n, double threshold, const L* classes, L* out);

}  // namespace cumlrkt::glm::mbsgd
