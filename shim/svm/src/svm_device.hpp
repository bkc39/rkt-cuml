#pragma once

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <utility>

namespace cumlrkt::svm {

int64_t first_bad_code(const int32_t* codes, int64_t n,
                       rmm::cuda_stream_view stream);

template <typename T>
void codes_to(const int32_t* codes, int64_t n, T* out,
              rmm::cuda_stream_view stream);

template <typename T>
std::pair<double, double> class_weights(const int32_t* codes, const T* w,
                                        int64_t n,
                                        rmm::cuda_stream_view stream);

template <typename T>
double penalty_total(const T* w, int64_t n, double C,
                     rmm::cuda_stream_view stream);

template <typename T>
double largest_squared_norm(const T* x, int64_t n, int64_t d,
                            rmm::cuda_stream_view stream);

template <typename T>
double absolute_total(const T* x, int64_t n, rmm::cuda_stream_view stream);

template <typename T>
std::pair<int64_t, int64_t> sign_counts(const T* dual, int64_t n,
                                        rmm::cuda_stream_view stream);

template <typename T, typename L>
void threshold_labels(const T* decision, int64_t n, const L* classes, L* out,
                      rmm::cuda_stream_view stream);

}  // namespace cumlrkt::svm
