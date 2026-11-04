#include "cumlrkt/vectors.hpp"

#include <cstdint>
#include <format>
#include <rmm/cuda_stream_view.hpp>

#include "cumlrkt/common.hpp"
#include "cumlrkt/rows.hpp"
#include "cumlrkt/support.hpp"

namespace cumlrkt {

template <typename T>
void require_finite_elements(const char* name, const T* x, int64_t n,
                             rmm::cuda_stream_view stream) {
  const int64_t i = first_non_finite(x, n, stream);
  if (i >= 0) {
    throw logic_error(
        std::format("{}: element {}: expected a finite value, got {}", name, i,
                    printed(static_cast<double>(read_one(x, i, stream)))));
  }
}

template <typename T>
void require_non_negative_elements(const char* name, const T* x, int64_t n,
                                   rmm::cuda_stream_view stream) {
  const int64_t i = first_negative(x, n, stream);
  if (i >= 0) {
    throw logic_error(
        std::format("{}: element {}: expected at least 0, got {}", name, i,
                    printed(static_cast<double>(read_one(x, i, stream)))));
  }
}

template <typename T>
double require_sample_weights(const T* w, int64_t n,
                              rmm::cuda_stream_view stream) {
  require_finite_elements("sample-weight", w, n, stream);
  require_non_negative_elements("sample-weight", w, n, stream);
  const double largest = largest_magnitude(w, n, stream);
  if (!(largest > 0)) {
    throw logic_error("sample-weight: expected a positive sum, got 0");
  }
  return largest;
}

template void require_finite_elements<float>(const char*, const float*, int64_t,
                                             rmm::cuda_stream_view);
template void require_finite_elements<double>(const char*, const double*,
                                              int64_t, rmm::cuda_stream_view);
template void require_non_negative_elements<float>(const char*, const float*,
                                                   int64_t,
                                                   rmm::cuda_stream_view);
template void require_non_negative_elements<double>(const char*, const double*,
                                                    int64_t,
                                                    rmm::cuda_stream_view);
template double require_sample_weights<float>(const float*, int64_t,
                                              rmm::cuda_stream_view);
template double require_sample_weights<double>(const double*, int64_t,
                                               rmm::cuda_stream_view);

}  // namespace cumlrkt
