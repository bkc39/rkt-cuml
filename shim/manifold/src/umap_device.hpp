#pragma once

#include <cstdint>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace cumlrkt::manifold {

struct label_codes {
  rmm::device_uvector<float> codes;
  int64_t labelled;
};

template <typename T>
label_codes float_codes(const T* labels, int64_t n,
                        rmm::cuda_stream_view stream);

}  // namespace cumlrkt::manifold
