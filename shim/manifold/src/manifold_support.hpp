#pragma once

#include <cstdint>
#include <cuml/common/distance_type.hpp>
#include <rmm/cuda_stream_view.hpp>

#include "cumlrkt/support.hpp"
#include "raftrkt/array.h"

namespace cumlrkt::manifold {

ML::distance::DistanceType distance_of(int32_t metric);

matrix<float, int64_t> float_rows(const rr_view* view, const char* name);

void require_rows_fit(const char* name, const matrix<float, int64_t>& x);

void require_scanned(const char* name, const matrix<float, int64_t>& x,
                     int32_t metric, rmm::cuda_stream_view stream);

void require_converged(const char* name, const float* out, int64_t rows,
                       int64_t cols, const char* advice,
                       rmm::cuda_stream_view stream);

}  // namespace cumlrkt::manifold
