#include "manifold_support.hpp"

#include <cstdint>
#include <cuml/common/distance_type.hpp>
#include <format>
#include <limits>
#include <rmm/cuda_stream_view.hpp>

#include "cumlrkt/manifold.h"
#include "cumlrkt/rows.hpp"
#include "cumlrkt/support.hpp"
#include "raftrkt/array.h"
#include "raftrkt/view.hpp"

namespace cumlrkt::manifold {

namespace {

constexpr float least_float = std::numeric_limits<float>::min();

}  // namespace

ML::distance::DistanceType distance_of(int32_t metric) {
  using DistanceType = ML::distance::DistanceType;
  switch (metric) {
    case CUML_MANIFOLD_METRIC_EUCLIDEAN:
      return DistanceType::L2SqrtExpanded;
    case CUML_MANIFOLD_METRIC_SQEUCLIDEAN:
      return DistanceType::L2Expanded;
    case CUML_MANIFOLD_METRIC_MANHATTAN:
      return DistanceType::L1;
    case CUML_MANIFOLD_METRIC_CHEBYSHEV:
      return DistanceType::Linf;
    case CUML_MANIFOLD_METRIC_COSINE:
      return DistanceType::CosineExpanded;
    default:
      throw logic_error(
          "metric: expected euclidean, sqeuclidean, manhattan, chebyshev or "
          "cosine");
  }
}

matrix<float, int64_t> float_rows(const rr_view* view, const char* name) {
  const rr_view& v = *raftrkt::require(view, name);
  if (v.dtype != RR_DTYPE_FLOAT32) {
    throw logic_error(std::format("{}: expected float32, got {}", name,
                                  raftrkt::dtype_name(v.dtype)));
  }
  return rows_of<float, int64_t>(view, name, raftrkt::any_extent,
                                 raftrkt::any_extent);
}

void require_rows_fit(const char* name, const matrix<float, int64_t>& x) {
  require_rows(x.n, name);
  if (x.d > most_index) {
    throw logic_error(
        std::format("{}: {} columns do not fit the index type", name, x.d));
  }
}

void require_scanned(const char* name, const matrix<float, int64_t>& x,
                     int32_t metric, rmm::cuda_stream_view stream) {
  require_finite(name, x, stream);
  require_bounded_rows(name, x, stream);
  if (metric == CUML_MANIFOLD_METRIC_COSINE) {
    const float least = static_cast<float>(x.d) * least_float;
    const auto [row, squared] =
        first_normless_row(x.data, x.n, x.d, least, stream);
    if (row >= 0) {
      throw logic_error(std::format(
          "{}: row {}: expected a squared norm of at least {} under cosine, "
          "got {}",
          name, row, printed(least), printed<float>(squared)));
    }
  }
}

void require_converged(const char* name, const float* out, int64_t rows,
                       int64_t cols, const char* advice,
                       rmm::cuda_stream_view stream) {
  const int64_t i = first_non_finite(out, rows * cols, stream);
  if (i >= 0) {
    const float got = read_one(out, i, stream);
    throw logic_error(std::format(
        "{}: row {}, column {}: the optimisation diverged to {}; {}", name,
        i / cols, i % cols, printed(static_cast<double>(got)), advice));
  }
}

}  // namespace cumlrkt::manifold
