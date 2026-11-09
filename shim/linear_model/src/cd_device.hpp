#pragma once

#include <cstdint>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <vector>

#include "cumlrkt/support.hpp"
#include "glm_staging.hpp"
#include "glm_staging_device.hpp"

namespace cumlrkt::glm {

struct cd_centres {
  std::vector<double> mean;
  std::vector<double> remainder;
  centre y;
};

template <typename T>
cd_centres centres_of(const T* y, const T* w, row_selection rows,
                      const staging& s, bool fit_intercept,
                      rmm::cuda_stream_view stream);

template <typename T>
double weight_total(const T* w, row_selection rows,
                    rmm::cuda_stream_view stream);

template <typename T>
staged_problem<T> stage_weighted(const raft::handle_t& handle,
                                 const matrix<T, int64_t>& x, const T* y,
                                 const T* w, row_selection rows,
                                 const staging& s, const cd_centres& c,
                                 double total);

}  // namespace cumlrkt::glm
