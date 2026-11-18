#pragma once

#include <cstdint>
#include <optional>
#include <raft/core/handle.hpp>
#include <vector>

#include "cumlrkt/labels.hpp"
#include "cumlrkt/support.hpp"
#include "glm_staging.hpp"
#include "logistic_fit.hpp"

namespace cumlrkt::glm::logistic {

struct weighted_rows {
  buffer<double> weights;
  buffer<int64_t> index;
  row_selection rows;
  bool uniform;
  double scale;
  bool sampled;
};

const double* weights_in(const std::optional<weighted_rows>& wt);

row_selection rows_in(const std::optional<weighted_rows>& wt, int64_t n);

template <typename T, typename L>
std::optional<std::vector<double>> class_weights_of(
    const raft::handle_t& handle, const fit_views& views,
    const fit_options& options, const T* w, const dense_labels<L>& labels,
    const std::vector<L>& classes);

template <typename T, typename L>
std::optional<weighted_rows> weighted_rows_of(
    const raft::handle_t& handle, const T* w, const dense_labels<L>& labels,
    const std::optional<std::vector<double>>& cw,
    const std::vector<L>& classes);

}  // namespace cumlrkt::glm::logistic
