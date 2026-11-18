#include "logistic_weights.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <utility>
#include <vector>

#include "cumlrkt/labels.hpp"
#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "cumlrkt/vectors.hpp"
#include "glm_staging.hpp"
#include "glm_staging_device.hpp"
#include "logistic_device.hpp"
#include "logistic_fit.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt::glm::logistic {

namespace {

std::vector<double> given_weights(const fit_views& v, int64_t k,
                                  rmm::cuda_stream_view stream) {
  return to_host(
      raftrkt::vector_data<double, int32_t>(v.class_weight, "class-weight", k),
      k, stream);
}

template <typename L>
void require_weighted_classes(const std::vector<double>& totals,
                              const std::vector<L>& classes) {
  for (size_t c = 0; c < totals.size(); ++c) {
    if (!(totals[c] > 0)) {
      throw logic_error(
          std::format("y: class {} has no weight", printed(classes[c])));
    }
  }
}

}  // namespace

const double* weights_in(const std::optional<weighted_rows>& wt) {
  return wt ? wt->weights.data() : nullptr;
}

row_selection rows_in(const std::optional<weighted_rows>& wt, int64_t n) {
  return wt ? wt->rows : row_selection{nullptr, n};
}

template <typename T, typename L>
std::optional<std::vector<double>> class_weights_of(
    const raft::handle_t& handle, const fit_views& views,
    const fit_options& options, const T* w, const dense_labels<L>& labels,
    const std::vector<L>& classes) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  if (options.class_mode == CUML_LINEAR_MODEL_CLASS_WEIGHT_GIVEN) {
    return given_weights(views, labels.k, stream);
  }
  if (options.class_mode != CUML_LINEAR_MODEL_CLASS_WEIGHT_BALANCED) {
    return std::nullopt;
  }
  const auto n = static_cast<int64_t>(labels.codes.size());
  const buffer<double> base =
      effective_weights<T>(handle, w, labels.codes.data(), nullptr, n);
  buffer<int64_t> index = scratch<int64_t>(n, stream);
  const int64_t m = positive_weights(base.data(), n, index.data(), stream);
  const row_selection rows{m < n ? index.data() : nullptr, m};
  const std::vector<double> weighted =
      class_totals(stream, base.data(), labels.codes.data(), rows, labels.k);
  require_weighted_classes(weighted, classes);
  const bool counted = w != nullptr && equal_weights(base.data(), rows, stream);
  const std::vector<double> totals =
      counted
          ? class_totals(stream, nullptr, labels.codes.data(), rows, labels.k)
          : weighted;
  double total = 0;
  for (const double t : totals) {
    total += t;
  }
  std::vector<double> cw(totals.size());
  std::ranges::transform(totals, cw.begin(), [&](double t) {
    return total / (static_cast<double>(totals.size()) * t);
  });
  return cw;
}

template <typename T, typename L>
std::optional<weighted_rows> weighted_rows_of(
    const raft::handle_t& handle, const T* w, const dense_labels<L>& labels,
    const std::optional<std::vector<double>>& cw,
    const std::vector<L>& classes) {
  if (w == nullptr && !cw) {
    return std::nullopt;
  }
  const rmm::cuda_stream_view stream = stream_of(handle);
  const auto n = static_cast<int64_t>(labels.codes.size());
  std::optional<buffer<double>> device_cw;
  if (cw) {
    device_cw.emplace(to_device(*cw, stream));
  }
  buffer<double> eff =
      effective_weights<T>(handle, w, labels.codes.data(),
                           device_cw ? device_cw->data() : nullptr, n);
  require_weighted_classes(class_totals(stream, eff.data(), labels.codes.data(),
                                        row_selection{nullptr, n}, labels.k),
                           classes);
  buffer<int64_t> index = scratch<int64_t>(n, stream);
  const int64_t m = positive_weights(eff.data(), n, index.data(), stream);
  const row_selection rows{m < n ? index.data() : nullptr, m};
  const bool uniform = equal_weights(eff.data(), rows, stream);
  if (labels.k > 2 && !uniform) {
    throw logic_error(
        "sample-weight: weights with 3 or more classes must be equal on every "
        "row (cuML's softmax ignores them); got unequal weights");
  }
  const double scale =
      static_cast<double>(m) / weight_sum(eff.data(), rows, stream);
  return weighted_rows{
      .weights = std::move(eff),
      .index = std::move(index),
      .rows = rows,
      .uniform = uniform,
      .scale = scale,
      .sampled = w != nullptr,
  };
}

#define CUMLRKT_LOGISTIC_WEIGHTS(T, L)                                       \
  template std::optional<std::vector<double>> class_weights_of<T, L>(        \
      const raft::handle_t&, const fit_views&, const fit_options&, const T*, \
      const dense_labels<L>&, const std::vector<L>&);                        \
  template std::optional<weighted_rows> weighted_rows_of<T, L>(              \
      const raft::handle_t&, const T*, const dense_labels<L>&,               \
      const std::optional<std::vector<double>>&, const std::vector<L>&);

#define CUMLRKT_LOGISTIC_WEIGHTS_OF(T) \
  CUMLRKT_LOGISTIC_WEIGHTS(T, int32_t) \
  CUMLRKT_LOGISTIC_WEIGHTS(T, int64_t) \
  CUMLRKT_LOGISTIC_WEIGHTS(T, float)   \
  CUMLRKT_LOGISTIC_WEIGHTS(T, double)

CUMLRKT_LOGISTIC_WEIGHTS_OF(float)
CUMLRKT_LOGISTIC_WEIGHTS_OF(double)

}  // namespace cumlrkt::glm::logistic
