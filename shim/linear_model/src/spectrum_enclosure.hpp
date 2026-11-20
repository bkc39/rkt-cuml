#pragma once

#include <cstdint>
#include <optional>
#include <raft/core/handle.hpp>
#include <vector>

#include "ritz.hpp"

namespace cumlrkt::glm {

struct window_inputs {
  const std::vector<double>& basis;
  const std::vector<double>& applied;
  const std::vector<double>& crossed;
  int64_t window;
  int64_t vectors;
};

struct window_errors {
  double gram;
  double applied;
  double mean;
};

template <typename S>
ritz::enclosure_norms enclosure_norms_of(const raft::handle_t& handle,
                                         const std::vector<S>& m,
                                         const ritz::eigen_pairs& e, int64_t d);

std::optional<ritz::spectrum_interval> enclosed_spectrum(
    const raft::handle_t& handle, const std::vector<float>& m,
    const ritz::eigen_pairs& e, int64_t d);

std::optional<ritz::spectrum_interval> window_spectrum(
    const raft::handle_t& handle, const std::vector<float>& m,
    const ritz::eigen_pairs& e, window_inputs w, window_errors r);

std::optional<double> enclosed_kappa(const raft::handle_t& handle,
                                     const std::vector<float>& m,
                                     const ritz::eigen_pairs& e,
                                     const ritz::gram_plan& plan,
                                     ritz::gram_shape shape, double limit,
                                     double gram);

}  // namespace cumlrkt::glm
