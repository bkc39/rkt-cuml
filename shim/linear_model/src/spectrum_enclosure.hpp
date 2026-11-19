#pragma once

#include <cstdint>
#include <optional>
#include <raft/core/handle.hpp>
#include <vector>

#include "ritz.hpp"

namespace cumlrkt::glm {

ritz::enclosure_norms enclosure_norms_of(const raft::handle_t& handle,
                                         const std::vector<float>& m,
                                         const ritz::eigen_pairs& e, int64_t d);

std::optional<ritz::spectrum_interval> enclosed_spectrum(
    const raft::handle_t& handle, const std::vector<float>& m,
    const ritz::eigen_pairs& e, int64_t d);

std::optional<double> enclosed_kappa(const raft::handle_t& handle,
                                     const std::vector<float>& m,
                                     const ritz::eigen_pairs& e,
                                     const ritz::gram_plan& plan,
                                     ritz::gram_shape shape, double limit);

}  // namespace cumlrkt::glm
