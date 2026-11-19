#pragma once

#include <cstdint>
#include <optional>
#include <raft/core/handle.hpp>
#include <vector>

#include "ritz.hpp"

namespace cumlrkt::glm {

template <typename S>
std::optional<ritz::eigen_pairs> eigen_of(const raft::handle_t& handle,
                                          const std::vector<S>& m, int64_t d,
                                          bool vectors);

std::optional<ritz::eigen_pairs> pencil_of(const raft::handle_t& handle,
                                           const std::vector<double>& h,
                                           const std::vector<double>& s,
                                           int64_t d);

}  // namespace cumlrkt::glm
