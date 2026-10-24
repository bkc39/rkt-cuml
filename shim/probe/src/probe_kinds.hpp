#pragma once

#include "cumlrkt/model.hpp"
#include "cumlrkt/probe.h"

namespace cumlrkt {

inline constexpr model_kind probe_a{CUML_PROBE_KIND_A, "probe-a"};
inline constexpr model_kind probe_b{CUML_PROBE_KIND_B, "probe-b"};

}  // namespace cumlrkt
