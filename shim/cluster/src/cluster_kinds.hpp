#pragma once

#include "cumlrkt/cluster.h"
#include "cumlrkt/model.hpp"

namespace cumlrkt::cluster {

inline constexpr model_kind hdbscan_kind{CUML_CLUSTER_MODEL_KIND_HDBSCAN,
                                         "hdbscan"};

}  // namespace cumlrkt::cluster
