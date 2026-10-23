#pragma once

#include <cstdint>
#include <string>

#include "cluster_fixture.hpp"
#include "cumlrkt/cluster.h"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {

constexpr int32_t default_c = 15;

struct linkage_call {
  const rr_view* x = nullptr;
  rr_deposit* out = nullptr;
  int64_t n_clusters = 2;
  int32_t metric = CUML_CLUSTER_LINKAGE_METRIC_EUCLIDEAN;
  int32_t connectivity = CUML_CLUSTER_LINKAGE_CONNECTIVITY_KNN;
  int32_t c = default_c;
};

inline std::string linkage(const resources& res, const linkage_call& c) {
  return outcome(cuml_cluster_single_linkage_fit(
      res.handle, c.x, c.out, c.n_clusters, c.metric, c.connectivity, c.c));
}

constexpr int32_t pairwise = CUML_CLUSTER_LINKAGE_CONNECTIVITY_PAIRWISE;
constexpr int32_t knn = CUML_CLUSTER_LINKAGE_CONNECTIVITY_KNN;

}  // namespace cumlrkt::test
