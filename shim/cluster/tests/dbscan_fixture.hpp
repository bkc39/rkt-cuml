#pragma once

#include <cstdint>
#include <string>

#include "cluster_fixture.hpp"
#include "cumlrkt/cluster.h"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {

constexpr double default_eps = 0.5;
constexpr int32_t default_min_samples = 5;

struct dbscan_call {
  const rr_view* x = nullptr;
  const rr_view* sample_weight = nullptr;
  rr_deposit* out = nullptr;
  double eps = default_eps;
  int32_t min_samples = default_min_samples;
  int32_t metric = CUML_CLUSTER_DBSCAN_METRIC_EUCLIDEAN;
  int32_t out_dtype = RR_DTYPE_INT32;
  int64_t max_mbytes_per_batch = 0;
};

inline std::string dbscan(const resources& res, const dbscan_call& c) {
  return outcome(cuml_cluster_dbscan_fit(
      res.handle, c.x, c.sample_weight, c.out, c.eps, c.min_samples, c.metric,
      c.out_dtype, c.max_mbytes_per_batch));
}

}  // namespace cumlrkt::test
