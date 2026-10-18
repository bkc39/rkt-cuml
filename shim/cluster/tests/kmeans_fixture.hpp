#pragma once

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cluster_fixture.hpp"
#include "cumlrkt/cluster.h"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {

using values = std::vector<double>;

constexpr int32_t pair = 2;
constexpr int32_t default_batch = 32768;
constexpr double default_tol = 1e-4;
constexpr int32_t default_max_iter = 300;

template <typename T>
constexpr int32_t float_dtype =
    std::is_same_v<T, float> ? RR_DTYPE_FLOAT32 : RR_DTYPE_FLOAT64;

template <typename T>
inline std::unique_ptr<device_array> upload(const resources& res,
                                            std::vector<int64_t> shape,
                                            const values& host) {
  auto a =
      std::make_unique<device_array>(res.r, float_dtype<T>, std::move(shape));
  a->write(std::vector<T>(host.begin(), host.end()));
  return a;
}

template <typename T>
inline values download(const device_array& a) {
  const std::vector<T> host = a.read<T>();
  return {host.begin(), host.end()};
}

struct fit_call {
  const rr_view* x = nullptr;
  const rr_view* sample_weight = nullptr;
  const rr_view* init = nullptr;
  const rr_view* centroids = nullptr;
  const rr_view* labels = nullptr;
  int32_t n_clusters = pair;
  int32_t init_method = CUML_CLUSTER_KMEANS_INIT_ARRAY;
  int32_t max_iter = default_max_iter;
  double tol = default_tol;
  int32_t n_init = 1;
  double oversampling_factor = 2.0;
  int32_t batch_samples = default_batch;
  uint64_t seed = 0;
};

struct fit_outcome {
  std::string status;
  double inertia;
  int32_t n_iter;
};

inline fit_outcome fit(const resources& res, const fit_call& c) {
  double inertia = -1;
  int32_t n_iter = -1;
  const int status = cuml_cluster_kmeans_fit(
      res.handle, c.x, c.sample_weight, c.init, c.centroids, c.labels,
      c.n_clusters, c.init_method, c.max_iter, c.tol, c.n_init,
      c.oversampling_factor, c.batch_samples, c.seed, &inertia, &n_iter);
  return {outcome(status), inertia, n_iter};
}

inline std::string predict(const resources& res, const rr_view* centroids,
                           const rr_view* x, const rr_view* sample_weight,
                           const rr_view* labels,
                           int32_t batch_samples = default_batch) {
  double inertia = -1;
  return outcome(cuml_cluster_kmeans_predict(res.handle, centroids, x,
                                             sample_weight, labels,
                                             batch_samples, &inertia));
}

inline std::string transform(const resources& res, const rr_view* centroids,
                             const rr_view* x, const rr_view* distances,
                             int32_t batch_samples = default_batch) {
  return outcome(cuml_cluster_kmeans_transform(res.handle, centroids, x,
                                               distances, batch_samples));
}

}  // namespace cumlrkt::test
