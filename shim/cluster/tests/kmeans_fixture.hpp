#pragma once

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cstddef>
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

struct record {
  explicit record(const resources& res) {
    r.handle = res.handle;
    r.device = 0;
    r.state = RR_DEPOSIT_OPEN;
  }
  ~record() {
    if (r.state == RR_DEPOSIT_FILLED) {
      r.release(r.owner);
    }
  }
  record(const record&) = delete;
  record& operator=(const record&) = delete;
  record(record&&) = delete;
  record& operator=(record&&) = delete;

  template <typename T>
  [[nodiscard]] std::vector<T> read(int32_t i) const {
    const rr_view& v = r.regions[i];
    std::vector<T> host(static_cast<std::size_t>(v.shape[0]) *
                        (v.rank == 2 ? v.shape[1] : 1));
    if (!host.empty()) {
      EXPECT_EQ(cudaMemcpy(host.data(), v.data, host.size() * sizeof(T),
                           cudaMemcpyDeviceToHost),
                cudaSuccess);
    }
    return host;
  }

  rr_deposit r{};
};

inline std::vector<int64_t> shapes(const record& d) {
  std::vector<int64_t> found{d.r.count};
  for (int32_t i = 0; i < d.r.count; ++i) {
    const rr_view& v = d.r.regions[i];
    found.push_back(v.dtype);
    found.insert(found.end(), v.shape, v.shape + v.rank);
  }
  return found;
}

template <typename T>
inline values download(const record& d, int32_t i) {
  const std::vector<T> host = d.template read<T>(i);
  return {host.begin(), host.end()};
}

struct fit_call {
  const rr_view* x = nullptr;
  const rr_view* sample_weight = nullptr;
  const rr_view* init = nullptr;
  rr_deposit* out = nullptr;
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
      res.handle, c.x, c.sample_weight, c.init, c.out, c.n_clusters,
      c.init_method, c.max_iter, c.tol, c.n_init, c.oversampling_factor,
      c.batch_samples, c.seed, &inertia, &n_iter);
  return {outcome(status), inertia, n_iter};
}

inline std::string predict(const resources& res, const rr_view* centroids,
                           const rr_view* x, rr_deposit* out,
                           int32_t batch_samples = default_batch) {
  return outcome(cuml_cluster_kmeans_predict(res.handle, centroids, x, out,
                                             batch_samples));
}

inline std::string score(const resources& res, const rr_view* centroids,
                         const rr_view* x, const rr_view* sample_weight,
                         int32_t batch_samples = default_batch) {
  double inertia = -1;
  return outcome(cuml_cluster_kmeans_score(
      res.handle, centroids, x, sample_weight, batch_samples, &inertia));
}

inline std::string transform(const resources& res, const rr_view* centroids,
                             const rr_view* x, rr_deposit* out,
                             int32_t batch_samples = default_batch) {
  return outcome(cuml_cluster_kmeans_transform(res.handle, centroids, x, out,
                                               batch_samples));
}

}  // namespace cumlrkt::test
