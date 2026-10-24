#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <numbers>
#include <string>
#include <type_traits>
#include <vector>

#include "cluster_fixture.hpp"
#include "cumlrkt/cluster.h"
#include "kmeans_fixture.hpp"
#include "raftrkt/c_api.h"
#include "refusal_fixture.hpp"

namespace cumlrkt::test {

constexpr int64_t default_min_cluster_size = 5;
constexpr int64_t default_batch_size = 4096;
constexpr int32_t model_regions = 8;
constexpr int32_t plain_regions = 7;
constexpr int64_t blob_rows = 100;
constexpr int64_t blob_count = 3;
constexpr double blob_gap = 50;

struct hdbscan_call {
  const rr_view* x = nullptr;
  rr_deposit* model_out = nullptr;
  rr_deposit* out = nullptr;
  int64_t min_cluster_size = default_min_cluster_size;
  int64_t min_samples = default_min_cluster_size;
  double cluster_selection_epsilon = 0;
  int64_t max_cluster_size = 0;
  double alpha = 1;
  int32_t cluster_selection_method = CUML_CLUSTER_HDBSCAN_EOM;
  int32_t allow_single_cluster = 0;
  int64_t* opaque_bytes = nullptr;
};

template <typename T>
hdbscan_call with(hdbscan_call c, T hdbscan_call::* field,
                  std::type_identity_t<T> value) {
  c.*field = value;
  return c;
}

inline std::string hdbscan_fit(const resources& res, const hdbscan_call& c) {
  int64_t ignored = 0;
  return outcome(cuml_cluster_hdbscan_fit(
      res.handle, c.x, c.model_out, c.out, c.min_cluster_size, c.min_samples,
      c.cluster_selection_epsilon, c.max_cluster_size, c.alpha,
      c.cluster_selection_method, c.allow_single_cluster,
      c.opaque_bytes == nullptr ? &ignored : c.opaque_bytes));
}

inline std::string approximate_predict(const resources& res,
                                       const rr_view* model, const rr_view* x,
                                       const rr_view* points, rr_deposit* out) {
  return outcome(cuml_cluster_hdbscan_approximate_predict(res.handle, model, x,
                                                          points, out));
}

inline std::string membership_vector(const resources& res, const rr_view* model,
                                     const rr_view* x, const rr_view* points,
                                     rr_deposit* out,
                                     int64_t batch_size = default_batch_size) {
  return outcome(cuml_cluster_hdbscan_membership_vector(
      res.handle, model, x, points, out, batch_size));
}

inline std::string all_points(const resources& res, const rr_view* model,
                              const rr_view* x, rr_deposit* out,
                              int64_t batch_size = default_batch_size) {
  return outcome(cuml_cluster_hdbscan_all_points_membership_vectors(
      res.handle, model, x, out, batch_size));
}

inline std::string release_model(const resources& res, const rr_view* model) {
  return outcome(cuml_cluster_model_release(res.handle, model));
}

inline void expect_refused(const allocation_meter& meter,
                           const std::vector<const record*>& outs,
                           const std::string& status,
                           const std::string& message, int64_t most_bytes = 0) {
  EXPECT_EQ(status, message);
  EXPECT_EQ(cuml_cluster_last_error_kind(), RR_ERROR_LOGIC) << status;
  EXPECT_LE(meter.peak(), most_bytes) << status;
  EXPECT_EQ(meter.current(), 0) << status;
  for (const record* out : outs) {
    expect_untouched(*out);
  }
}

struct lcg {
  uint64_t state;

  double bits53() {
    state = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
    return static_cast<double>(state >> 11);
  }
};

inline values three_blobs(int64_t n_each = blob_rows) {
  values host;
  lcg draws{0x9e3779b97f4a7c15ULL};
  const auto uniform = [&draws] { return (draws.bits53() + 0.5) * 0x1p-53; };
  for (int64_t g = 0; g < blob_count; ++g) {
    for (int64_t i = 0; i < n_each; ++i) {
      for (int j = 0; j < 2; ++j) {
        const double r = std::sqrt(-2 * std::log(uniform()));
        host.push_back((static_cast<double>(g) * blob_gap) +
                       (r * std::cos(2 * std::numbers::pi * uniform())));
      }
    }
  }
  return host;
}

inline values jittered_pairs(int64_t count, int64_t side, double spacing) {
  values host;
  lcg draws{1};
  const auto jitter = [&draws] { return (draws.bits53() * 0x1p-52) - 1; };
  for (int64_t i = 0; i < count; ++i) {
    const auto column = static_cast<double>(i % side);
    const auto row = static_cast<double>(i / side);
    const double at = (column * spacing) + jitter();
    const double up = (row * spacing) + jitter();
    host.insert(host.end(), {at, up, at, up});
  }
  return host;
}

struct fitted_model {
  explicit fitted_model(const resources& res, const values& host,
                        int64_t n_cols = 2, int64_t min_samples = 5)
      : n(static_cast<int64_t>(host.size()) / n_cols),
        d(n_cols),
        x(upload<float>(res, {n, d}, host)),
        model(res),
        out(res),
        status(hdbscan_fit(res, {.x = &x->view,
                                 .model_out = &model.r,
                                 .out = &out.r,
                                 .min_samples = min_samples,
                                 .opaque_bytes = &opaque})) {}

  [[nodiscard]] const rr_view* anchor() const {
    return &model.r.regions[0];
  }
  [[nodiscard]] int64_t clusters() const {
    return model.r.regions[6].shape[0];
  }

  int64_t n;
  int64_t d;
  std::unique_ptr<device_array> x;
  record model;
  record out;
  int64_t opaque = 0;
  std::string status;
};

}  // namespace cumlrkt::test
