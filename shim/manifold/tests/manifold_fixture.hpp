#pragma once

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cumlrkt/manifold.h"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {

inline std::string gpu_unavailable_reason() {
  int32_t count = 0;
  if (rr_device_count(&count) != RR_OK) {
    return std::string("no GPU: ") + rr_last_error();
  }
  return count == 0 ? "no GPU: no CUDA device" : "";
}

#define CUMLRKT_REQUIRE_GPU()                            \
  do {                                                   \
    const std::string cumlrkt_reason =                   \
        cumlrkt::test::gpu_unavailable_reason();         \
    if (!cumlrkt_reason.empty()) {                       \
      std::printf("SKIP: %s\n", cumlrkt_reason.c_str()); \
      GTEST_SKIP() << cumlrkt_reason;                    \
    }                                                    \
  } while (0)

struct resources {
  resources() {
    EXPECT_EQ(rr_resources_create(0, &r), RR_OK) << rr_last_error();
    EXPECT_EQ(rr_resources_handle(r, &handle), RR_OK) << rr_last_error();
  }
  ~resources() {
    rr_resources_free(r);
  }
  resources(const resources&) = delete;
  resources& operator=(const resources&) = delete;
  resources(resources&&) = delete;
  resources& operator=(resources&&) = delete;
  rr_resources* r = nullptr;
  void* handle = nullptr;
};

constexpr std::size_t wide_item = 8;
constexpr std::size_t narrow_item = 4;

inline std::size_t itemsize(int32_t dtype) {
  return dtype == RR_DTYPE_FLOAT64 || dtype == RR_DTYPE_INT64 ? wide_item
                                                              : narrow_item;
}

template <typename T>
constexpr int32_t dtype_of = std::is_same_v<T, float>     ? RR_DTYPE_FLOAT32
                             : std::is_same_v<T, double>  ? RR_DTYPE_FLOAT64
                             : std::is_same_v<T, int32_t> ? RR_DTYPE_INT32
                                                          : RR_DTYPE_INT64;

inline rr_view shaped(rr_view v, std::initializer_list<int64_t> shape,
                      bool col_major = false) {
  v.rank = static_cast<int32_t>(shape.size());
  int32_t i = 0;
  for (const int64_t extent : shape) {
    v.shape[i++] = extent;
  }
  if (v.rank == 1) {
    v.strides[0] = 1;
  } else if (col_major) {
    v.strides[0] = 1;
    v.strides[1] = v.shape[0];
  } else {
    v.strides[0] = v.shape[1];
    v.strides[1] = 1;
  }
  return v;
}

struct device_array {
  device_array(rr_resources* r, int32_t dtype, std::vector<int64_t> shape) {
    std::size_t count = 1;
    for (const int64_t extent : shape) {
      count *= static_cast<std::size_t>(extent);
    }
    EXPECT_EQ(
        rr_buffer_alloc(r, std::max<std::size_t>(count, 1) * itemsize(dtype),
                        &buffer),
        RR_OK)
        << rr_last_error();
    view.dtype = dtype;
    view.rank = static_cast<int32_t>(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i) {
      view.shape[i] = shape[i];
    }
    if (shape.size() == 1) {
      view.strides[0] = 1;
    } else {
      view.strides[0] = shape[1];
      view.strides[1] = 1;
    }
    EXPECT_EQ(rr_buffer_view(buffer, 0, &view), RR_OK) << rr_last_error();
  }
  ~device_array() {
    rr_buffer_free(buffer);
  }
  device_array(const device_array&) = delete;
  device_array& operator=(const device_array&) = delete;
  device_array(device_array&&) = delete;
  device_array& operator=(device_array&&) = delete;

  template <typename T>
  void write(const std::vector<T>& host) const {
    if (!host.empty()) {
      EXPECT_EQ(rr_copy_h2d(buffer, host.data(), host.size() * sizeof(T)),
                RR_OK)
          << rr_last_error();
    }
  }

  rr_buffer* buffer = nullptr;
  rr_view view{};
};

template <typename T>
struct filled {
  filled(const resources& res, const std::vector<T>& host,
         std::vector<int64_t> shape)
      : array(res.r, dtype_of<T>, std::move(shape)) {
    array.write(host);
  }
  device_array array;
  [[nodiscard]] const rr_view* view() const {
    return &array.view;
  }
};

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

inline void expect_untouched(const record& out) {
  EXPECT_EQ(out.r.state, RR_DEPOSIT_OPEN);
  EXPECT_EQ(out.r.count, 0);
  EXPECT_EQ(out.r.owner, nullptr);
}

inline std::string outcome(int status) {
  return status == RR_OK ? std::string("accepted")
                         : std::string(cuml_manifold_last_error());
}

struct umap_options {
  int64_t n_neighbors = 15;
  int64_t n_components = 2;
  int32_t epochs_given = 0;
  int64_t n_epochs = 0;
  double learning_rate = 1.0;
  double a = 1.5769434603113077;
  double b = 0.8950608781227859;
  int32_t init = CUML_MANIFOLD_INIT_RANDOM;
  int64_t negative_sample_rate = 5;
  uint64_t random_state = 0;
  int32_t deterministic = 1;
  int32_t build_algo = CUML_MANIFOLD_BUILD_BRUTE_FORCE;
  int32_t metric = CUML_MANIFOLD_METRIC_EUCLIDEAN;
};

inline int umap_fit(const resources& res, const rr_view* x, const rr_view* y,
                    record& out, const umap_options& o = {}) {
  return cuml_manifold_umap_fit(
      res.handle, x, y, &out.r, o.n_neighbors, o.n_components, o.epochs_given,
      o.n_epochs, o.learning_rate, o.a, o.b, o.init, o.negative_sample_rate,
      o.random_state, o.deterministic, o.build_algo, o.metric);
}

struct umap_trained {
  const rr_view* x;
  const rr_view* embedding;
};

inline int umap_transform(const resources& res, const rr_view* x,
                          const umap_trained& t, record& out,
                          const umap_options& o = {}) {
  return cuml_manifold_umap_transform(
      res.handle, x, t.x, t.embedding, &out.r, o.n_neighbors, o.n_components,
      o.epochs_given, o.n_epochs, o.learning_rate, o.a, o.b,
      o.negative_sample_rate, o.random_state, o.deterministic, o.metric);
}

struct tsne_options {
  int64_t n_components = 2;
  double perplexity = 5.0;
  double early_exaggeration = 12.0;
  double learning_rate = 200.0;
  int64_t max_iter = 1000;
  int64_t n_neighbors = 15;
  int32_t method = CUML_MANIFOLD_TSNE_FFT;
  int32_t metric = CUML_MANIFOLD_METRIC_EUCLIDEAN;
  int32_t init = CUML_MANIFOLD_TSNE_INIT_RANDOM;
  int64_t random_state = 0;
};

struct tsne_stats {
  double kl_divergence = -1;
  int64_t n_iter = -1;
};

inline int tsne_fit(const resources& res, const rr_view* x, record& out,
                    const tsne_options& o = {}, tsne_stats* stats = nullptr) {
  tsne_stats ignored;
  tsne_stats* s = stats == nullptr ? &ignored : stats;
  return cuml_manifold_tsne_fit(
      res.handle, x, &out.r, o.n_components, o.perplexity, o.early_exaggeration,
      o.learning_rate, o.max_iter, o.n_neighbors, o.method, o.metric, o.init,
      o.random_state, &s->kl_divergence, &s->n_iter);
}

constexpr uint64_t blob_modulus = 7;
constexpr float blob_gap = 10.0F;
constexpr float blob_step = 0.01F;

inline std::vector<float> blobs(int64_t n, int64_t d) {
  std::vector<float> host(static_cast<std::size_t>(n * d));
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < d; ++j) {
      const auto at = static_cast<std::size_t>((i * d) + j);
      const auto jitter = static_cast<float>(
          (static_cast<uint64_t>(at) * static_cast<uint64_t>(at)) %
          blob_modulus);
      host[at] = (static_cast<float>(i % 2) * blob_gap) + (jitter * blob_step);
    }
  }
  return host;
}

constexpr double spread_step = 0x1p44;

inline std::vector<float> spread_rows(double a) {
  constexpr int64_t n = 16;
  constexpr int64_t d = 3;
  constexpr int64_t thirds = 3;
  std::vector<float> host(static_cast<std::size_t>(n * d));
  for (int64_t i = 0; i < n; ++i) {
    const auto at = static_cast<std::size_t>(i * d);
    host[at] = static_cast<float>(i % 2 == 0 ? a : -a);
    host[at + 1] = static_cast<float>(static_cast<double>(i / 2) * spread_step);
    host[at + 2] =
        static_cast<float>(static_cast<double>(i % thirds) * spread_step);
  }
  return host;
}

}  // namespace cumlrkt::test
