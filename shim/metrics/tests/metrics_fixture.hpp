#pragma once

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda/memory_resource>
#include <initializer_list>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cumlrkt/metrics.h"
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
                         : std::string(cuml_metrics_last_error());
}

class allocation_meter {
 public:
  allocation_meter()
      : previous_(rmm::mr::set_current_device_resource(
            rmm::mr::cuda_memory_resource{})),
        stats_(previous_) {
    rmm::mr::set_current_device_resource(stats_);
  }
  ~allocation_meter() {
    rmm::mr::set_current_device_resource(std::move(previous_));
  }
  allocation_meter(const allocation_meter&) = delete;
  allocation_meter& operator=(const allocation_meter&) = delete;
  allocation_meter(allocation_meter&&) = delete;
  allocation_meter& operator=(allocation_meter&&) = delete;

  [[nodiscard]] int64_t peak() const {
    return stats_.get_bytes_counter().peak;
  }
  [[nodiscard]] int64_t current() const {
    return stats_.get_bytes_counter().value;
  }

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> previous_;
  rmm::mr::statistics_resource_adaptor stats_;
};

}  // namespace cumlrkt::test
