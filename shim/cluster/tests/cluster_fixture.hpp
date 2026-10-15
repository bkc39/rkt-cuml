#pragma once

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "cumlrkt/cluster.h"
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

struct device_array {
  device_array(rr_resources* r, int32_t dtype, std::vector<int64_t> shape) {
    std::size_t count = 1;
    for (const int64_t extent : shape) {
      count *= static_cast<std::size_t>(extent);
    }
    EXPECT_EQ(rr_buffer_alloc(r, count * itemsize(dtype), &buffer), RR_OK)
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
  std::vector<T> read() const {
    std::vector<T> host(static_cast<std::size_t>(view.shape[0]) *
                        (view.rank == 2 ? view.shape[1] : 1));
    EXPECT_EQ(rr_copy_d2h(host.data(), buffer, host.size() * sizeof(T)), RR_OK)
        << rr_last_error();
    return host;
  }

  rr_buffer* buffer = nullptr;
  rr_view view{};
};

inline std::string outcome(int status) {
  return status == RR_OK ? std::string("accepted")
                         : std::string(cuml_cluster_last_error());
}

}  // namespace cumlrkt::test
