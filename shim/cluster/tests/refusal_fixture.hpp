#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda/memory_resource>
#include <initializer_list>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cluster_fixture.hpp"
#include "kmeans_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {

template <typename T>
using other_t = std::conditional_t<std::is_same_v<T, float>, double, float>;

template <typename T>
inline std::string name() {
  return std::is_same_v<T, float> ? "float32" : "float64";
}

inline rr_view shaped(const device_array& a,
                      std::initializer_list<int64_t> shape,
                      bool col_major = false) {
  rr_view v = a.view;
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

inline rr_view on_device(rr_view v, int32_t device) {
  v.device = device;
  return v;
}

constexpr std::size_t refusal_pool = 64;
constexpr double float_overflow = 1e20;

template <typename T>
struct refusal_case {
  refusal_case(const resources& res, int64_t rows, int64_t cols)
      : data(res.r, float_dtype<T>, {refusal_pool}),
        other(res.r, float_dtype<other_t<T>>, {refusal_pool}),
        ints(res.r, RR_DTYPE_INT32, {refusal_pool}),
        x(shaped(data, {rows, cols})) {
    data.write(std::vector<T>(refusal_pool, 1));
  }

  device_array data;
  device_array other;
  device_array ints;
  rr_view x;
};

template <typename Call>
struct refusal {
  Call call;
  std::string message;
};

inline values with_row(const std::vector<double>& row, int64_t at, int64_t rows,
                       int64_t cols) {
  values host(static_cast<std::size_t>(rows * cols), 1.0);
  std::ranges::copy(row, host.begin() + static_cast<std::ptrdiff_t>(at * cols));
  return host;
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

class pool_cap {
 public:
  explicit pool_cap(std::size_t limit)
      : previous_(rmm::mr::set_current_device_resource(
            rmm::mr::cuda_memory_resource{})),
        cap_(previous_, limit) {
    rmm::mr::set_current_device_resource(cap_);
  }
  ~pool_cap() {
    rmm::mr::set_current_device_resource(std::move(previous_));
  }
  pool_cap(const pool_cap&) = delete;
  pool_cap& operator=(const pool_cap&) = delete;
  pool_cap(pool_cap&&) = delete;
  pool_cap& operator=(pool_cap&&) = delete;

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> previous_;
  rmm::mr::limiting_resource_adaptor cap_;
};

inline void expect_untouched(const record& out) {
  EXPECT_EQ(out.r.state, RR_DEPOSIT_OPEN);
  EXPECT_EQ(out.r.count, 0);
  EXPECT_EQ(out.r.owner, nullptr);
}

}  // namespace cumlrkt::test
