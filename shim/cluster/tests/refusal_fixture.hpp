#pragma once

#include <gtest/gtest.h>

#include <cstdint>
#include <cuda/memory_resource>
#include <initializer_list>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <string>
#include <type_traits>
#include <utility>

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

inline rr_view shaped(const device_array& a, std::initializer_list<int64_t> shape,
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

inline void expect_untouched(const record& out) {
  EXPECT_EQ(out.r.state, RR_DEPOSIT_OPEN);
  EXPECT_EQ(out.r.count, 0);
  EXPECT_EQ(out.r.owner, nullptr);
}

}  // namespace cumlrkt::test
