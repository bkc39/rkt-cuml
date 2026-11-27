#pragma once

#include <cstdint>
#include <format>
#include <memory>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <type_traits>
#include <utility>

#include "cumlrkt/common.hpp"
#include "cumlrkt/labels.hpp"
#include "cumlrkt/support.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt {

template <typename L>
void require_integral(const L* y, int64_t n, rmm::cuda_stream_view stream) {
  if constexpr (!std::is_integral_v<L>) {
    const int64_t i = first_non_integral(y, n, stream);
    if (i >= 0) {
      throw logic_error(
          std::format("y: expected integral class labels, got {}",
                      printed(static_cast<double>(read_one(y, i, stream)))));
    }
  }
}

template <typename V>
void deposit_vector(const raft::handle_t& handle, rr_deposit* out,
                    buffer<V> values) {
  auto owned = std::make_unique<buffer<V>>(std::move(values));
  const rr_view region = raftrkt::region(*owned);
  settle(handle);
  raftrkt::deposit(out, "out", handle, std::move(owned), {region});
}

template <typename L>
void encode_into(const raft::handle_t& handle, const rr_view* yv,
                 const rr_view* classesv, rr_deposit* out) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const L* y = raftrkt::vector_data<L, int32_t>(yv, "y", raftrkt::any_extent);
  const int64_t n = raftrkt::require(yv, "y")->shape[0];
  const L* classes = raftrkt::vector_data<L, int32_t>(classesv, "classes",
                                                      raftrkt::any_extent);
  const int64_t k = raftrkt::require(classesv, "classes")->shape[0];
  buffer<int32_t> codes = scratch<int32_t>(n, stream);
  encode_against<L>(y, n, classes, k, codes.data(), stream);
  deposit_vector(handle, out, std::move(codes));
}

template <typename L>
void classes_into(const raft::handle_t& handle, const rr_view* yv,
                  rr_deposit* out) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const L* y = raftrkt::vector_data<L, int32_t>(yv, "y", raftrkt::any_extent);
  const int64_t n = raftrkt::require(yv, "y")->shape[0];
  require_integral(y, n, stream);
  dense_labels<L> labels = dense_codes(y, n, stream);
  deposit_vector(handle, out, std::move(labels.values));
}

inline int label_encode(void* handle, const rr_view* y, const rr_view* classes,
                        rr_deposit* out) {
  return guarded(handle, {{"y", y}, {"classes", classes}},
                 [&](const raft::handle_t& h) {
                   rr_deposit* const into = raftrkt::require(out, "out");
                   on_class_labels(y, "y", [&](auto tag) {
                     encode_into<decltype(tag)>(h, y, classes, into);
                   });
                 });
}

inline int label_classes(void* handle, const rr_view* y, rr_deposit* out) {
  return guarded(handle, {{"y", y}}, [&](const raft::handle_t& h) {
    rr_deposit* const into = raftrkt::require(out, "out");
    on_class_labels(y, "y",
                    [&](auto tag) { classes_into<decltype(tag)>(h, y, into); });
  });
}

}  // namespace cumlrkt

#define CUMLRKT_LABEL_ENTRIES(prefix)                                         \
  int cuml_##prefix##_label_encode(void* handle, const rr_view* y,            \
                                   const rr_view* classes, rr_deposit* out) { \
    return cumlrkt::label_encode(handle, y, classes, out);                    \
  }                                                                           \
  CUMLRKT_JOB(cuml_##prefix##_label_encode)                                   \
  int cuml_##prefix##_label_classes(void* handle, const rr_view* y,           \
                                    rr_deposit* out) {                        \
    return cumlrkt::label_classes(handle, y, out);                            \
  }                                                                           \
  CUMLRKT_JOB(cuml_##prefix##_label_classes)
