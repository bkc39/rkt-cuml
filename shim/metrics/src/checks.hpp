#pragma once

#include <cstdint>
#include <format>
#include <string>

#include "cumlrkt/common.hpp"
#include "cumlrkt/limits.hpp"
#include "raftrkt/array.h"
#include "raftrkt/view.hpp"

namespace cumlrkt::metrics {

inline constexpr int64_t most_distinct_labels = 46340;

template <typename Fn>
void on_labels(const rr_view* labels, const char* name, Fn&& fn) {
  const rr_view& view = *raftrkt::require(labels, name);
  if (view.dtype == RR_DTYPE_INT32) {
    fn(int32_t{});
  } else if (view.dtype == RR_DTYPE_INT64) {
    fn(int64_t{});
  } else {
    throw logic_error(std::string(name) + ": expected int32 or int64, got " +
                      raftrkt::dtype_name(view.dtype));
  }
}

template <typename T>
const T* labels_of(const rr_view* labels, const char* name, int64_t n) {
  return raftrkt::vector_data<T, int32_t>(labels, name, n);
}

inline void require_entries(const std::string& what, int64_t rows,
                            int64_t each) {
  int64_t entries = 0;
  if (__builtin_mul_overflow(rows, each, &entries)) {
    throw logic_error(std::format("{} need more than {} entries", what,
                                  most_index));
  }
  if (entries > most_index) {
    throw logic_error(std::format("{} need {} entries, more than {}", what,
                                  entries, most_index));
  }
}

}  // namespace cumlrkt::metrics
