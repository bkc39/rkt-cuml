#pragma once

#include <array>
#include <cstddef>
#include <cuml/version_config.hpp>
#include <new>
#include <string>
#include <type_traits>

#include "raftrkt/abi.h"
#include "raftrkt/array.h"
#include "raftrkt/error.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt {

using raftrkt::logic_error;

inline raftrkt::error_slot& last_error_slot() noexcept {
  thread_local raftrkt::error_slot slot;
  return slot;
}

template <typename T>
T& never_destroyed() noexcept {
  alignas(T) static std::byte storage[sizeof(T)];
  static T* const shared = ::new (static_cast<void*>(storage)) T;
  return *shared;
}

constexpr int decimal_base = 10;
constexpr std::size_t release_length = 9;

using release_text = std::array<char, release_length>;

constexpr release_text rapids_release(int major, int minor, int patch) {
  const auto digit = [](int n) { return static_cast<char>('0' + n); };
  return {digit(major / decimal_base), digit(major % decimal_base), '.',
          digit(minor / decimal_base), digit(minor % decimal_base), '.',
          digit(patch / decimal_base), digit(patch % decimal_base), '\0'};
}

inline constexpr release_text libcuml_release =
    rapids_release(CUML_VERSION_MAJOR, CUML_VERSION_MINOR, CUML_VERSION_PATCH);

inline int check_abi(const rr_abi_tag* loaded) noexcept {
  return raftrkt::translate_exceptions(last_error_slot(),
                                       [&] { raftrkt::require_abi(loaded); });
}

template <typename Fn>
void on_floats(const rr_view* x, const char* name, Fn&& fn) {
  const rr_view& view = *raftrkt::require(x, name);
  if (view.dtype == RR_DTYPE_FLOAT32) {
    fn(float{});
  } else if (view.dtype == RR_DTYPE_FLOAT64) {
    fn(double{});
  } else {
    throw logic_error(std::string(name) +
                      ": expected float32 or float64, got " +
                      raftrkt::dtype_name(view.dtype));
  }
}

template <typename Fn>
void on_same_floats(const rr_view* a, const char* a_name, const rr_view* b,
                    const char* b_name, Fn&& fn) {
  on_floats(a, a_name, [&](auto a_tag) {
    on_floats(b, b_name, [&](auto b_tag) {
      using A = decltype(a_tag);
      using B = decltype(b_tag);
      if constexpr (std::is_same_v<A, B>) {
        fn(a_tag);
      } else {
        throw logic_error(std::string(a_name) + ": expected " +
                          raftrkt::dtype_name(raftrkt::dtype_code<B>) +
                          ", got " +
                          raftrkt::dtype_name(raftrkt::dtype_code<A>));
      }
    });
  });
}

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

}  // namespace cumlrkt
