#pragma once

#include <cuda_runtime_api.h>
#include <cuml/svm/svm_model.h>

#include <cstdint>
#include <cstdio>
#include <cuml/svm/svc.hpp>
#include <exception>
#include <raft/core/handle.hpp>
#include <string_view>
#include <utility>

#include "cumlrkt/support.hpp"
#include "cumlrkt/transpose.hpp"
#include "raftrkt/error.hpp"
#include "svm_device.hpp"

namespace cumlrkt::svm {

template <typename T>
class model_guard {
 public:
  model_guard(const raft::handle_t& handle, ML::SVM::SvmModel<T>& model)
      : handle_(handle), model_(model) {}
  ~model_guard() noexcept {
    try {
      ML::SVM::svmFreeBuffers(handle_, model_);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "cumlrkt: svmFreeBuffers: %s\n", e.what());
    } catch (...) {
      std::fputs("cumlrkt: svmFreeBuffers threw\n", stderr);
    }
  }
  model_guard(const model_guard&) = delete;
  model_guard& operator=(const model_guard&) = delete;
  model_guard(model_guard&&) = delete;
  model_guard& operator=(model_guard&&) = delete;

 private:
  const raft::handle_t& handle_;
  ML::SVM::SvmModel<T>& model_;
};

template <typename T>
struct svc_parts {
  buffer<T> support_vectors;
  buffer<int32_t> support;
  buffer<T> dual;
  buffer<T> intercept;
  int64_t n_support;
  int64_t n_negative;
  int64_t n_positive;
};

template <typename T>
void copy_device(const T* from, int64_t count, T* to,
                 rmm::cuda_stream_view stream) {
  if (count > 0) {
    raftrkt::cuda_check(
        cudaMemcpyAsync(to, from, static_cast<std::size_t>(count) * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream.value()),
        "cudaMemcpyAsync");
  }
}

template <typename T>
svc_parts<T> copy_out(const raft::handle_t& handle,
                      const ML::SVM::SvmModel<T>& model) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const int64_t ns = model.n_support;
  const int64_t d = model.n_cols;
  svc_parts<T> parts{scratch<T>(ns * d, stream),
                     scratch<int32_t>(ns, stream),
                     scratch<T>(ns, stream),
                     scratch<T>(1, stream),
                     ns,
                     0,
                     0};
  to_rows(handle, model.support_matrix.data, ns, d,
          static_cast<const T*>(nullptr), parts.support_vectors.data());
  copy_device(model.support_idx, ns, parts.support.data(), stream);
  copy_device(model.dual_coefs, ns, parts.dual.data(), stream);
  raftrkt::cuda_check(
      cudaMemcpyAsync(parts.intercept.data(), &model.b, sizeof(T),
                      cudaMemcpyHostToDevice, stream.value()),
      "cudaMemcpyAsync");
  const auto [negative, positive] = sign_counts(parts.dual.data(), ns, stream);
  parts.n_negative = negative;
  parts.n_positive = positive;
  settle(handle);
  return parts;
}

template <typename T>
ML::SVM::SvmModel<T> rebuilt(int n_support, int n_cols, T b, T* dual,
                             T* support_columns) {
  ML::SVM::SvmModel<T> model{};
  model.n_support = n_support;
  model.n_cols = n_cols;
  model.b = b;
  model.dual_coefs = dual;
  model.support_matrix.nnz = -1;
  model.support_matrix.data = support_columns;
  model.support_idx = nullptr;
  model.n_classes = 0;
  model.unique_labels = nullptr;
  return model;
}

inline constexpr std::string_view stalled_working_set =
    "Working set has already been initialized";

template <typename Fn>
auto stall_translated(Fn&& fn) {
  try {
    return std::forward<Fn>(fn)();
  } catch (const std::exception& e) {
    if (std::string_view(e.what()).find(stalled_working_set) !=
        std::string_view::npos) {
      throw logic_error("the SMO solver stopped making progress; raise #:tol");
    }
    throw;
  }
}

}  // namespace cumlrkt::svm
