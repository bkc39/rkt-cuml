#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <shared_mutex>
#include <stdexcept>

#include "raftrkt/array.h"
#include "raftrkt/deposit.h"

namespace cumlrkt {

enum class model_kind : int32_t { probe_a = 1, probe_b = 2 };

inline const char* kind_name(model_kind /*kind*/) noexcept {
  return "unimplemented";
}

class model {
 public:
  model(const raft::handle_t& /*handle*/, model_kind kind) : kind_(kind) {
    throw std::logic_error("unimplemented");
  }
  virtual ~model() = default;
  model(const model&) = delete;
  model& operator=(const model&) = delete;
  model(model&&) = delete;
  model& operator=(model&&) = delete;

  [[nodiscard]] model_kind kind() const noexcept {
    return kind_;
  }
  rr_view anchor_region() {
    throw std::logic_error("unimplemented");
  }
  bool release() {
    throw std::logic_error("unimplemented");
  }

 protected:
  virtual void drop() noexcept = 0;

 private:
  const model_kind kind_;
  rmm::device_uvector<int32_t> anchor_{0, rmm::cuda_stream_view{}};
  std::shared_mutex guard_;
  bool released_ = false;
  template <typename M>
  friend class model_use;
};

template <typename M>
class model_use {
 public:
  M& operator*() const noexcept {
    return *model_;
  }
  M* operator->() const noexcept {
    return model_;
  }

 private:
  std::shared_lock<std::shared_mutex> lock_;
  M* model_ = nullptr;
};

inline std::size_t registered_models() {
  throw std::logic_error("unimplemented");
}

inline model& any_model(const rr_view* /*anchor*/, const char* /*name*/) {
  throw std::logic_error("unimplemented");
}

template <typename M>
model_use<M> use_model(const rr_view* /*anchor*/, const char* /*name*/,
                       model_kind /*expected*/) {
  throw std::logic_error("unimplemented");
}

template <typename M, typename... Regions>
void publish_model(rr_deposit* /*out*/, const char* /*name*/,
                   const raft::handle_t& /*handle*/,
                   std::unique_ptr<M>&& /*m*/, const Regions&... /*regions*/) {
  throw std::logic_error("unimplemented");
}

}  // namespace cumlrkt
