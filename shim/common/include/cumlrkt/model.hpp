#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include "raftrkt/array.h"
#include "raftrkt/deposit.h"
#include "raftrkt/deposit.hpp"
#include "raftrkt/error.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt {

enum class model_kind : std::uint8_t { probe_a = 1, probe_b = 2 };

inline const char* kind_name(model_kind kind) noexcept {
  switch (kind) {
    case model_kind::probe_a:
      return "probe-a";
    case model_kind::probe_b:
      return "probe-b";
  }
  return "unknown";
}

class model;

namespace detail {

struct model_registry {
  std::mutex lock;
  std::unordered_map<const void*, model*> models;
};

// Never destroyed: models are deleted on raft's reaper thread, which can run
// after static destructors at exit.
inline model_registry& registry() noexcept {
  alignas(model_registry) static std::byte storage[sizeof(model_registry)];
  static model_registry* const shared =
      ::new (static_cast<void*>(storage)) model_registry;
  return *shared;
}

inline void enroll(const void* anchor, model* m) {
  model_registry& r = registry();
  const std::lock_guard<std::mutex> held{r.lock};
  r.models.emplace(anchor, m);
}

inline void withdraw(const void* anchor) noexcept {
  model_registry& r = registry();
  const std::lock_guard<std::mutex> held{r.lock};
  r.models.erase(anchor);
}

inline model* registered(const rr_view& view) {
  if (view.dtype != RR_DTYPE_INT32 || view.rank != 1 || view.shape[0] != 1) {
    return nullptr;
  }
  model_registry& r = registry();
  const std::lock_guard<std::mutex> held{r.lock};
  const auto found = r.models.find(view.data);
  return found == r.models.end() ? nullptr : found->second;
}

}  // namespace detail

class model {
 public:
  model(const raft::handle_t& handle, model_kind kind)
      : kind_(kind), anchor_(1, raft::resource::get_cuda_stream(handle)) {
    anchor_.set_element(0, static_cast<int32_t>(kind),
                        raft::resource::get_cuda_stream(handle));
    detail::enroll(anchor_.data(), this);
  }
  virtual ~model() {
    detail::withdraw(anchor_.data());
  }
  model(const model&) = delete;
  model& operator=(const model&) = delete;
  model(model&&) = delete;
  model& operator=(model&&) = delete;

  [[nodiscard]] model_kind kind() const noexcept {
    return kind_;
  }

  rr_view anchor_region() {
    return raftrkt::region(anchor_);
  }

  bool release() {
    const std::unique_lock<std::shared_mutex> held{guard_};
    if (released_) {
      return false;
    }
    drop();
    released_ = true;
    return true;
  }

 protected:
  // Frees opaque members only, never a published region, on the creating
  // handle's stream.
  virtual void drop() noexcept = 0;

 private:
  const model_kind kind_;
  rmm::device_uvector<int32_t> anchor_;
  std::shared_mutex guard_;
  bool released_ = false;
  template <typename M>
  friend class model_use;
};

template <typename M>
class model_use {
 public:
  model_use(M& m, const char* name)
      : lock_(static_cast<model&>(m).guard_), model_(&m) {
    if (static_cast<model&>(m).released_) {
      throw raftrkt::logic_error(std::string(name) +
                                 ": expected a live model, got a released one");
    }
  }

  M& operator*() const noexcept {
    return *model_;
  }
  M* operator->() const noexcept {
    return model_;
  }

 private:
  std::shared_lock<std::shared_mutex> lock_;
  M* model_;
};

inline std::size_t registered_models() {
  detail::model_registry& r = detail::registry();
  const std::lock_guard<std::mutex> held{r.lock};
  return r.models.size();
}

// The caller's view pins the anchor's buffer, and with it the model, for the
// whole call, so the model found here outlives the call.
inline model& any_model(const rr_view* anchor, const char* name) {
  model* found = detail::registered(*raftrkt::require(anchor, name));
  if (found == nullptr) {
    throw raftrkt::logic_error(std::string(name) +
                               ": expected a model anchor, got another array");
  }
  return *found;
}

template <typename M>
model_use<M> use_model(const rr_view* anchor, const char* name,
                       model_kind expected) {
  model& found = any_model(anchor, name);
  if (found.kind() != expected) {
    throw raftrkt::logic_error(std::string(name) + ": expected a " +
                               kind_name(expected) + " model, got a " +
                               kind_name(found.kind()) + " model");
  }
  return model_use<M>(static_cast<M&>(found), name);
}

template <typename M, typename... Regions>
void publish_model(rr_deposit* out, const char* name,
                   const raft::handle_t& handle, std::unique_ptr<M>&& m,
                   const Regions&... regions) {
  const rr_view anchor = raftrkt::require(m.get(), "model")->anchor_region();
  raftrkt::deposit(out, name, handle, std::move(m), {anchor, regions...});
}

}  // namespace cumlrkt
