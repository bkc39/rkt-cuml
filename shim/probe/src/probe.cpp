#include "cumlrkt/probe.h"

#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <string>
#include <utility>
#include <vector>

#include "cumlrkt/common.hpp"
#include "cumlrkt/model.hpp"
#include "probe_kinds.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/error.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"

namespace {

using cumlrkt::logic_error;
using cumlrkt::model_kind;

constexpr auto gate_limit = std::chrono::seconds(5);

struct tallies {
  std::atomic<int64_t> created{0};
  std::atomic<int64_t> released{0};
  std::atomic<int64_t> destroyed{0};
  std::atomic<int64_t> stale{0};
};

tallies& counts() noexcept {
  return cumlrkt::never_destroyed<tallies>();
}

rmm::cuda_stream_view stream_of(const raft::handle_t& handle) {
  return raft::resource::get_cuda_stream(handle);
}

void settle(rmm::cuda_stream_view stream) {
  raftrkt::cuda_check(cudaStreamSynchronize(stream.value()),
                      "cudaStreamSynchronize");
}

struct shape {
  model_kind kind;
  int64_t payload_bytes;
  int32_t region_length;
};

class probe_model final : public cumlrkt::model {
 public:
  probe_model(const raft::handle_t& handle, const shape& s)
      : model(handle, s.kind),
        handle_(handle),
        payload_(static_cast<std::size_t>(s.payload_bytes), stream_of(handle)),
        region_(static_cast<std::size_t>(s.region_length), stream_of(handle)) {
    counts().created.fetch_add(1);
  }
  ~probe_model() override {
    counts().destroyed.fetch_add(1);
    const cudaError_t status = cudaStreamQuery(stream_of(handle_).value());
    if (status != cudaSuccess && status != cudaErrorNotReady) {
      counts().stale.fetch_add(1);
      static_cast<void>(cudaGetLastError());
    }
  }
  probe_model(const probe_model&) = delete;
  probe_model& operator=(const probe_model&) = delete;
  probe_model(probe_model&&) = delete;
  probe_model& operator=(probe_model&&) = delete;

  void fill() {
    std::vector<int32_t> values(region_.size());
    std::iota(values.begin(), values.end(), 0);
    const rmm::cuda_stream_view stream = stream_of(handle_);
    if (!values.empty()) {
      raftrkt::cuda_check(
          cudaMemcpyAsync(region_.data(), values.data(),
                          values.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice, stream.value()),
          "cudaMemcpyAsync");
    }
    settle(stream);
  }

  rr_view region() {
    return raftrkt::region(region_);
  }

  int64_t touch(rmm::cuda_stream_view stream) {
    if (!payload_.is_empty()) {
      raftrkt::cuda_check(
          cudaMemsetAsync(payload_.data(), 1, payload_.size(), stream.value()),
          "cudaMemsetAsync");
    }
    return static_cast<int64_t>(payload_.size());
  }

 protected:
  void drop() noexcept override {
    try {
      const rmm::cuda_stream_view stream = stream_of(handle_);
      payload_ = rmm::device_uvector<std::byte>(0, stream);
      static_cast<void>(cudaStreamSynchronize(stream.value()));
      counts().released.fetch_add(1);
    } catch (...) {
      static_cast<void>(cudaGetLastError());
    }
  }

 private:
  const raft::handle_t& handle_;
  rmm::device_uvector<std::byte> payload_;
  rmm::device_uvector<int32_t> region_;
};

struct gate {
  std::mutex lock;
  std::condition_variable opened;
  bool open = false;
  int32_t waiters = 0;
};

struct gate_table {
  std::mutex lock;
  std::vector<std::shared_ptr<gate>> gates;
};

gate_table& gates() noexcept {
  return cumlrkt::never_destroyed<gate_table>();
}

std::shared_ptr<gate> gate_at(int32_t id) {
  gate_table& table = gates();
  const std::lock_guard<std::mutex> held{table.lock};
  if (id < 0 || static_cast<std::size_t>(id) >= table.gates.size()) {
    throw logic_error("gate: no gate " + std::to_string(id));
  }
  return table.gates.at(static_cast<std::size_t>(id));
}

void wait_at(int32_t id) {
  if (id < 0) {
    return;
  }
  const std::shared_ptr<gate> g = gate_at(id);
  std::unique_lock<std::mutex> held{g->lock};
  ++g->waiters;
  const bool opened =
      g->opened.wait_for(held, gate_limit, [&] { return g->open; });
  --g->waiters;
  if (!opened) {
    throw logic_error("gate: still shut after 5 s");
  }
}

template <typename Run>
void in_device_scope(void* handle, const rr_view* anchor, Run&& run) {
  const raft::handle_t& h = raftrkt::handle_of(handle);
  const int device = raftrkt::device_of(h);
  const raftrkt::device_scope scope{device};
  raftrkt::require_on_device(device, {{"model", anchor}});
  raftrkt::sync_guard sync{stream_of(h)};
  run(h);
  sync.finish();
}

template <typename Fn>
int translated(Fn&& fn) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(),
                                       std::forward<Fn>(fn));
}

void require_at_least(const char* what, int64_t got, int64_t least) {
  if (got < least) {
    throw logic_error(std::string(what) + ": expected at least " +
                      std::to_string(least) + ", got " + std::to_string(got));
  }
}

model_kind kind_of(int32_t code) {
  switch (code) {
    case CUML_PROBE_KIND_A:
      return cumlrkt::probe_a;
    case CUML_PROBE_KIND_B:
      return cumlrkt::probe_b;
    default:
      throw logic_error("kind: expected 1 or 2, got " + std::to_string(code));
  }
}

shape checked_shape(int64_t payload_bytes, int32_t region_length,
                    int32_t kind) {
  require_at_least("payload-bytes", payload_bytes, 0);
  require_at_least("region-length", region_length, 0);
  return {kind_of(kind), payload_bytes, region_length};
}

}  // namespace

extern "C" {

const char* cuml_probe_last_error(void) {
  return cumlrkt::last_error_slot().message.data();
}

int cuml_probe_last_error_kind(void) {
  return static_cast<int>(cumlrkt::last_error_slot().kind);
}

int cuml_probe_check_abi(const rr_abi_tag* loaded) {
  return cumlrkt::check_abi(loaded);
}

const char* cuml_probe_libcuml_version(void) {
  return cumlrkt::libcuml_release.data();
}

int cuml_probe_model_create(void* handle, rr_deposit* out,
                            int64_t payload_bytes, int32_t region_length,
                            int32_t kind) {
  return translated([&] {
    rr_deposit* const into = raftrkt::require(out, "out");
    const shape s = checked_shape(payload_bytes, region_length, kind);
    in_device_scope(handle, nullptr, [&](const raft::handle_t& h) {
      auto m = std::make_unique<probe_model>(h, s);
      m->fill();
      const std::array<rr_view, 1> regions{m->region()};
      cumlrkt::publish_model(into, "out", h, std::move(m), regions);
    });
  });
}

void cuml_probe_model_create_job(void* args, size_t args_len,
                                 rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_probe_model_create>(args, args_len, outcome,
                                             cumlrkt::last_error_slot());
}

int cuml_probe_model_touch(void* handle, const rr_view* anchor, int32_t gate,
                           int64_t* payload_bytes) {
  return translated([&] {
    int64_t* const answer = raftrkt::require(payload_bytes, "payload-bytes");
    in_device_scope(handle, anchor, [&](const raft::handle_t& h) {
      const auto use = cumlrkt::use_model<probe_model>(
          anchor, "model", cumlrkt::probe_a, stream_of(h));
      wait_at(gate);
      *answer = use->touch(stream_of(h));
    });
  });
}

void cuml_probe_model_touch_job(void* args, size_t args_len,
                                rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_probe_model_touch>(args, args_len, outcome,
                                            cumlrkt::last_error_slot());
}

int cuml_probe_model_release(void* handle, const rr_view* anchor) {
  return translated([&] {
    in_device_scope(handle, anchor, [&](const raft::handle_t& /*h*/) {
      static_cast<void>(cumlrkt::any_model(anchor, "model").release());
    });
  });
}

void cuml_probe_model_release_job(void* args, size_t args_len,
                                  rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_probe_model_release>(args, args_len, outcome,
                                              cumlrkt::last_error_slot());
}

int cuml_probe_model_counts(int64_t* created, int64_t* released,
                            int64_t* destroyed, int64_t* stale) {
  return translated([&] {
    *raftrkt::require(created, "created") = counts().created.load();
    *raftrkt::require(released, "released") = counts().released.load();
    *raftrkt::require(destroyed, "destroyed") = counts().destroyed.load();
    *raftrkt::require(stale, "stale") = counts().stale.load();
  });
}

int cuml_probe_gate_create(int32_t* gate_id) {
  return translated([&] {
    int32_t* const answer = raftrkt::require(gate_id, "gate");
    gate_table& table = gates();
    const std::lock_guard<std::mutex> held{table.lock};
    table.gates.push_back(std::make_shared<gate>());
    *answer = static_cast<int32_t>(table.gates.size() - 1);
  });
}

int cuml_probe_gate_open(int32_t gate_id) {
  return translated([&] {
    const std::shared_ptr<gate> g = gate_at(gate_id);
    {
      const std::lock_guard<std::mutex> held{g->lock};
      g->open = true;
    }
    g->opened.notify_all();
  });
}

int cuml_probe_gate_waiters(int32_t gate_id, int32_t* waiters) {
  return translated([&] {
    int32_t* const answer = raftrkt::require(waiters, "waiters");
    const std::shared_ptr<gate> g = gate_at(gate_id);
    const std::lock_guard<std::mutex> held{g->lock};
    *answer = g->waiters;
  });
}
}
