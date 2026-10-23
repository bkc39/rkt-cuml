#include "cumlrkt/model.hpp"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <future>
#include <memory>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cumlrkt/probe.h"
#include "probe_kinds.hpp"
#include "raftrkt/c_api.h"
#include "raftrkt/deposit.hpp"

// raft installs no header for its holder entries; these match its
// src/detail/deposit_api.h.
extern "C" {
int rr_holder_create(rr_resources* resources, rr_holder** out);
int rr_holder_record(rr_holder* holder, rr_deposit** out);
int rr_holder_close(rr_holder* holder);
int rr_holder_adopt(rr_holder* holder, int32_t index, rr_view* view,
                    rr_buffer** out);
}

namespace {

using cumlrkt::model_kind;
using cumlrkt::probe_a;
using cumlrkt::probe_b;
using namespace std::chrono_literals;

constexpr auto settle_limit = 5s;
constexpr int64_t payload_bytes = 1 << 20;
constexpr int32_t region_length = 4;

std::string gpu_unavailable_reason() {
  int32_t count = 0;
  if (rr_device_count(&count) != RR_OK) {
    return std::string("no GPU: ") + rr_last_error();
  }
  return count == 0 ? "no GPU: no CUDA device" : "";
}

#define CUMLRKT_REQUIRE_GPU()                                    \
  do {                                                           \
    const std::string cumlrkt_reason = gpu_unavailable_reason(); \
    if (!cumlrkt_reason.empty()) {                               \
      std::printf("SKIP: %s\n", cumlrkt_reason.c_str());         \
      GTEST_SKIP() << cumlrkt_reason;                            \
    }                                                            \
  } while (0)

struct session {
  rr_resources* resources = nullptr;
  rr_holder* holder = nullptr;
  rr_deposit* record = nullptr;
  const raft::handle_t* handle = nullptr;
  void* raw = nullptr;
};

session open_session() {
  session s;
  EXPECT_EQ(rr_resources_create(0, &s.resources), RR_OK) << rr_last_error();
  EXPECT_EQ(rr_resources_handle(s.resources, &s.raw), RR_OK) << rr_last_error();
  s.handle = static_cast<const raft::handle_t*>(s.raw);
  EXPECT_EQ(rr_holder_create(s.resources, &s.holder), RR_OK) << rr_last_error();
  EXPECT_EQ(rr_holder_record(s.holder, &s.record), RR_OK) << rr_last_error();
  return s;
}

rmm::cuda_stream_view stream_of(const session& s) {
  return raft::resource::get_cuda_stream(*s.handle);
}

void close_session(const session& s) {
  rr_holder_free(s.holder);
  rr_resources_free(s.resources);
}

void in_session(const std::function<void(const session&)>& body) {
  const session s = open_session();
  body(s);
  close_session(s);
}

struct counts {
  int64_t created = 0;
  int64_t released = 0;
  int64_t destroyed = 0;
  int64_t stale = 0;
};

counts counts_now() {
  counts c;
  EXPECT_EQ(
      cuml_probe_model_counts(&c.created, &c.released, &c.destroyed, &c.stale),
      RR_OK)
      << cuml_probe_last_error();
  return c;
}

bool settles_within(const std::function<bool()>& done) {
  const auto end = std::chrono::steady_clock::now() + settle_limit;
  while (std::chrono::steady_clock::now() < end) {
    if (rr_releases_pending() == 0 && done()) {
      return true;
    }
    std::this_thread::sleep_for(10ms);
  }
  return false;
}

class test_model final : public cumlrkt::model {
 public:
  test_model(const raft::handle_t& handle, model_kind kind,
             std::atomic<int>& destroyed)
      : model(handle, kind), destroyed_(&destroyed) {}
  ~test_model() override {
    destroyed_->fetch_add(1);
  }
  test_model(const test_model&) = delete;
  test_model& operator=(const test_model&) = delete;
  test_model(test_model&&) = delete;
  test_model& operator=(test_model&&) = delete;

  [[nodiscard]] int drops() const noexcept {
    return drops_.load();
  }

 protected:
  void drop() noexcept override {
    drops_.fetch_add(1);
  }

 private:
  std::atomic<int>* destroyed_;
  std::atomic<int> drops_{0};
};

void expect_refusal(const std::function<void()>& call,
                    const std::string& message) {
  try {
    call();
    ADD_FAILURE() << "no refusal; expected: " << message;
  } catch (const std::logic_error& e) {
    EXPECT_EQ(std::string(e.what()), message);
  }
}

TEST(Model, AFailedPublishDeletesTheModelAndUnregistersIt) {
  CUMLRKT_REQUIRE_GPU();
  const session s = open_session();
  ASSERT_EQ(rr_holder_close(s.holder), RR_OK) << rr_last_error();
  const std::size_t before = cumlrkt::registered_models();
  std::atomic<int> destroyed{0};
  {
    const test_model held(*s.handle, probe_a, destroyed);
    EXPECT_EQ(cumlrkt::registered_models(), before + 1);
  }
  EXPECT_EQ(destroyed.load(), 1);
  expect_refusal(
      [&] {
        cumlrkt::publish_model(
            s.record, "out", *s.handle,
            std::make_unique<test_model>(*s.handle, probe_a, destroyed));
      },
      "out: closed");
  EXPECT_EQ(destroyed.load(), 2);
  EXPECT_EQ(cumlrkt::registered_models(), before);
  close_session(s);
}

void refuse_non_anchors(const session& s) {
  rmm::device_uvector<int32_t> other(
      1, raft::resource::get_cuda_stream(*s.handle));
  const rr_view plain = raftrkt::region(other);
  std::atomic<int> destroyed{0};
  test_model m(*s.handle, probe_a, destroyed);
  rr_view retyped = m.anchor_region();
  retyped.dtype = RR_DTYPE_FLOAT32;
  const std::array<const rr_view*, 2> others{&plain, &retyped};
  for (const rr_view* view : others) {
    expect_refusal(
        [&] { static_cast<void>(cumlrkt::any_model(view, "model")); },
        "model: expected a model anchor, got another array");
  }
  expect_refusal(
      [&] { static_cast<void>(cumlrkt::any_model(nullptr, "model")); },
      "model is NULL");
}

TEST(Model, LookupRefusesAnArrayThatIsNotAnAnchor) {
  CUMLRKT_REQUIRE_GPU();
  in_session(refuse_non_anchors);
}

void refuse_kind_and_release(const session& s) {
  std::atomic<int> destroyed{0};
  test_model b(*s.handle, probe_b, destroyed);
  const rr_view anchor = b.anchor_region();
  const auto use_as = [&](model_kind kind) {
    static_cast<void>(
        cumlrkt::use_model<test_model>(&anchor, "model", kind, stream_of(s)));
  };
  expect_refusal([&] { use_as(probe_a); },
                 "model: expected a probe-a model, got a probe-b model");
  EXPECT_EQ(
      cumlrkt::use_model<test_model>(&anchor, "model", probe_b, stream_of(s))
          .operator->(),
      &b);
  EXPECT_TRUE(b.release());
  EXPECT_FALSE(b.release());
  EXPECT_EQ(b.drops(), 1);
  expect_refusal([&] { use_as(probe_b); },
                 "model: expected a live model, got a released one");
  EXPECT_EQ(&cumlrkt::any_model(&anchor, "model"), &b);
}

TEST(Model, LookupRefusesAnotherKindAndAReleasedModel) {
  CUMLRKT_REQUIRE_GPU();
  in_session(refuse_kind_and_release);
}

void free_step(const session& s, rr_buffer* anchor, int step) {
  switch (step) {
    case 0:
      rr_holder_free(s.holder);
      break;
    case 1:
      rr_buffer_free(anchor);
      break;
    default:
      rr_resources_free(s.resources);
      break;
  }
}

void expect_counted(const counts& before, const counts& expected,
                    const std::string& label) {
  const counts after = counts_now();
  EXPECT_EQ(after.created - before.created, expected.created) << label;
  EXPECT_EQ(after.released - before.released, expected.released) << label;
  EXPECT_EQ(after.destroyed - before.destroyed, expected.destroyed) << label;
  EXPECT_EQ(after.stale - before.stale, expected.stale) << label;
}

void adopt_anchor(const session& s, rr_buffer** anchor) {
  ASSERT_EQ(cuml_probe_model_create(s.raw, s.record, payload_bytes,
                                    region_length, CUML_PROBE_KIND_A),
            RR_OK)
      << cuml_probe_last_error();
  ASSERT_EQ(rr_holder_close(s.holder), RR_OK) << rr_last_error();
  rr_view view{};
  ASSERT_EQ(rr_holder_adopt(s.holder, 0, &view, anchor), RR_OK)
      << rr_last_error();
}

void expect_destroyed_once_in_order(const std::array<int, 3>& order) {
  const counts before = counts_now();
  const session s = open_session();
  rr_buffer* anchor = nullptr;
  adopt_anchor(s, &anchor);
  ASSERT_NE(anchor, nullptr);
  const std::string label = "order " + std::to_string(order[0]) +
                            std::to_string(order[1]) + std::to_string(order[2]);
  for (const int step : order) {
    EXPECT_EQ(counts_now().destroyed, before.destroyed) << label;
    free_step(s, anchor, step);
  }
  EXPECT_TRUE(settles_within([&] {
    return counts_now().destroyed == before.destroyed + 1;
  })) << label;
  expect_counted(before, {1, 0, 1, 0}, label);
}

TEST(Model, EveryOrderOfFreeingDestroysTheModelOnceWithItsHandleAlive) {
  CUMLRKT_REQUIRE_GPU();
  const uint64_t failures = rr_release_failure_count();
  std::array<int, 3> order{0, 1, 2};
  int orders = 0;
  do {
    expect_destroyed_once_in_order(order);
    ++orders;
  } while (std::next_permutation(order.begin(), order.end()));
  EXPECT_EQ(orders, 6);
  EXPECT_EQ(rr_release_failure_count(), failures);
}

void block_release_while_used(const session& s) {
  std::atomic<int> destroyed{0};
  test_model m(*s.handle, probe_a, destroyed);
  const rr_view anchor = m.anchor_region();
  std::future<bool> releasing;
  {
    const auto use =
        cumlrkt::use_model<test_model>(&anchor, "model", probe_a, stream_of(s));
    releasing = std::async(std::launch::async, [&] { return m.release(); });
    EXPECT_EQ(releasing.wait_for(200ms), std::future_status::timeout);
    EXPECT_EQ(use->drops(), 0);
  }
  ASSERT_EQ(releasing.wait_for(settle_limit), std::future_status::ready);
  EXPECT_TRUE(releasing.get());
  EXPECT_EQ(m.drops(), 1);
}

TEST(Model, AUseBlocksReleaseUntilItEnds) {
  CUMLRKT_REQUIRE_GPU();
  in_session(block_release_while_used);
}

std::atomic<bool> gate_open{false};

void CUDART_CB wait_for_gate(void* /*unused*/) {
  const auto end = std::chrono::steady_clock::now() + settle_limit;
  while (!gate_open.load() && std::chrono::steady_clock::now() < end) {
    std::this_thread::yield();
  }
}

class payload_model final : public cumlrkt::model {
 public:
  explicit payload_model(const raft::handle_t& handle)
      : model(handle, probe_a),
        handle_(handle),
        payload_(static_cast<std::size_t>(payload_bytes),
                 raft::resource::get_cuda_stream(handle)) {
    EXPECT_EQ(
        cudaStreamSynchronize(raft::resource::get_cuda_stream(handle).value()),
        cudaSuccess);
  }

  [[nodiscard]] const std::byte* payload() const noexcept {
    return payload_.data();
  }

 protected:
  void drop() noexcept override {
    try {
      const rmm::cuda_stream_view stream =
          raft::resource::get_cuda_stream(handle_);
      payload_ = rmm::device_uvector<std::byte>(0, stream);
      static_cast<void>(cudaStreamSynchronize(stream.value()));
    } catch (...) {
      static_cast<void>(cudaGetLastError());
    }
  }

 private:
  const raft::handle_t& handle_;
  rmm::device_uvector<std::byte> payload_;
};

void copy_under_a_use(const rr_view& anchor, rmm::cuda_stream_view stream,
                      std::byte* into, std::promise<void>& enqueued) {
  const auto use =
      cumlrkt::use_model<payload_model>(&anchor, "model", probe_a, stream);
  EXPECT_EQ(cudaLaunchHostFunc(stream.value(), wait_for_gate, nullptr),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpyAsync(into, use->payload(),
                            static_cast<std::size_t>(payload_bytes),
                            cudaMemcpyDeviceToDevice, stream.value()),
            cudaSuccess);
  enqueued.set_value();
}

void block_release_while_the_callers_stream_reads(const session& creating) {
  const session caller = open_session();
  const rmm::cuda_stream_view stream = stream_of(caller);
  {
    payload_model m(*creating.handle);
    const rr_view anchor = m.anchor_region();
    rmm::device_uvector<std::byte> copy(static_cast<std::size_t>(payload_bytes),
                                        stream);
    gate_open.store(false);
    std::promise<void> enqueued;
    std::future<void> using_it = std::async(std::launch::async, [&] {
      copy_under_a_use(anchor, stream, copy.data(), enqueued);
    });
    ASSERT_EQ(enqueued.get_future().wait_for(settle_limit),
              std::future_status::ready);
    std::future<bool> releasing =
        std::async(std::launch::async, [&] { return m.release(); });
    EXPECT_EQ(releasing.wait_for(200ms), std::future_status::timeout);
    gate_open.store(true);
    EXPECT_EQ(releasing.wait_for(settle_limit), std::future_status::ready);
    EXPECT_TRUE(releasing.get());
    using_it.get();
    EXPECT_EQ(cudaStreamSynchronize(stream.value()), cudaSuccess);
  }
  close_session(caller);
}

TEST(Model, AReleaseWaitsForAUsesWorkOnTheCallersStream) {
  CUMLRKT_REQUIRE_GPU();
  in_session(block_release_while_the_callers_stream_reads);
}

std::vector<int32_t> read_region(const rr_view& view) {
  std::vector<int32_t> host(static_cast<std::size_t>(view.shape[0]));
  EXPECT_EQ(cudaMemcpy(host.data(), view.data, host.size() * sizeof(int32_t),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  return host;
}

void expect_one_int32(const rr_view& anchor) {
  EXPECT_EQ(anchor.dtype, RR_DTYPE_INT32);
  EXPECT_EQ(anchor.rank, 1);
  EXPECT_EQ(anchor.shape[0], 1);
}

void expect_anchor_holds(int32_t kind) {
  const session s = open_session();
  ASSERT_EQ(cuml_probe_model_create(s.raw, s.record, 0, region_length, kind),
            RR_OK)
      << cuml_probe_last_error();
  ASSERT_EQ(s.record->count, 2);
  expect_one_int32(s.record->regions[0]);
  EXPECT_EQ(read_region(s.record->regions[0]), std::vector<int32_t>{kind});
  EXPECT_EQ(read_region(s.record->regions[1]),
            (std::vector<int32_t>{0, 1, 2, 3}));
  close_session(s);
}

TEST(Model, TheAnchorIsOneInt32HoldingTheKind) {
  CUMLRKT_REQUIRE_GPU();
  expect_anchor_holds(CUML_PROBE_KIND_A);
  expect_anchor_holds(CUML_PROBE_KIND_B);
}

}  // namespace
