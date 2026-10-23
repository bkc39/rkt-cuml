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
#include <rmm/device_uvector.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cumlrkt/model.hpp"
#include "cumlrkt/probe.h"
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

#define CUMLRKT_REQUIRE_GPU()                                     \
  do {                                                            \
    const std::string cumlrkt_reason = gpu_unavailable_reason();  \
    if (!cumlrkt_reason.empty()) {                                \
      std::printf("SKIP: %s\n", cumlrkt_reason.c_str());          \
      GTEST_SKIP() << cumlrkt_reason;                             \
    }                                                             \
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
  EXPECT_EQ(rr_holder_create(s.resources, &s.holder), RR_OK)
      << rr_last_error();
  EXPECT_EQ(rr_holder_record(s.holder, &s.record), RR_OK) << rr_last_error();
  return s;
}

void close_session(const session& s) {
  rr_holder_free(s.holder);
  rr_resources_free(s.resources);
}

struct counts {
  int64_t created = 0;
  int64_t released = 0;
  int64_t destroyed = 0;
  int64_t stale = 0;
};

counts counts_now() {
  counts c;
  EXPECT_EQ(cuml_probe_model_counts(&c.created, &c.released, &c.destroyed,
                                    &c.stale),
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
    const test_model held(*s.handle, model_kind::probe_a, destroyed);
    EXPECT_EQ(cumlrkt::registered_models(), before + 1);
  }
  EXPECT_EQ(destroyed.load(), 1);
  expect_refusal(
      [&] {
        cumlrkt::publish_model(s.record, "out", *s.handle,
                               std::make_unique<test_model>(
                                   *s.handle, model_kind::probe_a, destroyed));
      },
      "out: closed");
  EXPECT_EQ(destroyed.load(), 2);
  EXPECT_EQ(cumlrkt::registered_models(), before);
  close_session(s);
}

TEST(Model, LookupRefusesAnythingButALiveModelOfTheKind) {
  CUMLRKT_REQUIRE_GPU();
  const session s = open_session();
  const auto stream = raft::resource::get_cuda_stream(*s.handle);
  rmm::device_uvector<int32_t> other(1, stream);
  const rr_view plain = raftrkt::region(other);
  std::atomic<int> destroyed{0};
  test_model b(*s.handle, model_kind::probe_b, destroyed);
  const rr_view anchor = b.anchor_region();
  rr_view retyped = anchor;
  retyped.dtype = RR_DTYPE_FLOAT32;
  const std::array<const rr_view*, 2> others{&plain, &retyped};
  for (const rr_view* view : others) {
    expect_refusal([&] { static_cast<void>(cumlrkt::any_model(view, "model")); },
                   "model: expected a model anchor, got another array");
  }
  expect_refusal([&] { static_cast<void>(cumlrkt::any_model(nullptr, "model")); },
                 "model is NULL");
  expect_refusal(
      [&] {
        static_cast<void>(
            cumlrkt::use_model<test_model>(&anchor, "model", model_kind::probe_a));
      },
      "model: expected a probe-a model, got a probe-b model");
  EXPECT_EQ(cumlrkt::use_model<test_model>(&anchor, "model", model_kind::probe_b)
                .operator->(),
            &b);
  EXPECT_TRUE(b.release());
  EXPECT_FALSE(b.release());
  EXPECT_EQ(b.drops(), 1);
  expect_refusal(
      [&] {
        static_cast<void>(
            cumlrkt::use_model<test_model>(&anchor, "model", model_kind::probe_b));
      },
      "model: expected a live model, got a released one");
  EXPECT_EQ(&cumlrkt::any_model(&anchor, "model"), &b);
  EXPECT_STREQ(cumlrkt::kind_name(model_kind::probe_a), "probe-a");
  EXPECT_STREQ(cumlrkt::kind_name(model_kind::probe_b), "probe-b");
  close_session(s);
}

void expect_destroyed_once_in_order(const std::array<int, 3>& order) {
  const counts before = counts_now();
  const session s = open_session();
  ASSERT_EQ(cuml_probe_model_create(s.raw, s.record, payload_bytes,
                                    region_length, CUML_PROBE_KIND_A),
            RR_OK)
      << cuml_probe_last_error();
  ASSERT_EQ(rr_holder_close(s.holder), RR_OK) << rr_last_error();
  rr_view view{};
  rr_buffer* anchor = nullptr;
  ASSERT_EQ(rr_holder_adopt(s.holder, 0, &view, &anchor), RR_OK)
      << rr_last_error();
  const std::string label = "order " + std::to_string(order[0]) +
                            std::to_string(order[1]) + std::to_string(order[2]);
  for (const int step : order) {
    EXPECT_EQ(counts_now().destroyed, before.destroyed) << label;
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
  EXPECT_TRUE(settles_within(
      [&] { return counts_now().destroyed == before.destroyed + 1; }))
      << label;
  const counts after = counts_now();
  EXPECT_EQ(after.created - before.created, 1) << label;
  EXPECT_EQ(after.released - before.released, 0) << label;
  EXPECT_EQ(after.destroyed - before.destroyed, 1) << label;
  EXPECT_EQ(after.stale - before.stale, 0) << label;
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

TEST(Model, AUseBlocksReleaseUntilItEnds) {
  CUMLRKT_REQUIRE_GPU();
  const session s = open_session();
  std::atomic<int> destroyed{0};
  test_model m(*s.handle, model_kind::probe_a, destroyed);
  const rr_view anchor = m.anchor_region();
  std::future<bool> releasing;
  {
    const auto use =
        cumlrkt::use_model<test_model>(&anchor, "model", model_kind::probe_a);
    releasing = std::async(std::launch::async, [&] { return m.release(); });
    EXPECT_EQ(releasing.wait_for(200ms), std::future_status::timeout);
    EXPECT_EQ(use->drops(), 0);
  }
  ASSERT_EQ(releasing.wait_for(settle_limit), std::future_status::ready);
  EXPECT_TRUE(releasing.get());
  EXPECT_EQ(m.drops(), 1);
  close_session(s);
}

std::vector<int32_t> read_region(const rr_view& view) {
  std::vector<int32_t> host(static_cast<std::size_t>(view.shape[0]));
  EXPECT_EQ(cudaMemcpy(host.data(), view.data, host.size() * sizeof(int32_t),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  return host;
}

TEST(Model, TheAnchorIsOneInt32HoldingTheKind) {
  CUMLRKT_REQUIRE_GPU();
  for (const int32_t kind : {CUML_PROBE_KIND_A, CUML_PROBE_KIND_B}) {
    const session s = open_session();
    ASSERT_EQ(cuml_probe_model_create(s.raw, s.record, 0, region_length, kind),
              RR_OK)
        << cuml_probe_last_error();
    ASSERT_EQ(s.record->count, 2);
    const rr_view& anchor = s.record->regions[0];
    EXPECT_EQ(anchor.dtype, RR_DTYPE_INT32);
    EXPECT_EQ(anchor.rank, 1);
    EXPECT_EQ(anchor.shape[0], 1);
    EXPECT_EQ(read_region(anchor), std::vector<int32_t>{kind});
    EXPECT_EQ(read_region(s.record->regions[1]),
              (std::vector<int32_t>{0, 1, 2, 3}));
    close_session(s);
  }
}

}  // namespace
