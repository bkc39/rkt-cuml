#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "cluster_fixture.hpp"
#include "cumlrkt/cluster.h"
#include "kmeans_fixture.hpp"
#include "raftrkt/c_api.h"
#include "refusal_fixture.hpp"
#include "spectral_fixture.hpp"

namespace cumlrkt::test {
namespace {

constexpr double negative_weight = -5;
constexpr int64_t ring = 10;
constexpr std::size_t fault_entry = 7;
constexpr std::size_t loop_entry = 12;
constexpr std::size_t later_entry = 15;
constexpr int32_t loop_sample = 5;
constexpr int32_t far_index = 100000000;
constexpr double past_sum_weight = 0x1p125;

host_graph ring_graph() {
  host_graph g;
  for (int32_t i = 0; i < ring; ++i) {
    const auto j = static_cast<int32_t>((i + 1) % ring);
    g.rows.push_back(i);
    g.cols.push_back(j);
    g.vals.push_back(1);
    g.rows.push_back(j);
    g.cols.push_back(i);
    g.vals.push_back(1);
  }
  return g;
}

struct graph_fault_case {
  host_graph graph;
  std::string message;
};

graph_fault_case faulted(const std::function<void(host_graph&)>& edit,
                         const std::string& message) {
  host_graph g = ring_graph();
  edit(g);
  return {g, message};
}

std::vector<graph_fault_case> graph_faults() {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  return {
      faulted([](host_graph& g) { g.rows[fault_entry] = ring; },
              "rows: entry 7: expected an index in [0, 10), got 10"),
      faulted([](host_graph& g) { g.rows[fault_entry] = -1; },
              "rows: entry 7: expected an index in [0, 10), got -1"),
      faulted([](host_graph& g) { g.cols[fault_entry] = ring; },
              "cols: entry 7: expected an index in [0, 10), got 10"),
      faulted([](host_graph& g) { g.cols[fault_entry] = -1; },
              "cols: entry 7: expected an index in [0, 10), got -1"),
      faulted([](host_graph& g) { g.rows[fault_entry] = far_index; },
              "rows: entry 7: expected an index in [0, 10), got 100000000"),
      faulted(
          [](host_graph& g) {
            g.rows[loop_entry] = loop_sample;
            g.cols[loop_entry] = loop_sample;
          },
          "rows: entry 12: self-loop at sample 5; leave the diagonal out"),
      faulted([](host_graph& g) { g.vals[3] = negative_weight; },
              "vals: entry 3: expected a finite weight of at least 0, got -5"),
      faulted([nan](host_graph& g) { g.vals[3] = nan; },
              "vals: entry 3: expected a finite weight of at least 0, got nan"),
      faulted([inf](host_graph& g) { g.vals[3] = inf; },
              "vals: entry 3: expected a finite weight of at least 0, got inf"),
      faulted([inf](host_graph& g) { g.vals[3] = -inf; },
              "vals: entry 3: expected a finite weight of at least 0, got "
              "-inf"),
      faulted(
          [](host_graph& g) {
            g.vals[3] = std::numeric_limits<float>::denorm_min();
          },
          "vals: entry 3: expected 0 or a weight of at least "
          "1.1754943508222875e-38, got 1.401298464324817e-45"),
      faulted(
          [](host_graph& g) {
            for (double& v : g.vals) {
              v = past_sum_weight;
            }
          },
          "vals: the weights sum to 8.507059173023462e+38, more than "
          "3.4028234663852886e+38"),
      faulted(
          [](host_graph& g) {
            for (double& v : g.vals) {
              v = past_sum_weight;
            }
            g.rows[later_entry] = ring;
          },
          "rows: entry 15: expected an index in [0, 10), got 10"),
      faulted(
          [](host_graph& g) {
            for (double& v : g.vals) {
              v = 0;
            }
          },
          "vals: expected at least one positive weight, got none"),
      faulted(
          [](host_graph& g) {
            g.rows[fault_entry] = ring;
            g.cols[fault_entry] = -1;
            g.vals[fault_entry] = negative_weight;
          },
          "rows: entry 7: expected an index in [0, 10), got 10"),
      faulted(
          [](host_graph& g) {
            g.cols[fault_entry] = ring;
            g.vals[fault_entry] = negative_weight;
          },
          "cols: entry 7: expected an index in [0, 10), got 10"),
      faulted(
          [](host_graph& g) {
            g.rows[fault_entry] = ring;
            g.cols[fault_entry] = ring;
          },
          "rows: entry 7: expected an index in [0, 10), got 10"),
      faulted(
          [](host_graph& g) {
            g.rows[fault_entry] = 4;
            g.cols[fault_entry] = 4;
            g.vals[fault_entry] = negative_weight;
          },
          "rows: entry 7: self-loop at sample 4; leave the diagonal out"),
      faulted(
          [](host_graph& g) {
            g.vals[3] = negative_weight;
            g.rows[fault_entry] = ring;
            g.rows[loop_entry] = loop_sample;
            g.cols[loop_entry] = loop_sample;
          },
          "vals: entry 3: expected a finite weight of at least 0, got -5"),
      faulted(
          [](host_graph& g) {
            g.rows[loop_entry] = loop_sample;
            g.cols[loop_entry] = loop_sample;
            g.rows[later_entry] = ring;
          },
          "rows: entry 12: self-loop at sample 5; leave the diagonal out")};
}

void expect_scan_refusal(const resources& res, const graph_fault_case& f) {
  SCOPED_TRACE(f.message);
  const device_graph g(res, f.graph);
  record out(res);
  const allocation_meter meter;
  const graph_call call{.rows = &g.rows.view,
                        .cols = &g.cols.view,
                        .vals = &g.vals->view,
                        .n_samples = ring,
                        .out = &out.r};
  EXPECT_EQ(spectral_graph(res, call), f.message);
  EXPECT_EQ(cuml_cluster_last_error_kind(), RR_ERROR_LOGIC);
  EXPECT_LE(meter.peak(), scratch_bytes);
  EXPECT_EQ(meter.current(), 0);
  expect_untouched(out);
}

TEST(SpectralRefusals, GraphFaultsAreRefusedAtTheFirstEntry) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  for (const graph_fault_case& f : graph_faults()) {
    expect_scan_refusal(res, f);
  }
}

TEST(SpectralRefusals, LengthsPrecedeTheGraphScan) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  host_graph bad = ring_graph();
  bad.rows[fault_entry] = ring;
  const device_graph g(res, bad);
  const rr_view short_vals = shaped(*g.vals, {ring});
  record out(res);
  const allocation_meter meter;
  const graph_call call{.rows = &g.rows.view,
                        .cols = &g.cols.view,
                        .vals = &short_vals,
                        .n_samples = ring,
                        .out = &out.r};
  EXPECT_EQ(spectral_graph(res, call),
            "vals: expected 20 entries, as rows has, got 10");
  EXPECT_EQ(meter.peak(), 0);
  expect_untouched(out);
}

TEST(SpectralRefusals, AnEmptyGraphHasNoPositiveWeight) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const refusal_case<float> r(res, rows, cols);
  const rr_view ix = shaped(r.ints, {0});
  const rr_view w = shaped(r.data, {0});
  record out(res);
  const graph_call call{
      .rows = &ix, .cols = &ix, .vals = &w, .n_samples = ring, .out = &out.r};
  EXPECT_EQ(spectral_graph(res, call),
            "vals: expected at least one positive weight, got none");
  EXPECT_EQ(cuml_cluster_last_error_kind(), RR_ERROR_LOGIC);
  expect_untouched(out);
}

void expect_graph_refuses_filled(const resources& res, record& filled,
                                 const void* owner) {
  const device_graph g(res, ring_graph());
  const graph_call gc{.rows = &g.rows.view,
                      .cols = &g.cols.view,
                      .vals = &g.vals->view,
                      .n_samples = ring,
                      .out = &filled.r};
  EXPECT_EQ(spectral_graph(res, gc), "out: already filled");
  EXPECT_EQ(filled.r.owner, owner);
}

TEST(SpectralRefusals, FilledOutIsRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  values host;
  for (int64_t i = 0; i < rows * cols; ++i) {
    host.push_back(static_cast<double>((i * i) % spread));
  }
  const auto x = upload<float>(res, {rows, cols}, host);
  record filled(res);
  const spectral_call c{.x = &x->view, .out = &filled.r};
  ASSERT_EQ(spectral(res, c), "accepted");
  const void* owner = filled.r.owner;
  EXPECT_EQ(spectral(res, c), "out: already filled");
  EXPECT_EQ(filled.r.owner, owner);
  EXPECT_EQ(filled.r.count, 1);
  expect_graph_refuses_filled(res, filled, owner);
}

}  // namespace
}  // namespace cumlrkt::test
