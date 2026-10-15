#include "cumlrkt/cluster.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "cluster_fixture.hpp"
#include "cumlrkt/common.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {
namespace {

constexpr int64_t samples = 120;
constexpr int64_t features = 3;
constexpr int32_t clusters = 4;
constexpr double spread = 0.5;
constexpr double box = 10.0;
constexpr uint64_t seed = 7;

std::string blobs(const resources& res, const device_array& x,
                  const device_array& labels, int32_t n_clusters = clusters) {
  return outcome(cuml_cluster_make_blobs(res.handle, &x.view, &labels.view,
                                         n_clusters, spread, 1, -box, box,
                                         seed));
}

constexpr int august_2026 = 26;
constexpr int august = 8;
constexpr int december_2027 = 27;
constexpr int december = 12;
constexpr int third_patch = 3;
static_assert(std::string_view(rapids_release(august_2026, august, 0).data()) ==
              "26.08.00");
static_assert(std::string_view(rapids_release(december_2027, december,
                                              third_patch)
                                   .data()) == "27.12.03");

TEST(Abi, TheClusterShimAcceptsTheRaftItWasBuiltWith) {
  EXPECT_EQ(outcome(cuml_cluster_check_abi(rr_abi())), "accepted");
  rr_abi_tag forged = *rr_abi();
  forged.raft_minor += 2;
  EXPECT_EQ(outcome(cuml_cluster_check_abi(&forged)),
            "RAFT: built against 26.08.00, but libraftrkt has 26.10.00; "
            "build both against the same rapids package set");
  EXPECT_EQ(cuml_cluster_last_error_kind(), RR_ERROR_LOGIC);
}

TEST(Version, TheShimReportsTheRapidsReleaseOfItsRaft) {
  EXPECT_STREQ(cuml_cluster_libcuml_version(), rr_version());
}

TEST(Blobs, FillTheMatrixAndLabelEveryRowWithACluster) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const device_array x(res.r, RR_DTYPE_FLOAT32, {samples, features});
  const device_array labels(res.r, RR_DTYPE_INT32, {samples});
  ASSERT_EQ(blobs(res, x, labels), "accepted");
  const std::vector<int32_t> found = labels.read<int32_t>();
  EXPECT_EQ(*std::min_element(found.begin(), found.end()), 0);
  EXPECT_EQ(*std::max_element(found.begin(), found.end()), clusters - 1);
  const std::vector<float> data = x.read<float>();
  EXPECT_TRUE(std::all_of(data.begin(), data.end(),
                          [](float v) { return v > -2 * box && v < 2 * box; }));
}

TEST(Blobs, RefusalsNameTheArgument) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const device_array ints(res.r, RR_DTYPE_INT32, {samples, features});
  const device_array x(res.r, RR_DTYPE_FLOAT64, {samples, features});
  const device_array labels(res.r, RR_DTYPE_INT32, {samples});
  const device_array short_labels(res.r, RR_DTYPE_INT32, {samples - 1});
  EXPECT_EQ(blobs(res, ints, labels),
            "out: expected float32 or float64, got int32");
  EXPECT_EQ(blobs(res, x, labels, 0), "n-clusters: expected at least 1, got 0");
  EXPECT_EQ(blobs(res, x, short_labels).rfind("labels: ", 0), 0U);
  EXPECT_EQ(cuml_cluster_last_error_kind(), RR_ERROR_LOGIC);
}

}  // namespace
}  // namespace cumlrkt::test
