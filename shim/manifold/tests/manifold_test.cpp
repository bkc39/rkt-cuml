#include "cumlrkt/manifold.h"

#include <gtest/gtest.h>

#include <string>

#include "manifold_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {
namespace {

TEST(Abi, TheManifoldShimAcceptsTheRaftItWasBuiltWith) {
  EXPECT_EQ(outcome(cuml_manifold_check_abi(rr_abi())), "accepted");
  rr_abi_tag forged = *rr_abi();
  forged.raft_minor += 2;
  EXPECT_EQ(outcome(cuml_manifold_check_abi(&forged)),
            "RAFT: built against 26.08.00, but libraftrkt has 26.10.00; "
            "build both against the same rapids package set");
  EXPECT_EQ(cuml_manifold_last_error_kind(), RR_ERROR_LOGIC);
}

TEST(Version, TheManifoldShimReportsTheRapidsReleaseOfItsRaft) {
  EXPECT_STREQ(cuml_manifold_libcuml_version(), rr_version());
  EXPECT_EQ(std::string(cuml_manifold_libcuml_version()), "26.08.00");
}

}  // namespace
}  // namespace cumlrkt::test
