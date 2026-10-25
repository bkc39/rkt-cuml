#include "cumlrkt/preprocessing.h"

#include <gtest/gtest.h>

#include <string>

#include "preprocessing_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {
namespace {

TEST(Abi, ThePreprocessingShimAcceptsTheRaftItWasBuiltWith) {
  EXPECT_EQ(outcome(cuml_preprocessing_check_abi(rr_abi())), "accepted");
  rr_abi_tag forged = *rr_abi();
  forged.raft_minor += 2;
  EXPECT_EQ(outcome(cuml_preprocessing_check_abi(&forged)),
            "RAFT: built against 26.08.00, but libraftrkt has 26.10.00; "
            "build both against the same rapids package set");
  EXPECT_EQ(cuml_preprocessing_last_error_kind(), RR_ERROR_LOGIC);
}

TEST(Version, ThePreprocessingShimReportsTheRapidsReleaseOfItsRaft) {
  EXPECT_STREQ(cuml_preprocessing_libcuml_version(), rr_version());
  EXPECT_EQ(std::string(cuml_preprocessing_libcuml_version()), "26.08.00");
}

}  // namespace
}  // namespace cumlrkt::test
