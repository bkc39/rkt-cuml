#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "cumlrkt/manifold.h"
#include "manifold_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {
namespace {

constexpr int64_t wide_rows = 800;
constexpr int64_t wide_cols = 10;
constexpr int64_t wide_components = 600;
constexpr int64_t wide_epochs = 30;
constexpr int64_t wide_labels = 3;

struct run {
  std::string fit;
  std::vector<float> embedding;
  std::string transform;
  std::vector<float> output;
};

run fit_then_transform(const resources& res, const rr_view* x,
                       const rr_view* y, const umap_options& o) {
  run r;
  record trained(res);
  r.fit = outcome(umap_fit(res, x, y, trained, o));
  if (r.fit != "accepted") {
    return r;
  }
  r.embedding = trained.read<float>(0);
  record out(res);
  const umap_trained t{x, &trained.r.regions[0]};
  r.transform = outcome(umap_transform(res, x, t, out, o));
  if (r.transform == "accepted") {
    r.output = out.read<float>(0);
  }
  return r;
}

constexpr int wide_rounds = 3;

TEST(UmapWide, SpectralAndRandomFitsAndTransformsRepeatBitwise) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, blobs(wide_rows, wide_cols),
                        {wide_rows, wide_cols});
  std::vector<int64_t> host(static_cast<std::size_t>(wide_rows));
  for (int64_t i = 0; i < wide_rows; ++i) {
    host[static_cast<std::size_t>(i)] = i % wide_labels;
  }
  const filled<int64_t> y(res, host, {wide_rows});
  for (const int32_t init :
       {CUML_MANIFOLD_INIT_SPECTRAL, CUML_MANIFOLD_INIT_RANDOM}) {
    for (const rr_view* labels : {static_cast<const rr_view*>(nullptr),
                                  static_cast<const rr_view*>(y.view())}) {
      const umap_options o{.n_components = wide_components,
                           .epochs_given = 1,
                           .n_epochs = wide_epochs,
                           .init = init};
      const run first = fit_then_transform(res, x.view(), labels, o);
      EXPECT_EQ(first.fit, "accepted") << init;
      EXPECT_EQ(first.transform, "accepted") << init;
      for (int i = 1; i < wide_rounds; ++i) {
        const run again = fit_then_transform(res, x.view(), labels, o);
        EXPECT_EQ(again.fit, first.fit) << init;
        EXPECT_EQ(again.transform, first.transform) << init;
        EXPECT_EQ(again.embedding, first.embedding) << init;
        EXPECT_EQ(again.output, first.output) << init << (labels != nullptr);
      }
    }
  }
}

}  // namespace
}  // namespace cumlrkt::test
