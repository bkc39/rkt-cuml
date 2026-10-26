#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "cumlrkt/manifold.h"
#include "manifold_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {
namespace {

constexpr int64_t wide_rows = 800;
constexpr int64_t bound_rows = 100;
constexpr int64_t cols = 10;
constexpr int64_t wide_components = 600;
constexpr int64_t most_bound_components = 48;
constexpr int64_t epochs = 30;
constexpr int64_t groups = 5;
constexpr uint32_t seed = 50;
constexpr float spread = 10.0F;

std::vector<float> normal_blobs(int64_t n) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> centre(-spread, spread);
  std::normal_distribution<float> noise;
  std::vector<float> centres(static_cast<std::size_t>(groups * cols));
  for (float& c : centres) {
    c = centre(gen);
  }
  std::vector<float> host(static_cast<std::size_t>(n * cols));
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      host[static_cast<std::size_t>((i * cols) + j)] =
          centres[static_cast<std::size_t>(((i % groups) * cols) + j)] +
          noise(gen);
    }
  }
  return host;
}

umap_options spectral(int64_t components) {
  return {.n_components = components,
          .epochs_given = 1,
          .n_epochs = epochs,
          .init = CUML_MANIFOLD_INIT_SPECTRAL};
}

struct run {
  std::string fit;
  std::vector<float> embedding;
  std::string transform;
  std::vector<float> output;
};

run fit_then_transform(const resources& res, const rr_view* x, const rr_view* y,
                       const umap_options& o) {
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

void expect_repeatable(const resources& res, const rr_view* x,
                       const rr_view* y) {
  const umap_options o = spectral(most_bound_components);
  const run first = fit_then_transform(res, x, y, o);
  const run again = fit_then_transform(res, x, y, o);
  EXPECT_EQ(first.fit, "accepted");
  EXPECT_EQ(first.transform, "accepted");
  EXPECT_EQ(again.embedding, first.embedding);
  EXPECT_EQ(again.output, first.output);
}

TEST(UmapSpectralBound, FitsAndTransformsAtTheBoundRepeatBitwise) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, normal_blobs(bound_rows), {bound_rows, cols});
  std::vector<int64_t> host(static_cast<std::size_t>(bound_rows));
  for (int64_t i = 0; i < bound_rows; ++i) {
    host[static_cast<std::size_t>(i)] = i % groups;
  }
  const filled<int64_t> y(res, host, {bound_rows});
  expect_repeatable(res, x.view(), nullptr);
  expect_repeatable(res, x.view(), y.view());
}

TEST(UmapSpectralBound, ABasisNoWiderThanTheComponentsIsRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> bound_x(res, normal_blobs(bound_rows),
                              {bound_rows, cols});
  const filled<float> wide_x(res, normal_blobs(wide_rows), {wide_rows, cols});
  for (const auto& [x, components, message] :
       {std::tuple{bound_x.view(), most_bound_components + 1,
                   "n-components: spectral init of 49 components needs more "
                   "than 100 rows, got 100"},
        std::tuple{wide_x.view(), wide_components,
                   "n-components: spectral init of 600 components needs more "
                   "than 1202 rows, got 800"}}) {
    record out(res);
    EXPECT_EQ(outcome(umap_fit(res, x, nullptr, out, spectral(components))),
              message);
    expect_untouched(out);
  }
}

}  // namespace
}  // namespace cumlrkt::test
