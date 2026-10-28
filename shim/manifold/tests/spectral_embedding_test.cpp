#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <string>
#include <vector>

#include "cumlrkt/largest_positive.hpp"
#include "cumlrkt/manifold.h"
#include "manifold_fixture.hpp"
#include "raftrkt/c_api.h"
#include "raftrkt/view.hpp"

namespace cumlrkt::test {
namespace {

constexpr int64_t rows = 40;
constexpr int64_t cols = 3;
constexpr int64_t neighbours = 8;
constexpr int64_t path = 24;
constexpr int tiny_exponent = -70;
constexpr int unit_exponent = -3;
constexpr int tiny_weight_exponent = -40;

bool all_finite(const std::vector<float>& host) {
  return std::ranges::all_of(host, [](float v) { return std::isfinite(v); });
}

bool sign_ruled(const std::vector<float>& host, int64_t k) {
  const auto n = static_cast<int64_t>(host.size()) / k;
  for (int64_t c = 0; c < k; ++c) {
    float best = 0;
    for (int64_t r = 0; r < n; ++r) {
      const float v = host[static_cast<std::size_t>((r * k) + c)];
      if (std::fabs(v) > std::fabs(best)) {
        best = v;
      }
    }
    if (!(best > 0)) {
      return false;
    }
  }
  return true;
}

void expect_matrix(const record& out, int64_t n, const spectral_options& o) {
  EXPECT_EQ(out.r.count, 1);
  EXPECT_EQ(out.r.regions[0].rank, 2);
  EXPECT_EQ(out.r.regions[0].dtype, RR_DTYPE_FLOAT32);
  EXPECT_EQ(out.r.regions[0].shape[0], n);
  EXPECT_EQ(out.r.regions[0].shape[1], o.n_components);
  EXPECT_EQ(out.r.regions[0].strides[0], o.n_components);
}

std::vector<float> embedded(const resources& res, const rr_view* x, int64_t n,
                            const spectral_options& o = {}) {
  record out(res);
  EXPECT_EQ(outcome(spectral_fit(res, x, out, o)), "accepted");
  expect_matrix(out, n, o);
  return out.r.count == 1 ? out.read<float>(0) : std::vector<float>{};
}

TEST(SpectralEmbeddingFit, DenseIsSignRuledFiniteAndRepeatsBitwise) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, blobs(rows, cols), {rows, cols});
  const spectral_options o{.n_neighbors = neighbours};
  const std::vector<float> first = embedded(res, x.view(), rows, o);
  ASSERT_EQ(first.size(), static_cast<std::size_t>(rows * 2));
  EXPECT_TRUE(all_finite(first));
  EXPECT_TRUE(sign_ruled(first, 2));
  EXPECT_EQ(first, embedded(res, x.view(), rows, o));
  EXPECT_EQ(x.array.view.shape[0], rows);
}

TEST(SpectralEmbeddingFit, FitsAtTheLeastRowsForEachEigenvectorCount) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const std::vector<spectral_options> cases{
      {.n_components = 1, .drop_first = 1},
      {.n_components = 2, .drop_first = 1},
      {.n_components = 2, .drop_first = 0},
      {.n_components = 8, .drop_first = 1},
      {.n_components = 19, .drop_first = 1}};
  for (spectral_options o : cases) {
    const int64_t n = 2 * (o.n_components + o.drop_first + 1);
    o.n_neighbors = (n / 2) + 1;
    SCOPED_TRACE(n);
    const filled<float> x(res, blobs(n, cols), {n, cols});
    EXPECT_TRUE(all_finite(embedded(res, x.view(), n, o)));
  }
}

TEST(SpectralEmbeddingFit, TinyXEmbedsAsItsRescaledCopy) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  std::vector<float> host = blobs(rows, cols);
  std::ranges::transform(host, host.begin(),
                         [](float v) { return std::ldexp(v, unit_exponent); });
  std::vector<float> tiny(host.size());
  std::ranges::transform(host, tiny.begin(),
                         [](float v) { return std::ldexp(v, tiny_exponent); });
  const filled<float> x(res, host, {rows, cols});
  const filled<float> small(res, tiny, {rows, cols});
  const spectral_options o{.n_neighbors = neighbours};
  EXPECT_EQ(embedded(res, small.view(), rows, o),
            embedded(res, x.view(), rows, o));
  EXPECT_EQ(small.array.view.shape[1], cols);
}

struct path_graph {
  path_graph(const resources& res, int64_t n, float weight = 1.0F)
      : n(n),
        rows(res, ends(n, true), {2 * (n - 1)}),
        cols(res, ends(n, false), {2 * (n - 1)}),
        vals(res, std::vector<float>(2 * (n - 1), weight), {2 * (n - 1)}) {}
  static std::vector<int32_t> ends(int64_t n, bool first) {
    std::vector<int32_t> out;
    for (int32_t i = 0; i + 1 < n; ++i) {
      out.push_back(first ? i : i + 1);
      out.push_back(first ? i + 1 : i);
    }
    return out;
  }
  graph_input input() const {
    return {rows.view(), cols.view(), vals.view(), n};
  }
  int64_t n;
  filled<int32_t> rows;
  filled<int32_t> cols;
  filled<float> vals;
};

TEST(SpectralEmbeddingFitGraph, APathIsMonotoneAndSignRuled) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const path_graph g(res, path);
  record out(res);
  EXPECT_EQ(
      outcome(spectral_fit_graph(res, g.input(), out, {.n_components = 1})),
      "accepted");
  const std::vector<float> e = out.read<float>(0);
  ASSERT_EQ(e.size(), static_cast<std::size_t>(path));
  EXPECT_TRUE(sign_ruled(e, 1));
  EXPECT_TRUE(std::ranges::is_sorted(e) ||
              std::ranges::is_sorted(e, std::ranges::greater{}));
}

TEST(SpectralEmbeddingFitGraph, FitsAtTheLeastSamplesForEachEigenvectorCount) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const std::vector<spectral_options> cases{
      {.n_components = 1, .drop_first = 0},
      {.n_components = 1, .drop_first = 1},
      {.n_components = 2, .drop_first = 1}};
  for (const spectral_options& o : cases) {
    const int64_t n = 2 * (o.n_components + o.drop_first + 1);
    SCOPED_TRACE(n);
    const path_graph g(res, n);
    record out(res);
    EXPECT_EQ(outcome(spectral_fit_graph(res, g.input(), out, o)), "accepted");
    EXPECT_TRUE(all_finite(out.read<float>(0)));
  }
}

std::vector<float> graph_embedded(const resources& res, const path_graph& g,
                                  const spectral_options& o) {
  record out(res);
  const std::string got = outcome(spectral_fit_graph(res, g.input(), out, o));
  EXPECT_EQ(got, "accepted");
  return got == "accepted" ? out.read<float>(0) : std::vector<float>{};
}

TEST(SpectralEmbeddingFitGraph, TinyWeightsEmbedAsTheirRescaledCopy) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const path_graph unit(res, path);
  const path_graph tiny(res, path, std::ldexp(1.0F, tiny_weight_exponent));
  const spectral_options o{.norm_laplacian = 0};
  EXPECT_EQ(graph_embedded(res, tiny, o), graph_embedded(res, unit, o));
}

std::vector<float> sign_ruled_copy(const std::vector<float>& host, int64_t n,
                                   int64_t k) {
  const resources res;
  const filled<float> x(res, host, {n, k});
  const raft::handle_t& h = raftrkt::handle_of(res.handle);
  make_largest_positive<float>(h, static_cast<float*>(x.array.view.data), n, k);
  EXPECT_EQ(cudaStreamSynchronize(raft::resource::get_cuda_stream(h).value()),
            cudaSuccess);
  std::vector<float> back(host.size());
  EXPECT_EQ(cudaMemcpy(back.data(), x.array.view.data,
                       back.size() * sizeof(float), cudaMemcpyDeviceToHost),
            cudaSuccess);
  return back;
}

TEST(MakeLargestPositive, TheFirstLargestEntryOfEachColumnTurnsPositive) {
  CUMLRKT_REQUIRE_GPU();
  EXPECT_EQ(sign_ruled_copy({0, 3, -2, -3, 2, 1, 1, 0}, 4, 2),
            (std::vector<float>{0, 3, 2, -3, -2, 1, -1, 0}));
  EXPECT_EQ(sign_ruled_copy({-1, -1, 1}, 3, 1), (std::vector<float>{1, 1, -1}));
  EXPECT_EQ(sign_ruled_copy({0, 0}, 2, 1), (std::vector<float>{0, 0}));
}

TEST(SpectralEmbeddingFit, XOnAnotherDeviceIsRefused) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const filled<float> x(res, blobs(rows, cols), {rows, cols});
  rr_view elsewhere = *x.view();
  elsewhere.device = 1;
  record out(res);
  EXPECT_EQ(outcome(spectral_fit(res, &elsewhere, out)),
            "X: on device 1, but the resources are on device 0");
  expect_untouched(out);
}

}  // namespace
}  // namespace cumlrkt::test
