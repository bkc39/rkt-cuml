#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <vector>

#include "cumlrkt/manifold.h"
#include "cumlrkt/signs.hpp"
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

bool all_finite(const std::vector<float>& host) {
  return std::ranges::all_of(host, [](float v) { return std::isfinite(v); });
}

bool sign_ruled(const std::vector<float>& host, int64_t n, int64_t k) {
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

std::vector<float> embedded(const resources& res, const rr_view* x, int64_t n,
                            const spectral_options& o = {}) {
  record out(res);
  EXPECT_EQ(outcome(spectral_fit(res, x, out, o)), "accepted");
  EXPECT_EQ(out.r.count, 1);
  EXPECT_EQ(out.r.regions[0].rank, 2);
  EXPECT_EQ(out.r.regions[0].dtype, RR_DTYPE_FLOAT32);
  EXPECT_EQ(out.r.regions[0].shape[0], n);
  EXPECT_EQ(out.r.regions[0].shape[1], o.n_components);
  EXPECT_EQ(out.r.regions[0].strides[0], o.n_components);
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
  EXPECT_TRUE(sign_ruled(first, rows, 2));
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
  const std::vector<float> host = blobs(rows, cols);
  std::vector<float> tiny(host.size());
  std::ranges::transform(host, tiny.begin(), [](float v) {
    return std::ldexp(v, tiny_exponent);
  });
  const filled<float> x(res, host, {rows, cols});
  const filled<float> small(res, tiny, {rows, cols});
  const spectral_options o{.n_neighbors = neighbours};
  EXPECT_EQ(embedded(res, small.view(), rows, o),
            embedded(res, x.view(), rows, o));
  EXPECT_EQ(small.array.view.shape[1], cols);
}

struct path_graph {
  explicit path_graph(const resources& res)
      : rows(res, ends(true), {2 * (path - 1)}),
        cols(res, ends(false), {2 * (path - 1)}),
        vals(res, std::vector<float>(2 * (path - 1), 1.0F), {2 * (path - 1)}) {}
  static std::vector<int32_t> ends(bool first) {
    std::vector<int32_t> out;
    for (int32_t i = 0; i + 1 < path; ++i) {
      out.push_back(first ? i : i + 1);
      out.push_back(first ? i + 1 : i);
    }
    return out;
  }
  filled<int32_t> rows;
  filled<int32_t> cols;
  filled<float> vals;
};

TEST(SpectralEmbeddingFitGraph, APathIsMonotoneAndSignRuled) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const path_graph g(res);
  record out(res);
  EXPECT_EQ(outcome(spectral_fit_graph(
                res, {g.rows.view(), g.cols.view(), g.vals.view(), path}, out,
                {.n_components = 1})),
            "accepted");
  const std::vector<float> e = out.read<float>(0);
  ASSERT_EQ(e.size(), static_cast<std::size_t>(path));
  EXPECT_TRUE(sign_ruled(e, path, 1));
  EXPECT_TRUE(std::ranges::is_sorted(e) ||
              std::ranges::is_sorted(e, std::ranges::greater{}));
}

std::vector<float> sign_ruled_copy(const std::vector<float>& host, int64_t n,
                                   int64_t k) {
  const resources res;
  const filled<float> x(res, host, {n, k});
  const raft::handle_t& h = raftrkt::handle_of(res.handle);
  columns_sign_rule<float>(h, static_cast<float*>(x.array.view.data), n, k);
  EXPECT_EQ(cudaStreamSynchronize(raft::resource::get_cuda_stream(h).value()),
            cudaSuccess);
  std::vector<float> back(host.size());
  EXPECT_EQ(cudaMemcpy(back.data(), x.array.view.data,
                       back.size() * sizeof(float), cudaMemcpyDeviceToHost),
            cudaSuccess);
  return back;
}

TEST(ColumnsSignRule, TheFirstLargestEntryOfEachColumnTurnsPositive) {
  CUMLRKT_REQUIRE_GPU();
  EXPECT_EQ(sign_ruled_copy({0, 3, -2, -3, 2, 1, 1, 0}, 4, 2),
            (std::vector<float>{0, 3, 2, -3, -2, 1, -1, 0}));
  EXPECT_EQ(sign_ruled_copy({-1, -1, 1}, 3, 1),
            (std::vector<float>{1, 1, -1}));
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
