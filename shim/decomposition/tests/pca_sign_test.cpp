#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <vector>

#include "decomposition_fixture.hpp"
#include "pca_device.hpp"
#include "raftrkt/c_api.h"
#include "raftrkt/view.hpp"

namespace cumlrkt::test {
namespace {

using cumlrkt::decomposition::fit_arrays;
using cumlrkt::decomposition::post_fit;

constexpr int64_t width = 3;
constexpr auto cells = static_cast<std::size_t>(width);
constexpr int64_t samples = 5;
constexpr int64_t spectra = 3;

template <typename T>
using rows = std::vector<std::vector<T>>;

template <typename T>
std::vector<T> column_major(const rows<T>& components) {
  const std::size_t m = components.size();
  std::vector<T> out(m * cells);
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < cells; ++j) {
      out[i + (j * m)] = components[i][j];
    }
  }
  return out;
}

template <typename T>
rows<T> signed_rows(const rows<T>& components) {
  const resources res;
  const auto m = static_cast<int64_t>(components.size());
  const filled<T> in(res, column_major(components), {m, width});
  const filled<T> variance(res, std::vector<T>(components.size(), T{1}), {m});
  const device_array out(res.r, dtype_of<T>, {m, width});
  const device_array spectrum(res.r, dtype_of<T>, {spectra, m});
  const device_array totals(res.r, RR_DTYPE_FLOAT64, {2});
  const raft::handle_t& handle = raftrkt::handle_of(res.handle);
  T* const s = data<T>(spectrum);
  post_fit<T>(handle, data<T>(in.array), data<T>(variance.array), m, m, width,
              samples, fit_arrays<T>{data<T>(out), s, s + m, s + (2 * m)},
              data<double>(totals));
  EXPECT_EQ(
      cudaStreamSynchronize(raft::resource::get_cuda_stream(handle).value()),
      cudaSuccess);
  return row_slices(out.read<T>(), cells);
}

template <typename T>
void expect_first_largest_positive() {
  const rows<T> components{{T{0.1}, T{-0.9}, T{0.3}},
                           {T{0.2}, T{0.7}, T{-0.6}},
                           {T{-0.5}, T{0.5}, T{0.1}},
                           {T{0.5}, T{-0.5}, T{0.1}}};
  const rows<T> want{{T{-0.1}, T{0.9}, T{-0.3}},
                     {T{0.2}, T{0.7}, T{-0.6}},
                     {T{0.5}, T{-0.5}, T{-0.1}},
                     {T{0.5}, T{-0.5}, T{0.1}}};
  EXPECT_EQ(signed_rows(components), want);
}

TEST(PcaSigns, TheFirstEntryOfLargestMagnitudeIsMadePositive) {
  CUMLRKT_REQUIRE_GPU();
  expect_first_largest_positive<float>();
  expect_first_largest_positive<double>();
}

}  // namespace
}  // namespace cumlrkt::test
