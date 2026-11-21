#include <gtest/gtest.h>

#include <cstdint>
#include <raft/core/handle.hpp>
#include <vector>

#include "column_sums.cuh"
#include "glm_staging_device.hpp"
#include "linear_model_fixture.hpp"
#include "raftrkt/view.hpp"

namespace {

using cumlrkt::stream_of;
using cumlrkt::glm::column_part_rows;
using cumlrkt::glm::column_sums;
using cumlrkt::glm::to_host;
using cumlrkt::test::resources;

constexpr int64_t row_step = 7;
constexpr int64_t column_step = 3;
constexpr int64_t value_modulus = 11;
constexpr double value_centre = 5.0;
constexpr double column_offset = 0.125;
constexpr int64_t many_parts = 16;

struct eighths {
  __host__ __device__ double operator()(int64_t k, int64_t i) const {
    return static_cast<double>(((i * row_step) + (k * column_step)) %
                               value_modulus) -
           value_centre + (column_offset * static_cast<double>(k));
  }
};

std::vector<double> summed_on_host(int64_t n, int64_t d) {
  std::vector<double> sums(d, 0.0);
  for (int64_t k = 0; k < d; ++k) {
    for (int64_t i = 0; i < n; ++i) {
      sums[k] += eighths{}(k, i);
    }
  }
  return sums;
}

TEST(ColumnSums, OnePartAndManyPartsAddTheSameColumns) {
  CUMLRKT_REQUIRE_GPU();
  const resources res;
  const raft::handle_t& handle = raftrkt::handle_of(res.handle);
  for (const int64_t n :
       {int64_t{1}, column_part_rows - 1, column_part_rows,
        column_part_rows + 1, (many_parts * column_part_rows) + 3}) {
    for (const int64_t d : {int64_t{1}, int64_t{3}}) {
      const std::vector<double> got =
          to_host(column_sums(handle, eighths{}, n, d), stream_of(handle));
      EXPECT_EQ(got, summed_on_host(n, d)) << n << " " << d;
    }
  }
}

}  // namespace
