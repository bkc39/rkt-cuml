#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <raft/stats/contingency_matrix.cuh>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include "contingency_device.hpp"
#include "raftrkt/error.hpp"

namespace cumlrkt::metrics {

namespace {

using count_t = unsigned long long;

}  // namespace

void count_pairs(int64_t* counts, const int32_t* a, const int32_t* b, int64_t n,
                 int64_t k1, int64_t k2, rmm::cuda_stream_view stream) {
  const int64_t side = std::max(k1, k2);
  const auto top = static_cast<int32_t>(side - 1);
  rmm::device_uvector<count_t> square(static_cast<std::size_t>(side * side),
                                      stream);
  const std::size_t bytes =
      raft::stats::getContingencyMatrixWorkspaceSize<int32_t, count_t>(
          static_cast<int>(n), a, stream.value(), 0, top);
  rmm::device_buffer workspace(bytes, stream);
  raft::stats::contingencyMatrix<int32_t, count_t>(
      a, b, static_cast<int>(n), square.data(), stream.value(),
      workspace.data(), bytes, 0, top);
  const std::size_t row_bytes = static_cast<std::size_t>(k2) * sizeof(int64_t);
  raftrkt::cuda_check(
      cudaMemcpy2DAsync(counts, row_bytes, square.data(),
                        static_cast<std::size_t>(side) * sizeof(count_t),
                        row_bytes, static_cast<std::size_t>(k1),
                        cudaMemcpyDeviceToDevice, stream.value()),
      "cudaMemcpy2DAsync");
  raftrkt::cuda_check(cudaStreamSynchronize(stream.value()),
                      "cudaStreamSynchronize");
}

}  // namespace cumlrkt::metrics
