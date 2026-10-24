#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cuml/cluster/hdbscan.hpp>
#include <memory>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <utility>

#include "cluster_kinds.hpp"
#include "cumlrkt/model.hpp"
#include "raftrkt/error.hpp"

namespace cumlrkt::cluster {

namespace hdbscan_common = ML::HDBSCAN::Common;

struct hdbscan_shape {
  int64_t n_rows;
  int64_t n_cols;
  int64_t min_samples;
};

class hdbscan_model final : public cumlrkt::model {
 public:
  hdbscan_model(const raft::handle_t& handle, const hdbscan_shape& shape,
                rmm::device_uvector<float>&& core_dists,
                rmm::device_uvector<int64_t>&& labels,
                hdbscan_common::hdbscan_output<int64_t, float>& out)
      : model(handle, hdbscan_kind),
        core_dists(std::move(core_dists)),
        tree(std::move(out.get_condensed_tree())),
        inverse_label_map(std::move(out._get_inverse_label_map())),
        stabilities(static_cast<std::size_t>(out.get_n_clusters()),
                    raft::resource::get_cuda_stream(handle)),
        labels(std::move(labels)),
        n_rows(shape.n_rows),
        n_cols(shape.n_cols),
        min_samples(shape.min_samples),
        k(out.get_n_clusters()) {
    const rmm::cuda_stream_view stream =
        raft::resource::get_cuda_stream(handle);
    if (k > 0) {
      raftrkt::cuda_check(
          cudaMemcpyAsync(stabilities.data(), out.get_stabilities(),
                          stabilities.size() * sizeof(float),
                          cudaMemcpyDeviceToDevice, stream.value()),
          "cudaMemcpyAsync");
    }
    pd = std::make_unique<hdbscan_common::PredictionData<int64_t, float>>(
        handle, n_rows, n_cols, this->core_dists.data());
    hdbscan_common::generate_prediction_data(handle, tree, this->labels.data(),
                                             inverse_label_map.data(),
                                             static_cast<int>(k), *pd);
  }

  [[nodiscard]] int64_t opaque_bytes() {
    constexpr int64_t index_bytes = sizeof(int64_t);
    constexpr int64_t value_bytes = sizeof(float);
    const int64_t indices =
        pd->get_n_exemplars() + (k + 1) + k + (tree.get_n_edges() + 1);
    return (value_bytes * n_rows) + (index_bytes * indices) +
           (value_bytes * tree.get_n_clusters());
  }

  rmm::device_uvector<float> core_dists;
  hdbscan_common::CondensedHierarchy<int64_t, float> tree;
  rmm::device_uvector<int64_t> inverse_label_map;
  rmm::device_uvector<float> stabilities;
  rmm::device_uvector<int64_t> labels;
  int64_t n_rows;
  int64_t n_cols;
  int64_t min_samples;
  int64_t k;
  std::unique_ptr<hdbscan_common::PredictionData<int64_t, float>> pd;

 protected:
  void drop() noexcept override {
    try {
      const rmm::cuda_stream_view stream = core_dists.stream();
      pd.reset();
      core_dists = rmm::device_uvector<float>(0, stream);
      static_cast<void>(cudaStreamSynchronize(stream.value()));
    } catch (...) {
      static_cast<void>(cudaGetLastError());
    }
  }
};

}  // namespace cumlrkt::cluster
