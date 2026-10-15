#include <cstdint>
#include <cuml/datasets/make_blobs.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <string>

#include "cumlrkt/cluster.h"
#include "cumlrkt/common.hpp"
#include "raftrkt/job.hpp"

namespace {

using cumlrkt::logic_error;
using raftrkt::layout;

struct blob_arrays {
  const rr_view* out;
  const rr_view* labels;
};

struct blob_options {
  int32_t n_clusters;
  double cluster_std;
  bool shuffle;
  double center_box_min;
  double center_box_max;
  uint64_t seed;
};

template <typename T>
void make_blobs(const raft::handle_t& handle, const blob_arrays& arrays,
                const blob_options& options) {
  const rr_view& view = *raftrkt::require(arrays.out, "out");
  const bool row_major = raftrkt::has_layout(view, layout::row_major);
  T* data = raftrkt::matrix_data<T, int>(
      arrays.out, "out", row_major ? layout::row_major : layout::col_major);
  const int n = static_cast<int>(view.shape[0]);
  const int d = static_cast<int>(view.shape[1]);
  int* classes = raftrkt::vector_data<int32_t, int>(arrays.labels, "labels", n);
  if (options.n_clusters < 1) {
    throw logic_error("n-clusters: expected at least 1, got " +
                      std::to_string(options.n_clusters));
  }
  ML::Datasets::make_blobs(
      handle, data, classes, n, d, options.n_clusters, row_major, nullptr,
      nullptr, static_cast<T>(options.cluster_std), options.shuffle,
      static_cast<T>(options.center_box_min),
      static_cast<T>(options.center_box_max), options.seed);
}

}  // namespace

extern "C" {

int cuml_cluster_make_blobs(void* handle, const rr_view* out,
                            const rr_view* labels, int32_t n_clusters,
                            double cluster_std, int32_t shuffle,
                            double center_box_min, double center_box_max,
                            uint64_t seed) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [&] {
    const raft::handle_t& h = raftrkt::handle_of(handle);
    const int device = raftrkt::device_of(h);
    const raftrkt::device_scope scope{device};
    raftrkt::require_on_device(device, {{"out", out}, {"labels", labels}});
    raftrkt::sync_guard sync{raft::resource::get_cuda_stream(h)};
    const blob_arrays arrays{out, labels};
    const blob_options options{n_clusters,     cluster_std,    shuffle != 0,
                               center_box_min, center_box_max, seed};
    cumlrkt::on_floats(out, "out", [&](auto tag) {
      make_blobs<decltype(tag)>(h, arrays, options);
    });
    sync.finish();
  });
}

void cuml_cluster_make_blobs_job(void* args, size_t args_len,
                                 rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_make_blobs>(args, args_len, outcome,
                                             cumlrkt::last_error_slot());
}
}
