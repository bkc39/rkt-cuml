#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cuml/cluster/kmeans.hpp>
#include <initializer_list>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <string>
#include <utility>

#include "cumlrkt/cluster.h"
#include "cumlrkt/common.hpp"
#include "raftrkt/job.hpp"

namespace {

using cumlrkt::logic_error;
using raftrkt::any_extent;
using raftrkt::layout;
using Params = ML::kmeans::KMeansParams;

struct fit_arrays {
  const rr_view* x;
  const rr_view* sample_weight;
  const rr_view* init;
  const rr_view* centroids;
  const rr_view* labels;
};

struct fit_options {
  int32_t n_clusters;
  int32_t init_method;
  int32_t max_iter;
  double tol;
  int32_t n_init;
  double oversampling_factor;
  int32_t batch_samples;
  uint64_t seed;
};

struct fit_results {
  double* inertia;
  int32_t* n_iter;
};

template <typename T>
struct matrix {
  T* data;
  int n;
  int d;
};

template <typename T>
matrix<T> rows_of(const rr_view* view, const char* name, int64_t rows,
                  int64_t cols) {
  rr_view v = *raftrkt::require(view, name);
  if (v.rank == 2 && (v.shape[0] == 0 || v.shape[1] == 0)) {
    v.strides[0] = 0;
    v.strides[1] = 0;
  }
  T* data =
      raftrkt::matrix_data<T, int>(&v, name, layout::row_major, rows, cols);
  return {data, static_cast<int>(v.shape[0]), static_cast<int>(v.shape[1])};
}

void require_at_least(const std::string& what, int64_t got, int64_t least) {
  if (got < least) {
    throw logic_error(what + ": expected at least " + std::to_string(least) +
                      ", got " + std::to_string(got));
  }
}

void require_some(const char* name, const char* axis, int extent) {
  if (extent == 0) {
    throw logic_error(std::string(name) + ": expected at least 1 " + axis +
                      ", got 0");
  }
}

Params::InitMethod init_method_of(const fit_arrays& arrays, int32_t code) {
  if (code != CUML_CLUSTER_KMEANS_INIT_KMEANS_PLUS_PLUS &&
      code != CUML_CLUSTER_KMEANS_INIT_RANDOM &&
      code != CUML_CLUSTER_KMEANS_INIT_ARRAY) {
    throw logic_error(
        "init: expected scalable-k-means++, k-means++, random or a centroids "
        "matrix");
  }
  const bool array = code == CUML_CLUSTER_KMEANS_INIT_ARRAY;
  if (array && arrays.init == nullptr) {
    throw logic_error("init: an array init needs a centroids matrix");
  }
  if (!array && arrays.init != nullptr) {
    throw logic_error("init: a centroids matrix needs the array init");
  }
  if (code == CUML_CLUSTER_KMEANS_INIT_RANDOM) {
    return Params::InitMethod::Random;
  }
  return array ? Params::InitMethod::Array : Params::InitMethod::KMeansPlusPlus;
}

struct sizing {
  int32_t n_clusters;
  int32_t batch_samples;
};

Params params_of(const sizing& size) {
  Params p;
  p.verbosity = rapids_logger::level_enum::warn;
  p.metric = ML::distance::DistanceType::L2Expanded;
  p.n_clusters = size.n_clusters;
  p.batch_samples = size.batch_samples;
  p.init_size = 0;
  p.device_buffer_samples = 0;
  return p;
}

template <typename T>
const T* weights_of(const rr_view* sample_weight, int n) {
  if (sample_weight == nullptr) {
    return nullptr;
  }
  return raftrkt::vector_data<T, int>(sample_weight, "sample-weight", n);
}

template <typename T>
void fit(const raft::handle_t& handle, const fit_arrays& arrays,
         const fit_options& options, const fit_results& results) {
  const matrix<T> x = rows_of<T>(arrays.x, "X", any_extent, any_extent);
  require_some("X", "column", x.d);
  require_at_least("n-clusters", options.n_clusters, 1);
  if (x.n < options.n_clusters) {
    throw logic_error("X: " + std::to_string(x.n) +
                      " samples are fewer than the " +
                      std::to_string(options.n_clusters) + " clusters");
  }
  const int k = options.n_clusters;
  const T* w = weights_of<T>(arrays.sample_weight, x.n);
  const Params::InitMethod method = init_method_of(arrays, options.init_method);
  const T* init = method == Params::InitMethod::Array
                      ? rows_of<T>(arrays.init, "init", k, x.d).data
                      : nullptr;
  require_at_least("max-iter", options.max_iter, 0);
  require_at_least("n-init", options.n_init, 1);
  require_at_least("max-samples-per-batch", options.batch_samples, 0);
  T* centroids = rows_of<T>(arrays.centroids, "centroids", k, x.d).data;
  int* labels =
      raftrkt::vector_data<int32_t, int>(arrays.labels, "labels", x.n);

  Params p = params_of({k, options.batch_samples});
  p.init = method;
  p.max_iter = options.max_iter;
  p.tol = options.tol;
  p.n_init = options.n_init;
  p.oversampling_factor = options.oversampling_factor;
  p.rng_state.seed = options.seed;
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  if (init != nullptr) {
    raftrkt::cuda_check(
        cudaMemcpyAsync(centroids, init,
                        sizeof(T) * static_cast<std::size_t>(k) *
                            static_cast<std::size_t>(x.d),
                        cudaMemcpyDeviceToDevice, stream.value()),
        "cudaMemcpyAsync");
  }
  T fit_inertia = 0;
  int n_iter = 0;
  ML::kmeans::fit(handle, p, x.data, x.n, x.d, w, centroids, fit_inertia,
                  n_iter);
  T inertia = 0;
  ML::kmeans::predict(handle, p, centroids, x.data, x.n, x.d, w, true, labels,
                      inertia);
  *results.inertia = static_cast<double>(inertia);
  *results.n_iter = n_iter;
}

template <typename T>
struct applied {
  matrix<T> centroids;
  matrix<T> x;
};

template <typename T, typename Arrays>
applied<T> apply_inputs(const Arrays& arrays) {
  const matrix<T> c =
      rows_of<T>(arrays.centroids, "centroids", any_extent, any_extent);
  require_some("centroids", "row", c.n);
  require_some("centroids", "column", c.d);
  const matrix<T> y = rows_of<T>(arrays.x, "X", any_extent, c.d);
  require_some("X", "row", y.n);
  return {c, y};
}

struct predict_arrays {
  const rr_view* centroids;
  const rr_view* x;
  const rr_view* sample_weight;
  const rr_view* labels;
};

template <typename T>
void predict(const raft::handle_t& handle, const predict_arrays& arrays,
             int32_t batch_samples, double* inertia) {
  const applied<T> in = apply_inputs<T>(arrays);
  const T* w = weights_of<T>(arrays.sample_weight, in.x.n);
  require_at_least("max-samples-per-batch", batch_samples, 0);
  const Params p = params_of({in.centroids.n, batch_samples});
  T found = 0;
  if (arrays.labels == nullptr) {
    auto scratch = raft::make_device_vector<int32_t, int>(handle, in.x.n);
    ML::kmeans::predict(handle, p, in.centroids.data, in.x.data, in.x.n, in.x.d,
                        w, true, scratch.data_handle(), found);
  } else {
    int* labels =
        raftrkt::vector_data<int32_t, int>(arrays.labels, "labels", in.x.n);
    ML::kmeans::predict(handle, p, in.centroids.data, in.x.data, in.x.n, in.x.d,
                        w, true, labels, found);
  }
  *inertia = static_cast<double>(found);
}

struct transform_arrays {
  const rr_view* centroids;
  const rr_view* x;
  const rr_view* distances;
};

template <typename T>
void transform(const raft::handle_t& handle, const transform_arrays& arrays,
               int32_t batch_samples) {
  const applied<T> in = apply_inputs<T>(arrays);
  require_at_least("max-samples-per-batch", batch_samples, 0);
  T* out =
      rows_of<T>(arrays.distances, "distances", in.x.n, in.centroids.n).data;
  const Params p = params_of({in.centroids.n, batch_samples});
  ML::kmeans::transform(handle, p, in.centroids.data, in.x.data, in.x.n, in.x.d,
                        out);
}

template <typename Fn>
void on_device(
    void* handle,
    std::initializer_list<std::pair<const char*, const rr_view*>> views,
    const rr_view* first, const char* first_name, Fn&& fn) {
  const raft::handle_t& h = raftrkt::handle_of(handle);
  const int device = raftrkt::device_of(h);
  const raftrkt::device_scope scope{device};
  raftrkt::require_on_device(device, views);
  raftrkt::sync_guard sync{raft::resource::get_cuda_stream(h)};
  cumlrkt::on_floats(first, first_name,
                     [&](auto tag) { fn(h, decltype(tag){}); });
  sync.finish();
}

}  // namespace

extern "C" {

int cuml_cluster_kmeans_fit(void* handle, const rr_view* x,
                            const rr_view* sample_weight, const rr_view* init,
                            const rr_view* centroids, const rr_view* labels,
                            int32_t n_clusters, int32_t init_method,
                            int32_t max_iter, double tol, int32_t n_init,
                            double oversampling_factor, int32_t batch_samples,
                            uint64_t seed, double* inertia, int32_t* n_iter) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [&] {
    const fit_results results{raftrkt::require(inertia, "inertia"),
                              raftrkt::require(n_iter, "n-iter")};
    *results.inertia = 0;
    *results.n_iter = 0;
    const fit_arrays arrays{x, sample_weight, init, centroids, labels};
    const fit_options options{n_clusters,    init_method, max_iter,
                              tol,           n_init,      oversampling_factor,
                              batch_samples, seed};
    on_device(handle,
              {{"X", x},
               {"sample-weight", sample_weight},
               {"init", init},
               {"centroids", centroids},
               {"labels", labels}},
              x, "X", [&](const raft::handle_t& h, auto tag) {
                fit<decltype(tag)>(h, arrays, options, results);
              });
  });
}

int cuml_cluster_kmeans_predict(void* handle, const rr_view* centroids,
                                const rr_view* x, const rr_view* sample_weight,
                                const rr_view* labels, int32_t batch_samples,
                                double* inertia) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [&] {
    *raftrkt::require(inertia, "inertia") = 0;
    const predict_arrays arrays{centroids, x, sample_weight, labels};
    on_device(handle,
              {{"centroids", centroids},
               {"X", x},
               {"sample-weight", sample_weight},
               {"labels", labels}},
              centroids, "centroids", [&](const raft::handle_t& h, auto tag) {
                predict<decltype(tag)>(h, arrays, batch_samples, inertia);
              });
  });
}

int cuml_cluster_kmeans_transform(void* handle, const rr_view* centroids,
                                  const rr_view* x, const rr_view* distances,
                                  int32_t batch_samples) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [&] {
    const transform_arrays arrays{centroids, x, distances};
    on_device(handle,
              {{"centroids", centroids}, {"X", x}, {"distances", distances}},
              centroids, "centroids", [&](const raft::handle_t& h, auto tag) {
                transform<decltype(tag)>(h, arrays, batch_samples);
              });
  });
}

void cuml_cluster_kmeans_fit_job(void* args, size_t args_len,
                                 rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_kmeans_fit>(args, args_len, outcome,
                                             cumlrkt::last_error_slot());
}

void cuml_cluster_kmeans_predict_job(void* args, size_t args_len,
                                     rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_kmeans_predict>(args, args_len, outcome,
                                                 cumlrkt::last_error_slot());
}

void cuml_cluster_kmeans_transform_job(void* args, size_t args_len,
                                       rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_kmeans_transform>(args, args_len, outcome,
                                                   cumlrkt::last_error_slot());
}
}
