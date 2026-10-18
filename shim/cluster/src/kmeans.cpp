#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cuml/cluster/kmeans.hpp>
#include <initializer_list>
#include <limits>
#include <memory>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <string>
#include <utility>

#include "cumlrkt/cluster.h"
#include "cumlrkt/common.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "support.hpp"

namespace {

using cumlrkt::logic_error;
using cumlrkt::cluster::matrix;
using cumlrkt::cluster::on_device;
using cumlrkt::cluster::require_at_least;
using cumlrkt::cluster::require_non_negative;
using cumlrkt::cluster::require_positive;
using cumlrkt::cluster::require_some;
using cumlrkt::cluster::rows_of;
using cumlrkt::cluster::settle;
using raftrkt::any_extent;
using raftrkt::layout;
using Params = ML::kmeans::KMeansParams;

struct fit_arrays {
  const rr_view* x;
  const rr_view* sample_weight;
  const rr_view* init;
  rr_deposit* out;
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
struct fit_outputs {
  rmm::device_uvector<T> centroids;
  rmm::device_uvector<int32_t> labels;
};

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
  require_positive("tol", options.tol);
  require_non_negative("oversampling-factor", options.oversampling_factor);
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  const auto count =
      static_cast<std::size_t>(k) * static_cast<std::size_t>(x.d);
  auto outputs = std::make_unique<fit_outputs<T>>(fit_outputs<T>{
      rmm::device_uvector<T>(count, stream),
      rmm::device_uvector<int32_t>(static_cast<std::size_t>(x.n), stream)});
  T* centroids = outputs->centroids.data();
  int* labels = outputs->labels.data();

  Params p = params_of({k, options.batch_samples});
  p.init = method;
  p.max_iter = options.max_iter;
  p.tol = options.tol;
  p.n_init = options.n_init;
  p.oversampling_factor = options.oversampling_factor;
  p.rng_state.seed = options.seed;
  if (init != nullptr) {
    raftrkt::cuda_check(
        cudaMemcpyAsync(centroids, init, sizeof(T) * count,
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
  const std::initializer_list<rr_view> regions{
      raftrkt::region(centroids, k, x.d, layout::row_major),
      raftrkt::region(outputs->labels)};
  settle(handle);
  raftrkt::deposit(arrays.out, "out", handle, std::move(outputs), regions);
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
  rr_deposit* out;
};

template <typename T>
void predict(const raft::handle_t& handle, const predict_arrays& arrays,
             int32_t batch_samples) {
  const applied<T> in = apply_inputs<T>(arrays);
  require_at_least("max-samples-per-batch", batch_samples, 0);
  const Params p = params_of({in.centroids.n, batch_samples});
  auto labels = std::make_unique<rmm::device_uvector<int32_t>>(
      static_cast<std::size_t>(in.x.n),
      raft::resource::get_cuda_stream(handle));
  T inertia = 0;
  ML::kmeans::predict(handle, p, in.centroids.data, in.x.data, in.x.n, in.x.d,
                      nullptr, true, labels->data(), inertia);
  const std::initializer_list<rr_view> regions{raftrkt::region(*labels)};
  settle(handle);
  raftrkt::deposit(arrays.out, "out", handle, std::move(labels), regions);
}

struct score_arrays {
  const rr_view* centroids;
  const rr_view* x;
  const rr_view* sample_weight;
};

template <typename T>
void score(const raft::handle_t& handle, const score_arrays& arrays,
           int32_t batch_samples, double* inertia) {
  const applied<T> in = apply_inputs<T>(arrays);
  const T* w = weights_of<T>(arrays.sample_weight, in.x.n);
  require_at_least("max-samples-per-batch", batch_samples, 0);
  const Params p = params_of({in.centroids.n, batch_samples});
  auto scratch = raft::make_device_vector<int32_t, int>(handle, in.x.n);
  T found = 0;
  ML::kmeans::predict(handle, p, in.centroids.data, in.x.data, in.x.n, in.x.d,
                      w, true, scratch.data_handle(), found);
  *inertia = static_cast<double>(found);
}

struct transform_arrays {
  const rr_view* centroids;
  const rr_view* x;
  rr_deposit* out;
};

template <typename T>
void transform(const raft::handle_t& handle, const transform_arrays& arrays,
               int32_t batch_samples) {
  const applied<T> in = apply_inputs<T>(arrays);
  require_at_least("max-samples-per-batch", batch_samples, 0);
  const int64_t count = static_cast<int64_t>(in.x.n) * in.centroids.n;
  if (count > std::numeric_limits<int>::max()) {
    throw logic_error("distances: " + std::to_string(in.x.n) + "x" +
                      std::to_string(in.centroids.n) +
                      " elements do not fit the index type");
  }
  const Params p = params_of({in.centroids.n, batch_samples});
  auto distances = std::make_unique<rmm::device_uvector<T>>(
      static_cast<std::size_t>(count), raft::resource::get_cuda_stream(handle));
  ML::kmeans::transform(handle, p, in.centroids.data, in.x.data, in.x.n, in.x.d,
                        distances->data());
  const std::initializer_list<rr_view> regions{raftrkt::region(
      distances->data(), in.x.n, in.centroids.n, layout::row_major)};
  settle(handle);
  raftrkt::deposit(arrays.out, "out", handle, std::move(distances), regions);
}

}  // namespace

extern "C" {

int cuml_cluster_kmeans_fit(void* handle, const rr_view* x,
                            const rr_view* sample_weight, const rr_view* init,
                            rr_deposit* out, int32_t n_clusters,
                            int32_t init_method, int32_t max_iter, double tol,
                            int32_t n_init, double oversampling_factor,
                            int32_t batch_samples, uint64_t seed,
                            double* inertia, int32_t* n_iter) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [&] {
    const fit_results results{raftrkt::require(inertia, "inertia"),
                              raftrkt::require(n_iter, "n-iter")};
    *results.inertia = 0;
    *results.n_iter = 0;
    const fit_arrays arrays{x, sample_weight, init,
                            raftrkt::require(out, "out")};
    const fit_options options{n_clusters,    init_method, max_iter,
                              tol,           n_init,      oversampling_factor,
                              batch_samples, seed};
    on_device(handle,
              {{"X", x}, {"sample-weight", sample_weight}, {"init", init}}, x,
              "X", [&](const raft::handle_t& h, auto tag) {
                fit<decltype(tag)>(h, arrays, options, results);
              });
  });
}

int cuml_cluster_kmeans_predict(void* handle, const rr_view* centroids,
                                const rr_view* x, rr_deposit* out,
                                int32_t batch_samples) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [&] {
    const predict_arrays arrays{centroids, x, raftrkt::require(out, "out")};
    on_device(handle, {{"centroids", centroids}, {"X", x}}, centroids,
              "centroids", [&](const raft::handle_t& h, auto tag) {
                predict<decltype(tag)>(h, arrays, batch_samples);
              });
  });
}

int cuml_cluster_kmeans_score(void* handle, const rr_view* centroids,
                              const rr_view* x, const rr_view* sample_weight,
                              int32_t batch_samples, double* inertia) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [&] {
    *raftrkt::require(inertia, "inertia") = 0;
    const score_arrays arrays{centroids, x, sample_weight};
    on_device(
        handle,
        {{"centroids", centroids}, {"X", x}, {"sample-weight", sample_weight}},
        centroids, "centroids", [&](const raft::handle_t& h, auto tag) {
          score<decltype(tag)>(h, arrays, batch_samples, inertia);
        });
  });
}

int cuml_cluster_kmeans_transform(void* handle, const rr_view* centroids,
                                  const rr_view* x, rr_deposit* out,
                                  int32_t batch_samples) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [&] {
    const transform_arrays arrays{centroids, x, raftrkt::require(out, "out")};
    on_device(handle, {{"centroids", centroids}, {"X", x}}, centroids,
              "centroids", [&](const raft::handle_t& h, auto tag) {
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

void cuml_cluster_kmeans_score_job(void* args, size_t args_len,
                                   rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_kmeans_score>(args, args_len, outcome,
                                               cumlrkt::last_error_slot());
}

void cuml_cluster_kmeans_transform_job(void* args, size_t args_len,
                                       rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_kmeans_transform>(args, args_len, outcome,
                                                   cumlrkt::last_error_slot());
}
}
