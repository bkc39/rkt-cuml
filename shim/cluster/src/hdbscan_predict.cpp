#include <cstddef>
#include <cstdint>
#include <cuml/cluster/hdbscan.hpp>
#include <cuml/common/distance_type.hpp>
#include <format>
#include <functional>
#include <initializer_list>
#include <memory>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <utility>

#include "cluster_kinds.hpp"
#include "cumlrkt/cluster.h"
#include "cumlrkt/common.hpp"
#include "cumlrkt/model.hpp"
#include "hdbscan_model.hpp"
#include "hdbscan_support.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"
#include "support.hpp"

namespace {

using cumlrkt::logic_error;
using cumlrkt::cluster::hdbscan_kind;
using cumlrkt::cluster::hdbscan_model;
using cumlrkt::cluster::in_device_scope;
using cumlrkt::cluster::matrix;
using cumlrkt::cluster::require_at_least;
using cumlrkt::cluster::rows_of;
using cumlrkt::cluster::settle;
using cumlrkt::cluster::hdbscan_limits::most_index;
using cumlrkt::cluster::hdbscan_limits::require_bounded_rows;
using cumlrkt::cluster::hdbscan_limits::require_index;
using raftrkt::any_extent;

constexpr auto metric = ML::distance::DistanceType::L2SqrtExpanded;

using model_use = cumlrkt::model_use<hdbscan_model>;

model_use used(const rr_view* model, const raft::handle_t& handle) {
  return cumlrkt::use_model<hdbscan_model>(
      model, "model", hdbscan_kind, raft::resource::get_cuda_stream(handle));
}

const float* checked_training(const hdbscan_model& m, const rr_view* view) {
  const matrix<float, int64_t> x =
      rows_of<float, int64_t>(view, "X", any_extent, any_extent);
  if (x.n != m.n_rows || x.d != m.n_cols) {
    throw logic_error(
        std::format("X: expected {}x{}, as the model was fit on, got {}x{}",
                    m.n_rows, m.n_cols, x.n, x.d));
  }
  return x.data;
}

struct query {
  matrix<float, int64_t> points;
  const float* x;
};

query checked_query(const hdbscan_model& m, const rr_view* x_view,
                    const rr_view* view) {
  const matrix<float, int64_t> points =
      rows_of<float, int64_t>(view, "points", any_extent, any_extent);
  if (points.n == 0) {
    throw logic_error("points: expected at least 1 row, got 0");
  }
  if (points.d != m.n_cols) {
    throw logic_error(std::format(
        "points: expected {} columns, as the model was fit on, got {}",
        m.n_cols, points.d));
  }
  const float* x = checked_training(m, x_view);
  if (m.n_rows < 2 * m.min_samples) {
    throw logic_error(std::format(
        "model: prediction needs at least {} training rows (2 × min-samples), "
        "got {}",
        2 * m.min_samples, m.n_rows));
  }
  require_index("points", points.n, points.d);
  return {points, x};
}

void require_entries(const char* name, int64_t rows, int64_t k) {
  if (k > 0 && rows > most_index / k) {
    throw logic_error(std::format(
        "{}: {} rows with {} clusters need {} entries, more than {}", name,
        rows, k, rows * k, most_index));
  }
}

struct prediction_outputs {
  rmm::device_uvector<int64_t> labels;
  rmm::device_uvector<float> probabilities;
};

struct membership_outputs {
  rmm::device_uvector<float> memberships;
};

void deposit_memberships(const raft::handle_t& handle, rr_deposit* out,
                         int64_t rows, int64_t k,
                         const std::function<void(float*)>& compute) {
  auto outputs = std::make_unique<membership_outputs>(membership_outputs{
      rmm::device_uvector<float>(static_cast<std::size_t>(rows * k),
                                 raft::resource::get_cuda_stream(handle))});
  if (k > 0) {
    compute(outputs->memberships.data());
  }
  const std::initializer_list<rr_view> regions{raftrkt::region(
      outputs->memberships.data(), rows, k, raftrkt::layout::row_major)};
  settle(handle);
  raftrkt::deposit(out, "out", handle, std::move(outputs), regions);
}

void approximate_predict(const raft::handle_t& handle, const rr_view* model,
                         const rr_view* x, const rr_view* view,
                         rr_deposit* out) {
  const model_use m = used(model, handle);
  const query q = checked_query(*m, x, view);
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  require_bounded_rows("points", q.points, stream);
  const auto rows = static_cast<std::size_t>(q.points.n);
  auto outputs = std::make_unique<prediction_outputs>(
      prediction_outputs{rmm::device_uvector<int64_t>(rows, stream),
                         rmm::device_uvector<float>(rows, stream)});
  ML::out_of_sample_predict(
      handle, m->tree, *m->pd, q.x, m->labels.data(), q.points.data, rows,
      metric, static_cast<int>(m->min_samples), outputs->labels.data(),
      outputs->probabilities.data());
  const std::initializer_list<rr_view> regions{
      raftrkt::region(outputs->labels),
      raftrkt::region(outputs->probabilities)};
  settle(handle);
  raftrkt::deposit(out, "out", handle, std::move(outputs), regions);
}

void membership_vector(const raft::handle_t& handle, const rr_view* model,
                       const rr_view* x, const rr_view* view, rr_deposit* out,
                       int64_t batch_size) {
  const model_use m = used(model, handle);
  const query q = checked_query(*m, x, view);
  require_at_least("batch-size", batch_size, 1);
  require_entries("points", q.points.n, m->k);
  require_bounded_rows("points", q.points,
                       raft::resource::get_cuda_stream(handle));
  deposit_memberships(handle, out, q.points.n, m->k, [&](float* into) {
    ML::compute_membership_vector(handle, m->tree, *m->pd, q.x, q.points.data,
                                  static_cast<std::size_t>(q.points.n),
                                  static_cast<int>(m->min_samples), metric,
                                  into, static_cast<std::size_t>(batch_size));
  });
}

void all_points(const raft::handle_t& handle, const rr_view* model,
                const rr_view* x, rr_deposit* out, int64_t batch_size) {
  const model_use m = used(model, handle);
  const float* training = checked_training(*m, x);
  require_at_least("batch-size", batch_size, 1);
  require_entries("model", m->n_rows, m->k);
  deposit_memberships(handle, out, m->n_rows, m->k, [&](float* into) {
    ML::compute_all_points_membership_vectors(
        handle, m->tree, *m->pd, training, metric, into,
        static_cast<std::size_t>(batch_size));
  });
}

using views = std::initializer_list<std::pair<const char*, const rr_view*>>;

template <typename Run>
int on_model(void* handle, views checked, Run&& run) {
  return raftrkt::translate_exceptions(cumlrkt::last_error_slot(), [&] {
    in_device_scope(handle, checked, std::forward<Run>(run));
  });
}

}  // namespace

extern "C" {

int cuml_cluster_hdbscan_approximate_predict(void* handle, const rr_view* model,
                                             const rr_view* x,
                                             const rr_view* points,
                                             rr_deposit* out) {
  return on_model(handle, {{"model", model}, {"X", x}, {"points", points}},
                  [&](const raft::handle_t& h) {
                    approximate_predict(h, model, x, points,
                                        raftrkt::require(out, "out"));
                  });
}

void cuml_cluster_hdbscan_approximate_predict_job(void* args, size_t args_len,
                                                  rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_hdbscan_approximate_predict>(
      args, args_len, outcome, cumlrkt::last_error_slot());
}

int cuml_cluster_hdbscan_membership_vector(void* handle, const rr_view* model,
                                           const rr_view* x,
                                           const rr_view* points,
                                           rr_deposit* out,
                                           int64_t batch_size) {
  return on_model(handle, {{"model", model}, {"X", x}, {"points", points}},
                  [&](const raft::handle_t& h) {
                    membership_vector(h, model, x, points,
                                      raftrkt::require(out, "out"), batch_size);
                  });
}

void cuml_cluster_hdbscan_membership_vector_job(void* args, size_t args_len,
                                                rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_hdbscan_membership_vector>(
      args, args_len, outcome, cumlrkt::last_error_slot());
}

int cuml_cluster_hdbscan_all_points_membership_vectors(void* handle,
                                                       const rr_view* model,
                                                       const rr_view* x,
                                                       rr_deposit* out,
                                                       int64_t batch_size) {
  return on_model(
      handle, {{"model", model}, {"X", x}}, [&](const raft::handle_t& h) {
        all_points(h, model, x, raftrkt::require(out, "out"), batch_size);
      });
}

void cuml_cluster_hdbscan_all_points_membership_vectors_job(
    void* args, size_t args_len, rr_job_outcome* outcome) {
  raftrkt::run_job<&cuml_cluster_hdbscan_all_points_membership_vectors>(
      args, args_len, outcome, cumlrkt::last_error_slot());
}
}
