#include <cstddef>
#include <cstdint>
#include <cuml/cluster/hdbscan.hpp>
#include <cuml/common/distance_type.hpp>
#include <format>
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
#include "limits.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"
#include "support.hpp"

namespace {

using cumlrkt::logic_error;
using cumlrkt::cluster::guarded;
using cumlrkt::cluster::hdbscan_kind;
using cumlrkt::cluster::hdbscan_model;
using cumlrkt::cluster::matrix;
using cumlrkt::cluster::most_index;
using cumlrkt::cluster::require_at_least;
using cumlrkt::cluster::require_bounded_rows;
using cumlrkt::cluster::require_index;
using cumlrkt::cluster::rows_of;
using cumlrkt::cluster::settle;
using raftrkt::any_extent;

constexpr auto metric = ML::distance::DistanceType::L2SqrtExpanded;

using model_use = cumlrkt::model_use<hdbscan_model>;

struct fitted_views {
  const rr_view* model;
  const rr_view* x;
};

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

query checked_query(const hdbscan_model& m, const fitted_views& fitted,
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
  const float* x = checked_training(m, fitted.x);
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

void approximate_predict(const raft::handle_t& handle,
                         const fitted_views& fitted, const rr_view* view,
                         rr_deposit* out) {
  const model_use m = used(fitted.model, handle);
  const query q = checked_query(*m, fitted, view);
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

struct membership_request {
  const char* name;
  int64_t rows;
  int64_t batch_size;
  rr_deposit* out;
};

template <typename Check, typename Compute>
void memberships_for(const raft::handle_t& handle, const hdbscan_model& m,
                     const membership_request& r, Check&& check,
                     Compute&& compute) {
  require_at_least("batch-size", r.batch_size, 1);
  require_entries(r.name, r.rows, m.k);
  check();
  auto memberships = std::make_unique<rmm::device_uvector<float>>(
      static_cast<std::size_t>(r.rows * m.k),
      raft::resource::get_cuda_stream(handle));
  if (m.k > 0) {
    compute(memberships->data(), static_cast<std::size_t>(r.batch_size));
  }
  const std::initializer_list<rr_view> regions{raftrkt::region(
      memberships->data(), r.rows, m.k, raftrkt::layout::row_major)};
  settle(handle);
  raftrkt::deposit(r.out, "out", handle, std::move(memberships), regions);
}

void membership_vector(const raft::handle_t& handle, const fitted_views& fitted,
                       const rr_view* view, rr_deposit* out,
                       int64_t batch_size) {
  const model_use m = used(fitted.model, handle);
  const query q = checked_query(*m, fitted, view);
  memberships_for(
      handle, *m, {"points", q.points.n, batch_size, out},
      [&] {
        require_bounded_rows("points", q.points,
                             raft::resource::get_cuda_stream(handle));
      },
      [&](float* into, std::size_t batch) {
        ML::compute_membership_vector(
            handle, m->tree, *m->pd, q.x, q.points.data,
            static_cast<std::size_t>(q.points.n),
            static_cast<int>(m->min_samples), metric, into, batch);
      });
}

void all_points(const raft::handle_t& handle, const fitted_views& fitted,
                rr_deposit* out, int64_t batch_size) {
  const model_use m = used(fitted.model, handle);
  const float* training = checked_training(*m, fitted.x);
  memberships_for(
      handle, *m, {"model", m->n_rows, batch_size, out}, [] {},
      [&](float* into, std::size_t batch) {
        ML::compute_all_points_membership_vectors(
            handle, m->tree, *m->pd, training, metric, into, batch);
      });
}

}  // namespace

extern "C" {

int cuml_cluster_hdbscan_approximate_predict(void* handle, const rr_view* model,
                                             const rr_view* x,
                                             const rr_view* points,
                                             rr_deposit* out) {
  return guarded(handle, {{"model", model}, {"X", x}, {"points", points}},
                 [&](const raft::handle_t& h) {
                   approximate_predict(h, {model, x}, points,
                                       raftrkt::require(out, "out"));
                 });
}

CUMLRKT_JOB(cuml_cluster_hdbscan_approximate_predict)

int cuml_cluster_hdbscan_membership_vector(void* handle, const rr_view* model,
                                           const rr_view* x,
                                           const rr_view* points,
                                           rr_deposit* out,
                                           int64_t batch_size) {
  return guarded(handle, {{"model", model}, {"X", x}, {"points", points}},
                 [&](const raft::handle_t& h) {
                   membership_vector(h, {model, x}, points,
                                     raftrkt::require(out, "out"), batch_size);
                 });
}

CUMLRKT_JOB(cuml_cluster_hdbscan_membership_vector)

int cuml_cluster_hdbscan_all_points_membership_vectors(void* handle,
                                                       const rr_view* model,
                                                       const rr_view* x,
                                                       rr_deposit* out,
                                                       int64_t batch_size) {
  return guarded(
      handle, {{"model", model}, {"X", x}}, [&](const raft::handle_t& h) {
        all_points(h, {model, x}, raftrkt::require(out, "out"), batch_size);
      });
}

CUMLRKT_JOB(cuml_cluster_hdbscan_all_points_membership_vectors)
}
