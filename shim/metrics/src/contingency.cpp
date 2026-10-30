#include <cstddef>
#include <cstdint>
#include <cuml/metrics/metrics.hpp>
#include <format>
#include <initializer_list>
#include <memory>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <utility>

#include "checks.hpp"
#include "contingency_device.hpp"
#include "cumlrkt/labels.hpp"
#include "cumlrkt/metrics.h"
#include "cumlrkt/support.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"

namespace {

using cumlrkt::dense_codes;
using cumlrkt::dense_labels;
using cumlrkt::guarded;
using cumlrkt::logic_error;
using cumlrkt::on_labels;
using cumlrkt::settle;
using cumlrkt::metrics::count_pairs;
using cumlrkt::metrics::labels_of;
using cumlrkt::metrics::most_distinct_labels;

template <typename T>
dense_labels<T> distinct(const T* labels, int64_t n, const char* name,
                         rmm::cuda_stream_view stream) {
  dense_labels<T> found = dense_codes(labels, n, stream);
  if (found.k > most_distinct_labels) {
    throw logic_error(
        std::format("{}: expected at most {} distinct labels, got {}", name,
                    most_distinct_labels, found.k));
  }
  return found;
}

struct label_pair {
  const rr_view* truth;
  const rr_view* pred;
};

template <typename Use>
void on_label_pair(label_pair labels, Use&& use) {
  on_labels(labels.truth, "labels-true", [&](auto t) {
    using T = decltype(t);
    const T* a = labels_of<T>(labels.truth, "labels-true", raftrkt::any_extent);
    const int64_t n = labels.truth->shape[0];
    on_labels(labels.pred, "labels-pred", [&](auto p) {
      using P = decltype(p);
      const P* b = labels_of<P>(labels.pred, "labels-pred", n);
      use(a, b, n);
    });
  });
}

template <typename T, typename P>
struct table {
  rmm::device_uvector<int64_t> counts;
  rmm::device_uvector<T> rows;
  rmm::device_uvector<P> columns;
};

template <typename T, typename P>
void tabulate(const raft::handle_t& handle, const T* a, const P* b, int64_t n,
              rr_deposit* out) {
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  dense_labels<T> rows = distinct(a, n, "labels-true", stream);
  dense_labels<P> columns = distinct(b, n, "labels-pred", stream);
  const int64_t k1 = rows.k;
  const int64_t k2 = columns.k;
  rmm::device_uvector<int64_t> counts(static_cast<std::size_t>(k1 * k2),
                                      stream);
  if (n > 0) {
    count_pairs(counts.data(), rows.codes.data(), columns.codes.data(), n, k1,
                k2, stream);
  }
  auto found = std::make_unique<table<T, P>>(table<T, P>{
      std::move(counts), std::move(rows.values), std::move(columns.values)});
  const std::initializer_list<rr_view> regions{
      raftrkt::region(found->counts.data(), k1, k2, raftrkt::layout::row_major),
      raftrkt::region(found->rows.data(), k1),
      raftrkt::region(found->columns.data(), k2)};
  settle(handle);
  raftrkt::deposit(out, "out", handle, std::move(found), regions);
}

template <typename T, typename P>
double rand_index(const raft::handle_t& handle, const T* a, const P* b,
                  int64_t n) {
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  const dense_labels<T> rows = distinct(a, n, "labels-true", stream);
  const dense_labels<P> columns = distinct(b, n, "labels-pred", stream);
  if (n < 2) {
    return 1.0;
  }
  const double score = ML::Metrics::adjusted_rand_index(
      handle, rows.codes.data(), columns.codes.data(), static_cast<int>(n));
  settle(handle);
  return score;
}

}  // namespace

extern "C" {

int cuml_metrics_contingency_matrix(void* handle, const rr_view* labels_true,
                                    const rr_view* labels_pred,
                                    rr_deposit* out) {
  return guarded(handle,
                 {{"labels-true", labels_true}, {"labels-pred", labels_pred}},
                 [&](const raft::handle_t& h) {
                   rr_deposit* const into = raftrkt::require(out, "out");
                   on_label_pair({labels_true, labels_pred},
                                 [&](const auto* a, const auto* b, int64_t n) {
                                   tabulate(h, a, b, n, into);
                                 });
                 });
}

int cuml_metrics_adjusted_rand_score(void* handle, const rr_view* labels_true,
                                     const rr_view* labels_pred,
                                     double* score) {
  return guarded(handle,
                 {{"labels-true", labels_true}, {"labels-pred", labels_pred}},
                 [&](const raft::handle_t& h) {
                   double* const into = raftrkt::require(score, "score");
                   on_label_pair({labels_true, labels_pred},
                                 [&](const auto* a, const auto* b, int64_t n) {
                                   *into = rand_index(h, a, b, n);
                                 });
                 });
}

CUMLRKT_JOB(cuml_metrics_contingency_matrix)
CUMLRKT_JOB(cuml_metrics_adjusted_rand_score)
}
