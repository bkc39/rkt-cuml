#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <format>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <string>

#include "cumlrkt/common.hpp"
#include "cumlrkt/graph.hpp"
#include "cumlrkt/limits.hpp"
#include "cumlrkt/support.hpp"
#include "raftrkt/view.hpp"

namespace cumlrkt {

namespace {

struct scan {
  int64_t first;
  int64_t positive;
  double total;
};

struct entry_scan {
  const int32_t* rows;
  const int32_t* cols;
  const float* vals;
  int64_t nnz;
  int64_t n;

  __device__ scan operator()(int64_t i) const {
    const int64_t r = rows[i];
    const int64_t c = cols[i];
    const float w = vals[i];
    const bool faulty = r < 0 || r >= n || c < 0 || c >= n || r == c ||
                        !(w >= 0) || isinf(w) || (w > 0 && w < FLT_MIN);
    return {faulty ? i : nnz, w > 0 ? 1 : 0, static_cast<double>(w)};
  }
};

struct earliest {
  __device__ scan operator()(const scan& a, const scan& b) const {
    return {a.first < b.first ? a.first : b.first, a.positive | b.positive,
            a.total + b.total};
  }
};

void require_entries(const char* name, int64_t got, int64_t nnz) {
  if (got != nnz) {
    throw logic_error(std::format(
        "{}: expected {} entries, as rows has, got {}", name, nnz, got));
  }
}

}  // namespace

graph_scan scan_graph(const coo& c, int64_t n, rmm::cuda_stream_view stream) {
  const thrust::counting_iterator<int64_t> entries(0);
  const scan found = thrust::transform_reduce(
      rmm::exec_policy(stream), entries, entries + c.nnz,
      entry_scan{c.rows, c.cols, c.vals, c.nnz, n}, scan{c.nnz, 0, 0},
      earliest{});
  return {found.first == c.nnz ? -1 : found.first, found.positive != 0,
          found.total};
}

coo checked_graph(const graph_views& g) {
  coo c{raftrkt::vector_data<int32_t>(g.rows, "rows"),
        raftrkt::vector_data<int32_t>(g.cols, "cols"),
        raftrkt::vector_data<float>(g.vals, "vals"), g.rows->shape[0]};
  require_entries("cols", g.cols->shape[0], c.nnz);
  require_entries("vals", g.vals->shape[0], c.nnz);
  return c;
}

void require_samples(int64_t n, int64_t nnz, int64_t least) {
  require_at_least("n-samples", n, least);
  if (n > most_index) {
    throw logic_error(
        std::format("n-samples: expected at most {}, got {}", most_index, n));
  }
  if (nnz > most_index - n) {
    throw logic_error(std::format(
        "rows: {} entries and {} samples do not fit the index type", nnz, n));
  }
}

std::string fault_message(const coo& c, int64_t i, int64_t n,
                          rmm::cuda_stream_view stream) {
  const int64_t r = read_one(c.rows, i, stream);
  const int64_t col = read_one(c.cols, i, stream);
  const double w = read_one(c.vals, i, stream);
  if (r < 0 || r >= n || col < 0 || col >= n) {
    const bool row = r < 0 || r >= n;
    return std::format("{}: entry {}: expected an index in [0, {}), got {}",
                       row ? "rows" : "cols", i, n, row ? r : col);
  }
  if (r == col) {
    return std::format(
        "rows: entry {}: self-loop at sample {}; leave the diagonal out", i, r);
  }
  if (w > 0 && w < FLT_MIN) {
    return std::format(
        "vals: entry {}: expected 0 or a weight of at least {}, got {}", i,
        printed(FLT_MIN), printed(w));
  }
  return std::format(
      "vals: entry {}: expected a finite weight of at least 0, got {}", i,
      printed(w));
}

void require_sound_graph(const coo& c, int64_t n,
                         rmm::cuda_stream_view stream) {
  const graph_scan s = scan_graph(c, n, stream);
  if (s.first_fault >= 0) {
    throw logic_error(fault_message(c, s.first_fault, n, stream));
  }
  if (s.total > most_float) {
    throw logic_error(std::format("vals: the weights sum to {}, more than {}",
                                  printed(s.total), printed(most_float)));
  }
  if (!s.any_positive) {
    throw logic_error("vals: expected at least one positive weight, got none");
  }
}

}  // namespace cumlrkt
