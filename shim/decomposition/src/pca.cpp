#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuml/decomposition/params.hpp>
#include <cuml/decomposition/pca.hpp>
#include <format>
#include <initializer_list>
#include <memory>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <string>
#include <type_traits>
#include <utility>

#include "cumlrkt/decomposition.h"
#include "cumlrkt/support.hpp"
#include "pca_device.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"

namespace {

using cumlrkt::guarded;
using cumlrkt::logic_error;
using cumlrkt::matrix;
using cumlrkt::most_float;
using cumlrkt::most_index;
using cumlrkt::most_value;
using cumlrkt::on_floats;
using cumlrkt::require_at_least;
using cumlrkt::require_at_most;
using cumlrkt::require_finite;
using cumlrkt::require_finite_non_negative;
using cumlrkt::require_float;
using cumlrkt::require_magnitudes;
using cumlrkt::require_some;
using cumlrkt::rows_of;
using cumlrkt::settle;
using cumlrkt::decomposition::bounds_of;
using cumlrkt::decomposition::fit_arrays;
using cumlrkt::decomposition::post_fit;
using cumlrkt::decomposition::to_columns;
using cumlrkt::decomposition::to_rows;
using raftrkt::any_extent;
using raftrkt::layout;

constexpr int64_t most_columns = 46340;
constexpr double overflow_margin = 4.0;

template <typename T>
using buffer = rmm::device_uvector<T>;

template <typename T>
buffer<T> scratch(int64_t count, rmm::cuda_stream_view stream) {
  return buffer<T>(static_cast<std::size_t>(count), stream);
}

void require_rows(int64_t n) {
  if (n > most_index) {
    throw logic_error(std::format("X: {} rows do not fit the index type", n));
  }
}

struct fit_request {
  rr_deposit* out;
  bool all_components;
  int64_t n_components;
  int32_t solver;
  double tol;
  int64_t iterations;
  double* noise_variance;
  int64_t* n_samples;
};

void require_parameters(const fit_request& r, int64_t m) {
  if (r.solver != CUML_DECOMPOSITION_SOLVER_FULL &&
      r.solver != CUML_DECOMPOSITION_SOLVER_JACOBI) {
    throw logic_error("solver: expected full or jacobi");
  }
  if (!r.all_components) {
    require_at_least("n-components", r.n_components, 1);
    require_at_most("n-components", r.n_components, m);
  }
  require_finite_non_negative("tol", r.tol);
  require_float("tol", r.tol);
  require_at_least("iterations", r.iterations, 1);
  require_at_most("iterations", r.iterations, most_index);
}

template <typename T>
void require_distinct_rows(const matrix<T, int64_t>& x,
                           rmm::cuda_stream_view stream) {
  if (cumlrkt::first_row_unlike_first(x.data, x.n, x.d, stream) < 0) {
    throw logic_error("X: expected at least 2 distinct rows, got 1");
  }
}

template <typename T>
struct fit_outputs {
  fit_outputs(int64_t k, int64_t d, rmm::cuda_stream_view stream)
      : components(scratch<T>(k * d, stream)),
        variance(scratch<T>(k, stream)),
        ratio(scratch<T>(k, stream)),
        singular(scratch<T>(k, stream)),
        mean(scratch<T>(d, stream)) {}
  buffer<T> components;
  buffer<T> variance;
  buffer<T> ratio;
  buffer<T> singular;
  buffer<T> mean;
};

struct pca_shape {
  int64_t rows;
  int64_t cols;
  int64_t components;
};

ML::paramsPCA pca_params(const pca_shape& shape) {
  ML::paramsPCA p;
  p.n_rows = static_cast<std::size_t>(shape.rows);
  p.n_cols = static_cast<std::size_t>(shape.cols);
  p.n_components = static_cast<std::size_t>(shape.components);
  p.whiten = false;
  return p;
}

template <typename T>
void fit_into(const raft::handle_t& handle, const rr_view* view,
              const fit_request& r) {
  const matrix<T, int64_t> x =
      rows_of<T, int64_t>(view, "X", any_extent, any_extent);
  require_some("X", "row", x.n, 2);
  require_some("X", "column", x.d, 2);
  require_rows(x.n);
  if (x.d > most_columns) {
    throw logic_error(std::format(
        "X: {} columns: a {}x{} covariance does not fit the index type", x.d,
        x.d, x.d));
  }
  const int64_t m = std::min(x.n, x.d);
  require_parameters(r, m);
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  require_finite("X", x, stream);
  require_magnitudes(
      "X", x,
      std::sqrt(most_value<T> / (overflow_margin * static_cast<double>(x.n) *
                                 static_cast<double>(x.d))),
      stream);
  require_distinct_rows(x, stream);
  const int64_t k = r.all_components ? m : r.n_components;
  auto outputs = std::make_unique<fit_outputs<T>>(k, x.d, stream);
  double noise_variance = 0;
  {
    buffer<T> columns = scratch<T>(x.n * x.d, stream);
    to_columns<T>(handle, x.data, x.n, x.d, nullptr, columns.data());
    buffer<T> components = scratch<T>(m * x.d, stream);
    buffer<T> variance = scratch<T>(m, stream);
    buffer<T> ratio = scratch<T>(m, stream);
    buffer<T> singular = scratch<T>(m, stream);
    buffer<T> noise = scratch<T>(1, stream);
    buffer<double> totals = scratch<double>(2, stream);
    ML::paramsPCA p = pca_params({.rows = x.n, .cols = x.d, .components = m});
    p.algorithm = r.solver == CUML_DECOMPOSITION_SOLVER_JACOBI
                      ? ML::solver::COV_EIG_JACOBI
                      : ML::solver::COV_EIG_DQ;
    p.tol = static_cast<float>(r.tol);
    p.n_iterations = static_cast<uint32_t>(r.iterations);
    ML::pcaFit(handle, columns.data(), components.data(), variance.data(),
               ratio.data(), singular.data(), outputs->mean.data(),
               noise.data(), p);
    post_fit<T>(
        handle, components.data(), variance.data(), m, k, x.d, x.n,
        fit_arrays<T>{outputs->components.data(), outputs->variance.data(),
                      outputs->ratio.data(), outputs->singular.data()},
        totals.data());
    noise_variance = cumlrkt::read_one(totals.data(), 1, stream);
  }
  settle(handle);
  const std::initializer_list<rr_view> regions{
      raftrkt::region(outputs->components.data(), k, x.d, layout::row_major),
      raftrkt::region(outputs->variance), raftrkt::region(outputs->ratio),
      raftrkt::region(outputs->singular), raftrkt::region(outputs->mean)};
  raftrkt::deposit(r.out, "out", handle, std::move(outputs), regions);
  *r.noise_variance = noise_variance;
  *r.n_samples = x.n;
}

struct apply_views {
  const rr_view* x;
  const rr_view* components;
  const rr_view* mean;
  const rr_view* variance;
  rr_deposit* out;
  bool whiten;
};

template <typename T>
const T* explained_variance(const apply_views& v, int64_t k) {
  if (!v.whiten) {
    return nullptr;
  }
  if (v.variance == nullptr) {
    throw logic_error("explained-variance: expected a vector, got none");
  }
  return raftrkt::vector_data<T>(v.variance, "explained-variance", k);
}

template <typename T>
double forward_spread(const raft::handle_t& handle, const T* variance,
                      int64_t k) {
  return variance == nullptr
             ? 1.0
             : 1.0 / bounds_of(handle, variance, k).smallest_divisor;
}

template <typename T>
double inverse_spread(const raft::handle_t& handle, const T* variance,
                      int64_t k) {
  return variance == nullptr ? 1.0
                             : bounds_of(handle, variance, k).largest_root;
}

template <typename T>
struct operands {
  matrix<T, int64_t> x;
  matrix<T, int64_t> components;
};

template <typename T, bool Inverse>
void apply_into(const raft::handle_t& handle, const operands<T>& in,
                const apply_views& v) {
  const matrix<T, int64_t>& x = in.x;
  const matrix<T, int64_t>& c = in.components;
  const int64_t k = c.n;
  const int64_t d = c.d;
  const int64_t width = Inverse ? k : d;
  if (x.d != width) {
    throw logic_error(
        std::format("X: expected {} columns, got {}", width, x.d));
  }
  const T* mean = raftrkt::vector_data<T>(v.mean, "mean", d);
  const T* variance = explained_variance<T>(v, k);
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  require_finite("X", x, stream);
  require_magnitudes(
      "X", x,
      most_value<T> / (overflow_margin * static_cast<double>(width) *
                       (Inverse ? inverse_spread(handle, variance, k)
                                : forward_spread(handle, variance, k))),
      stream);
  const int64_t out_width = Inverse ? d : k;
  auto result =
      std::make_unique<buffer<T>>(scratch<T>(x.n * out_width, stream));
  {
    buffer<T> columns = scratch<T>(x.n * x.d, stream);
    to_columns<T>(handle, x.data, x.n, x.d, Inverse ? variance : nullptr,
                  columns.data());
    buffer<T> components = scratch<T>(k * d, stream);
    to_columns<T>(handle, c.data, k, d, nullptr, components.data());
    buffer<T> projected = scratch<T>(x.n * out_width, stream);
    const ML::paramsPCA p =
        pca_params({.rows = x.n, .cols = d, .components = k});
    if constexpr (Inverse) {
      ML::pcaInverseTransform(handle, columns.data(), components.data(),
                              nullptr, const_cast<T*>(mean), projected.data(),
                              p);
      to_rows<T>(handle, projected.data(), x.n, d, nullptr, result->data());
    } else {
      ML::pcaTransform(handle, columns.data(), components.data(),
                       projected.data(), nullptr, const_cast<T*>(mean), p);
      to_rows<T>(handle, projected.data(), x.n, k, variance, result->data());
    }
    settle(handle);
  }
  const std::initializer_list<rr_view> regions{
      raftrkt::region(result->data(), x.n, out_width, layout::row_major)};
  raftrkt::deposit(v.out, "out", handle, std::move(result), regions);
}

template <typename T>
matrix<T, int64_t> applied_x(const rr_view* view) {
  const matrix<T, int64_t> x =
      rows_of<T, int64_t>(view, "X", any_extent, any_extent);
  require_some("X", "row", x.n);
  require_rows(x.n);
  return x;
}

template <bool Inverse>
int applied(void* handle, const apply_views& v) {
  return guarded(handle,
                 {{"X", v.x},
                  {"components", v.components},
                  {"mean", v.mean},
                  {"explained-variance", v.variance}},
                 [&](const raft::handle_t& h) {
                   raftrkt::require(v.out, "out");
                   on_floats(v.x, "X", [&](auto x_tag) {
                     using X = decltype(x_tag);
                     const matrix<X, int64_t> x = applied_x<X>(v.x);
                     on_floats(v.components, "components", [&](auto c_tag) {
                       using C = decltype(c_tag);
                       const matrix<C, int64_t> c = rows_of<C, int64_t>(
                           v.components, "components", any_extent, any_extent);
                       require_some("components", "row", c.n);
                       require_some("components", "column", c.d, 2);
                       if constexpr (std::is_same_v<X, C>) {
                         apply_into<C, Inverse>(h, operands<C>{x, c}, v);
                       } else {
                         throw logic_error(std::format(
                             "X: expected {}, got {}",
                             raftrkt::dtype_name(raftrkt::dtype_code<C>),
                             raftrkt::dtype_name(raftrkt::dtype_code<X>)));
                       }
                     });
                   });
                 });
}

}  // namespace

extern "C" {

int cuml_decomposition_pca_fit(void* handle, const rr_view* x, rr_deposit* out,
                               int32_t all_components, int64_t n_components,
                               int32_t solver, double tol, int64_t iterations,
                               double* noise_variance, int64_t* n_samples) {
  return guarded(handle, {{"X", x}}, [&](const raft::handle_t& h) {
    const fit_request r{raftrkt::require(out, "out"),
                        all_components != 0,
                        n_components,
                        solver,
                        tol,
                        iterations,
                        raftrkt::require(noise_variance, "noise-variance"),
                        raftrkt::require(n_samples, "n-samples")};
    on_floats(x, "X", [&](auto tag) { fit_into<decltype(tag)>(h, x, r); });
  });
}

int cuml_decomposition_pca_transform(void* handle, const rr_view* x,
                                     const rr_view* components,
                                     const rr_view* mean,
                                     const rr_view* explained_variance,
                                     rr_deposit* out, int32_t whiten) {
  return applied<false>(
      handle, {x, components, mean, explained_variance, out, whiten != 0});
}

int cuml_decomposition_pca_inverse_transform(void* handle, const rr_view* x,
                                             const rr_view* components,
                                             const rr_view* mean,
                                             const rr_view* explained_variance,
                                             rr_deposit* out, int32_t whiten) {
  return applied<true>(
      handle, {x, components, mean, explained_variance, out, whiten != 0});
}

CUMLRKT_JOB(cuml_decomposition_pca_fit)
CUMLRKT_JOB(cuml_decomposition_pca_transform)
CUMLRKT_JOB(cuml_decomposition_pca_inverse_transform)
}
