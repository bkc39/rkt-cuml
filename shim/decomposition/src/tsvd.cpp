#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuml/decomposition/params.hpp>
#include <cuml/decomposition/tsvd.hpp>
#include <format>
#include <initializer_list>
#include <memory>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <type_traits>
#include <utility>

#include "cumlrkt/decomposition.h"
#include "cumlrkt/limits.hpp"
#include "cumlrkt/rows.hpp"
#include "cumlrkt/scale.hpp"
#include "cumlrkt/support.hpp"
#include "pca_device.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"
#include "tsvd_device.hpp"

namespace {

using cumlrkt::buffer;
using cumlrkt::guarded;
using cumlrkt::largest_magnitude;
using cumlrkt::logic_error;
using cumlrkt::matrix;
using cumlrkt::most_columns;
using cumlrkt::most_index;
using cumlrkt::most_value;
using cumlrkt::on_floats;
using cumlrkt::overflow_margin;
using cumlrkt::printed;
using cumlrkt::require_at_least;
using cumlrkt::require_at_most;
using cumlrkt::require_finite;
using cumlrkt::require_finite_non_negative;
using cumlrkt::require_float;
using cumlrkt::require_magnitudes;
using cumlrkt::require_rows;
using cumlrkt::require_some;
using cumlrkt::rows_of;
using cumlrkt::scale_exponent;
using cumlrkt::scratch;
using cumlrkt::settle;
using cumlrkt::squared_norm_bound;
using cumlrkt::decomposition::to_columns;
using cumlrkt::decomposition::to_rows;
using cumlrkt::decomposition::tsvd_arrays;
using cumlrkt::decomposition::tsvd_extents;
using raftrkt::any_extent;
using raftrkt::layout;

void require_columns(int64_t d) {
  if (d > most_columns) {
    throw logic_error(std::format(
        "X: {} columns: a {}x{} Gram matrix does not fit the index type", d, d,
        d));
  }
}

struct tsvd_shape {
  int64_t rows;
  int64_t cols;
  int64_t components;
};

struct fit_request {
  rr_deposit* out;
  int64_t n_components;
  int32_t solver;
  double tol;
  int64_t iterations;
  bool with_transform;
};

void require_parameters(const fit_request& r, int64_t m) {
  require_at_least("n-components", r.n_components, 1);
  require_at_most("n-components", r.n_components, m);
  if (r.solver != CUML_DECOMPOSITION_SOLVER_FULL &&
      r.solver != CUML_DECOMPOSITION_SOLVER_JACOBI) {
    throw logic_error("solver: expected full or jacobi");
  }
  require_finite_non_negative("tol", r.tol);
  require_float("tol", r.tol);
  require_at_least("iterations", r.iterations, 1);
  require_at_most("iterations", r.iterations, most_index);
}

template <typename T>
void require_frobenius(const raft::handle_t& handle,
                       const matrix<T, int64_t>& x) {
  const double most = squared_norm_bound<T>;
  const double got =
      cumlrkt::decomposition::squared_frobenius(handle, x.data, x.n * x.d);
  if (!(got <= most)) {
    throw logic_error(std::format(
        "X: expected a squared Frobenius norm of at most {}, got {}",
        printed(most), printed(got)));
  }
}

template <typename T>
struct fit_outputs {
  fit_outputs(const tsvd_shape& shape, rmm::cuda_stream_view stream)
      : components(scratch<T>(shape.components * shape.cols, stream)),
        singular(scratch<T>(shape.components, stream)),
        variance(scratch<T>(shape.components, stream)),
        ratio(scratch<T>(shape.components, stream)),
        transform(scratch<T>(shape.rows * shape.components, stream)) {}
  buffer<T> components;
  buffer<T> singular;
  buffer<T> variance;
  buffer<T> ratio;
  buffer<T> transform;
};

ML::paramsTSVD tsvd_params(const tsvd_shape& shape) {
  ML::paramsTSVD p;
  p.n_rows = static_cast<std::size_t>(shape.rows);
  p.n_cols = static_cast<std::size_t>(shape.cols);
  p.n_components = static_cast<std::size_t>(shape.components);
  return p;
}

template <typename T>
struct operands {
  matrix<T, int64_t> x;
  matrix<T, int64_t> c;
};

template <typename T>
struct apply_scratch {
  T* columns;
  T* components;
  T* applied;
};

template <typename T, bool Inverse>
void apply_rows(const raft::handle_t& handle, const operands<T>& o,
                const apply_scratch<T>& s, T* out) {
  const auto& [x, c] = o;
  to_columns<T>(handle, x.data, x.n, x.d, nullptr, s.columns, 0);
  to_columns<T>(handle, c.data, c.n, c.d, nullptr, s.components, 0);
  const ML::paramsTSVD p = tsvd_params(tsvd_shape{x.n, c.d, c.n});
  if constexpr (Inverse) {
    ML::tsvdInverseTransform(handle, s.columns, s.components, s.applied, p);
  } else {
    ML::tsvdTransform(handle, s.columns, s.components, s.applied, p);
  }
  to_rows<T>(handle, s.applied, x.n, Inverse ? c.d : c.n, nullptr, out);
}

template <typename T>
void fit_into(const raft::handle_t& handle, const rr_view* view,
              const fit_request& r) {
  const matrix<T, int64_t> x =
      rows_of<T, int64_t>(view, "X", any_extent, any_extent);
  require_some("X", "row", x.n, 2);
  require_some("X", "column", x.d, 2);
  require_rows(x.n);
  require_columns(x.d);
  require_parameters(r, std::min(x.n, x.d));
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  require_finite("X", x, stream);
  require_frobenius(handle, x);
  const int exponent =
      scale_exponent<T>(largest_magnitude(x.data, x.n * x.d, stream));
  const bool constant =
      cumlrkt::first_row_unlike_first(x.data, x.n, x.d, stream) < 0;
  const int64_t k = r.n_components;
  auto outputs = std::make_unique<fit_outputs<T>>(
      tsvd_shape{r.with_transform ? x.n : 0, x.d, k}, stream);
  {
    buffer<T> columns = scratch<T>(x.n * x.d, stream);
    to_columns<T>(handle, x.data, x.n, x.d, nullptr, columns.data(), exponent);
    buffer<T> components = scratch<T>(k * x.d, stream);
    buffer<T> singular = scratch<T>(k, stream);
    buffer<T> variance = scratch<T>(k, stream);
    buffer<T> ratio = scratch<T>(k, stream);
    buffer<T> transform = scratch<T>(x.n * k, stream);
    buffer<bool> flipped = scratch<bool>(k, stream);
    ML::paramsTSVD p = tsvd_params(tsvd_shape{x.n, x.d, k});
    p.algorithm = r.solver == CUML_DECOMPOSITION_SOLVER_JACOBI
                      ? ML::solver::COV_EIG_JACOBI
                      : ML::solver::COV_EIG_DQ;
    p.tol = static_cast<float>(r.tol);
    p.n_iterations = static_cast<uint32_t>(r.iterations);
    ML::tsvdFitTransform(handle, columns.data(), transform.data(),
                         components.data(), variance.data(), ratio.data(),
                         singular.data(), p);
    cumlrkt::decomposition::tsvd_post_fit<T>(
        handle,
        tsvd_arrays<T>{components.data(), singular.data(), variance.data(),
                       ratio.data(), transform.data()},
        tsvd_extents{x.n, x.d, k, constant, exponent}, flipped.data(),
        tsvd_arrays<T>{outputs->components.data(), outputs->singular.data(),
                       outputs->variance.data(), outputs->ratio.data(),
                       r.with_transform && exponent == 0
                           ? outputs->transform.data()
                           : nullptr});
    if (r.with_transform && exponent != 0) {
      apply_rows<T, false>(
          handle, operands<T>{x, {outputs->components.data(), k, x.d}},
          apply_scratch<T>{columns.data(), components.data(), transform.data()},
          outputs->transform.data());
    }
    settle(handle);
  }
  rr_view components_region =
      raftrkt::region(outputs->components.data(), k, x.d, layout::row_major);
  rr_view singular_region = raftrkt::region(outputs->singular);
  rr_view variance_region = raftrkt::region(outputs->variance);
  rr_view ratio_region = raftrkt::region(outputs->ratio);
  if (r.with_transform) {
    rr_view transform_region =
        raftrkt::region(outputs->transform.data(), x.n, k, layout::row_major);
    raftrkt::deposit(r.out, "out", handle, std::move(outputs),
                     {components_region, singular_region, variance_region,
                      ratio_region, transform_region});
  } else {
    raftrkt::deposit(
        r.out, "out", handle, std::move(outputs),
        {components_region, singular_region, variance_region, ratio_region});
  }
}

template <typename T, bool Inverse>
void apply_into(const raft::handle_t& handle, const operands<T>& o,
                rr_deposit* out) {
  const auto& [x, c] = o;
  const int64_t k = c.n;
  const int64_t d = c.d;
  const int64_t width = Inverse ? k : d;
  if (x.d != width) {
    throw logic_error(
        std::format("X: expected {} columns, got {}", width, x.d));
  }
  const rmm::cuda_stream_view stream = raft::resource::get_cuda_stream(handle);
  require_finite("X", x, stream);
  require_magnitudes(
      "X", x, most_value<T> / (overflow_margin * static_cast<double>(width)),
      stream);
  const int64_t out_width = Inverse ? d : k;
  auto result =
      std::make_unique<buffer<T>>(scratch<T>(x.n * out_width, stream));
  {
    buffer<T> columns = scratch<T>(x.n * x.d, stream);
    buffer<T> components = scratch<T>(k * d, stream);
    buffer<T> applied = scratch<T>(x.n * out_width, stream);
    apply_rows<T, Inverse>(
        handle, o,
        apply_scratch<T>{columns.data(), components.data(), applied.data()},
        result->data());
    settle(handle);
  }
  const std::initializer_list<rr_view> regions{
      raftrkt::region(result->data(), x.n, out_width, layout::row_major)};
  raftrkt::deposit(out, "out", handle, std::move(result), regions);
}

template <bool Inverse>
int applied(void* handle, const rr_view* x_view, const rr_view* c_view,
            rr_deposit* out) {
  return guarded(handle, {{"X", x_view}, {"components", c_view}},
                 [&](const raft::handle_t& h) {
                   raftrkt::require(out, "out");
                   on_floats(x_view, "X", [&](auto x_tag) {
                     using X = decltype(x_tag);
                     const matrix<X, int64_t> x = rows_of<X, int64_t>(
                         x_view, "X", any_extent, any_extent);
                     require_some("X", "row", x.n);
                     require_rows(x.n);
                     on_floats(c_view, "components", [&](auto c_tag) {
                       using C = decltype(c_tag);
                       const matrix<C, int64_t> c = rows_of<C, int64_t>(
                           c_view, "components", any_extent, any_extent);
                       require_some("components", "row", c.n);
                       require_some("components", "column", c.d, 2);
                       if constexpr (std::is_same_v<X, C>) {
                         apply_into<C, Inverse>(h, operands<C>{x, c}, out);
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

int cuml_decomposition_tsvd_fit(void* handle, const rr_view* x, rr_deposit* out,
                                int64_t n_components, int32_t solver,
                                double tol, int64_t iterations,
                                int32_t with_transform) {
  return guarded(handle, {{"X", x}}, [&](const raft::handle_t& h) {
    const fit_request r{
        raftrkt::require(out, "out"), n_components, solver, tol, iterations,
        with_transform != 0};
    on_floats(x, "X", [&](auto tag) { fit_into<decltype(tag)>(h, x, r); });
  });
}

int cuml_decomposition_tsvd_transform(void* handle, const rr_view* x,
                                      const rr_view* components,
                                      rr_deposit* out) {
  return applied<false>(handle, x, components, out);
}

int cuml_decomposition_tsvd_inverse_transform(void* handle, const rr_view* x,
                                              const rr_view* components,
                                              rr_deposit* out) {
  return applied<true>(handle, x, components, out);
}

CUMLRKT_JOB(cuml_decomposition_tsvd_fit)
CUMLRKT_JOB(cuml_decomposition_tsvd_transform)
CUMLRKT_JOB(cuml_decomposition_tsvd_inverse_transform)
}
