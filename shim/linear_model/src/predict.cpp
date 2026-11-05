#include <cmath>
#include <cstdint>
#include <cuml/linear_model/glm.hpp>
#include <format>
#include <memory>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <type_traits>
#include <utility>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/rows.hpp"
#include "cumlrkt/support.hpp"
#include "cumlrkt/transpose.hpp"
#include "cumlrkt/vectors.hpp"
#include "raftrkt/deposit.hpp"
#include "raftrkt/job.hpp"
#include "raftrkt/view.hpp"

namespace {

using cumlrkt::buffer;
using cumlrkt::logic_error;
using cumlrkt::matrix;
using cumlrkt::printed;
using cumlrkt::scratch;

template <typename T>
struct line {
  const T* coef;
  int64_t d;
  double intercept;
};

template <typename T>
line<T> line_of(const rr_view* coef, double intercept) {
  const T* c = raftrkt::vector_data<T, int32_t>(coef, "coef");
  const int64_t d = coef->shape[0];
  cumlrkt::require_some("coef", "element", d);
  return {c, d, intercept};
}

template <typename T>
void predict_into(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                  const line<T>& fitted, rr_deposit* out) {
  cumlrkt::require_some("X", "row", x.n);
  if (x.d != fitted.d) {
    throw logic_error(
        std::format("X: expected {} columns, got {}", fitted.d, x.d));
  }
  cumlrkt::require_index("X", x.n, x.d);
  const rmm::cuda_stream_view stream = cumlrkt::stream_of(handle);
  cumlrkt::require_finite("X", x, stream);
  cumlrkt::require_finite_elements("coef", fitted.coef, fitted.d, stream);
  if (!std::isfinite(fitted.intercept)) {
    throw logic_error(std::format("intercept: expected a finite value, got {}",
                                  printed(fitted.intercept)));
  }
  auto predictions = std::make_unique<buffer<T>>(scratch<T>(x.n, stream));
  {
    buffer<T> columns = scratch<T>(x.n * x.d, stream);
    cumlrkt::to_columns<T>(handle, x.data, x.n, x.d, nullptr, columns.data(),
                           0);
    ML::GLM::gemmPredict(handle, columns.data(), static_cast<size_t>(x.n),
                         static_cast<size_t>(x.d), fitted.coef,
                         static_cast<T>(fitted.intercept), predictions->data());
    const int64_t i =
        cumlrkt::first_non_finite(predictions->data(), x.n, stream);
    if (i >= 0) {
      throw logic_error(
          std::format("X: row {}: the prediction exceeds the {} range", i,
                      raftrkt::dtype_name(raftrkt::dtype_code<T>)));
    }
    cumlrkt::settle(handle);
  }
  const rr_view region = raftrkt::region(*predictions);
  raftrkt::deposit(out, "out", handle, std::move(predictions), {region});
}

}  // namespace

extern "C" {

int cuml_linear_model_gemm_predict(void* handle, const rr_view* x,
                                   const rr_view* coef, double intercept,
                                   rr_deposit* out) {
  return cumlrkt::guarded(
      handle, {{"X", x}, {"coef", coef}}, [&](const raft::handle_t& h) {
        rr_deposit* const into = raftrkt::require(out, "out");
        cumlrkt::on_floats(x, "X", [&](auto x_tag) {
          using X = decltype(x_tag);
          const matrix<X, int64_t> m = cumlrkt::rows_of<X, int64_t>(
              x, "X", raftrkt::any_extent, raftrkt::any_extent);
          cumlrkt::on_floats(coef, "coef", [&](auto c_tag) {
            using C = decltype(c_tag);
            const line<C> fitted = line_of<C>(coef, intercept);
            if constexpr (std::is_same_v<X, C>) {
              predict_into<C>(h, m, fitted, into);
            } else {
              throw logic_error(
                  std::format("X: expected {}, got {}",
                              raftrkt::dtype_name(raftrkt::dtype_code<C>),
                              raftrkt::dtype_name(raftrkt::dtype_code<X>)));
            }
          });
        });
      });
}

CUMLRKT_JOB(cuml_linear_model_gemm_predict)
}
