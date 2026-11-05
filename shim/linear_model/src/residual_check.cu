#include <algorithm>
#include <cmath>
#include <cstdint>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <utility>
#include <vector>

#include "cumlrkt/device.cuh"
#include "cumlrkt/scale.hpp"
#include "cumlrkt/support.hpp"
#include "glm_device.cuh"
#include "glm_staging_device.hpp"
#include "residual_check.hpp"

namespace cumlrkt::glm {

namespace {

constexpr int warp_lanes = 32;
constexpr int residual_threads = 256;
constexpr int64_t most_residual_blocks = int64_t{1} << 20;

template <typename T>
struct centred_column {
  const T* x;
  const int64_t* kept;
  const power_of_two<T>* stage;
  const double* mean;
  const double* remainder;
  int64_t count;
  int64_t d;

  __device__ double operator()(int64_t i, int64_t k) const {
    return (static_cast<double>(scaled_by(x[(i * d) + kept[k]], stage[k])) -
            mean[k]) -
           remainder[k];
  }
};

template <typename T>
struct centred_target {
  staged_target<T> y;
  centre c;

  __device__ double operator()(int64_t i) const {
    return (y(i) - c.mean) - c.remainder;
  }
};

template <typename T>
__global__ void centred_residuals(centred_column<T> c, const T* coef,
                                  centred_target<T> y, row_selection rows,
                                  double* r) {
  const int lane = static_cast<int>(threadIdx.x) % warp_lanes;
  const int64_t thread =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const int64_t warps =
      (static_cast<int64_t>(gridDim.x) * blockDim.x) / warp_lanes;
  for (int64_t p = thread / warp_lanes; p < rows.count; p += warps) {
    const int64_t i = source_row(rows, p);
    double fitted = 0;
    for (int64_t k = lane; k < c.count; k += warp_lanes) {
      fitted += c(i, k) * static_cast<double>(coef[k]);
    }
    for (int s = warp_lanes / 2; s > 0; s /= 2) {
      fitted += __shfl_down_sync(0xffffffffU, fitted, s);
    }
    if (lane == 0) {
      r[i] = y(i) - fitted;
    }
  }
}

struct moment_sums {
  product3 square;
  product3 cross;
};

struct add_moments {
  __device__ moment_sums operator()(moment_sums a, moment_sums b) const {
    const add_scaled add;
    return {add(a.square, b.square), add(a.cross, b.cross)};
  }
};

template <typename T>
struct moment_terms {
  centred_column<T> c;
  centred_target<T> y;
  const T* w;
  const double* r;

  __device__ moment_sums operator()(int64_t i, int64_t k) const {
    if (!counted(w, i)) {
      return {no_product(), no_product()};
    }
    const bool target = k == c.count;
    const double v = target ? y(i) : c(i, k);
    const double wi = weight(w, i);
    if (!moderate(wi) || !moderate(v)) {
      return {product_of(wi, v, v),
              target ? no_product() : product_of(wi, v, r[i])};
    }
    const double wv = w == nullptr ? v : wi * v;
    return {product_of(wv, v), target ? no_product() : product_of(wv, r[i])};
  }
};

scaled_sum sum_of(product3 p) {
  if (p.exponent == no_exponent || p.significand == 0) {
    return {0.0, 0};
  }
  const int e = std::ilogb(p.significand);
  return {std::scalbn(p.significand, -e), p.exponent + e};
}

}  // namespace

template <typename T>
residual_terms residual_moments(const raft::handle_t& handle,
                                const matrix<T, int64_t>& x, const T* y,
                                const T* w, row_selection rows,
                                const staging& s, const T* coef,
                                bool fit_intercept) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const auto count = static_cast<int64_t>(s.kept.size());
  std::vector<power_of_two<T>> stage(count);
  std::vector<double> mean(count, 0.0);
  std::vector<double> remainder(count, 0.0);
  std::vector<double> largest(count);
  for (int64_t k = 0; k < count; ++k) {
    const int64_t j = s.kept[k];
    const int e = s.column_exponent[j];
    stage[k] = power_of_two_of<T>(e);
    largest[k] = std::abs(std::ldexp(static_cast<T>(s.largest[j]), e));
    if (fit_intercept) {
      const int shift = e - unit_exponent(s.largest[j]);
      mean[k] = std::ldexp(s.normalised_mean[j], shift);
      remainder[k] = std::ldexp(s.normalised_remainder[j], shift);
    }
  }
  const buffer<int64_t> kept = to_device(s.kept, stream);
  const buffer<power_of_two<T>> stages = to_device(stage, stream);
  const buffer<double> means = to_device(mean, stream);
  const buffer<double> remainders = to_device(remainder, stream);
  const centred_column<T> c{x.data,       kept.data(),       stages.data(),
                            means.data(), remainders.data(), count,
                            x.d};
  const centred_target<T> target{
      staged_target<T>{y, power_of_two_of<T>(s.y_exponent)},
      fit_intercept ? target_centre(y, w, rows, s.y_exponent, stream)
                    : centre{0.0, 0.0}};
  buffer<double> r = scratch<double>(x.n, stream);
  const int64_t blocks =
      std::clamp<int64_t>((rows.count + (residual_threads / warp_lanes) - 1) /
                              (residual_threads / warp_lanes),
                          1, most_residual_blocks);
  centred_residuals<T>
      <<<static_cast<unsigned>(blocks), residual_threads, 0, stream.value()>>>(
          c, coef, target, rows, r.data());
  raftrkt::cuda_check(cudaGetLastError(), "centred_residuals");
  const std::vector<moment_sums> sums = column_reduce(
      stream, rows, count + 1, moment_terms<T>{c, target, w, r.data()},
      add_moments{}, moment_sums{no_product(), no_product()});
  std::vector<moment> moments(count);
  for (int64_t k = 0; k < count; ++k) {
    moments[k] = {sum_of(sums[k].square), sum_of(sums[k].cross)};
  }
  return {std::move(moments), std::move(largest), sum_of(sums[count].square)};
}

template residual_terms residual_moments<float>(const raft::handle_t&,
                                                const matrix<float, int64_t>&,
                                                const float*, const float*,
                                                row_selection, const staging&,
                                                const float*, bool);
template residual_terms residual_moments<double>(const raft::handle_t&,
                                                 const matrix<double, int64_t>&,
                                                 const double*, const double*,
                                                 row_selection, const staging&,
                                                 const double*, bool);

}  // namespace cumlrkt::glm
