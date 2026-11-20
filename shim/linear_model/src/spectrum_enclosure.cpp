#include "spectrum_enclosure.hpp"

#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <raft/core/cublas_macros.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <utility>
#include <vector>

#include "cumlrkt/support.hpp"
#include "glm_staging_device.hpp"
#include "ritz.hpp"
#include "symmetric_eigen.hpp"

namespace cumlrkt::glm {

namespace {

constexpr double mirrored = 2.0;
constexpr int64_t host_enclosure_columns = 16;
constexpr double float64_roundoff = 0x1p-53;
constexpr double host_steps = 16.0;
constexpr double infinite = std::numeric_limits<double>::infinity();

double rounded(double k) {
  const double ku = k * float64_roundoff;
  return ku / (1.0 - ku);
}

void blas(cublasStatus_t result) {
  RAFT_CUBLAS_TRY(result);
}

struct lower_symmetric {
  std::vector<double> matrix;
  double squares;
};

template <typename S>
lower_symmetric promoted(const std::vector<S>& m, int64_t d) {
  lower_symmetric a{std::vector<double>(d * d), 0.0};
  for (int64_t j = 0; j < d; ++j) {
    for (int64_t i = j; i < d; ++i) {
      const auto v = static_cast<double>(m[(j * d) + i]);
      a.matrix[(j * d) + i] = v;
      a.squares += (i == j ? 1.0 : mirrored) * v * v;
    }
  }
  return a;
}

double frobenius(const std::vector<double>& x) {
  double s = 0.0;
  for (const double v : x) {
    s += v * v;
  }
  return std::sqrt(s);
}

std::vector<double> product(const std::vector<double>& a,
                            const std::vector<double>& b, int64_t d,
                            bool transposed) {
  std::vector<double> c(d * d, 0.0);
  for (int64_t j = 0; j < d; ++j) {
    for (int64_t i = 0; i < d; ++i) {
      double s = 0.0;
      for (int64_t k = 0; k < d; ++k) {
        s += (transposed ? a[(i * d) + k] : a[(k * d) + i]) * b[(j * d) + k];
      }
      c[(j * d) + i] = s;
    }
  }
  return c;
}

template <typename S>
ritz::enclosure_norms host_norms(const std::vector<S>& m,
                                 const ritz::eigen_pairs& e, int64_t d) {
  const lower_symmetric a = promoted(m, d);
  std::vector<double> g = a.matrix;
  for (int64_t j = 0; j < d; ++j) {
    for (int64_t i = j + 1; i < d; ++i) {
      g[(i * d) + j] = g[(j * d) + i];
    }
  }
  const std::vector<double> p = product(g, e.vectors, d, false);
  std::vector<double> h = product(e.vectors, p, d, true);
  std::vector<double> w = product(e.vectors, e.vectors, d, true);
  for (int64_t k = 0; k < d; ++k) {
    h[(k * d) + k] -= e.values[k];
    w[(k * d) + k] -= 1.0;
  }
  return {.symmetric = frobenius(h),
          .orthogonality = frobenius(w),
          .matrix = std::sqrt(a.squares),
          .basis = frobenius(e.vectors),
          .product = frobenius(p)};
}

double squares(cublasHandle_t cublas, const buffer<double>& x) {
  double s = 0.0;
  blas(cublasDdot(cublas, static_cast<int>(x.size()), x.data(), 1, x.data(), 1,
                  &s));
  return s;
}

double residual_squares(cublasHandle_t cublas, const std::vector<double>& v,
                        const std::vector<double>& av,
                        const std::vector<double>& h, int64_t d, int64_t k,
                        rmm::cuda_stream_view stream) {
  const buffer<double> dv = to_device(v, stream);
  const buffer<double> dh = to_device(h, stream);
  buffer<double> r = to_device(av, stream);
  const double one = 1.0;
  const double minus = -1.0;
  const auto [rows, cols] = std::pair{static_cast<int>(d), static_cast<int>(k)};
  blas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols, cols, &minus,
                   dv.data(), rows, dh.data(), cols, &one, r.data(), rows));
  return squares(cublas, r);
}

double windowed(double low, double eta) {
  return low >= 0 ? low / (1.0 + eta) : low / (1.0 - eta);
}

}  // namespace

template <typename S>
ritz::enclosure_norms enclosure_norms_of(const raft::handle_t& handle,
                                         const std::vector<S>& m,
                                         const ritz::eigen_pairs& e,
                                         int64_t d) {
  if (d <= host_enclosure_columns) {
    return host_norms(m, e, d);
  }
  const rmm::cuda_stream_view stream = stream_of(handle);
  cublasHandle_t cublas = raft::resource::get_cublas_handle(handle);
  blas(cublasSetStream(cublas, stream.value()));
  blas(cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST));
  const lower_symmetric a = promoted(m, d);
  const buffer<double> g = to_device(a.matrix, stream);
  const buffer<double> v = to_device(e.vectors, stream);
  const buffer<double> mu = to_device(e.values, stream);
  const buffer<double> ones = to_device(std::vector<double>(d, 1.0), stream);
  buffer<double> p = scratch<double>(d * d, stream);
  buffer<double> h = scratch<double>(d * d, stream);
  buffer<double> w = scratch<double>(d * d, stream);
  const auto k = static_cast<int>(d);
  const double one = 1.0;
  const double zero = 0.0;
  const double minus = -1.0;
  blas(cublasDsymm(cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, k, k, &one,
                   g.data(), k, v.data(), k, &zero, p.data(), k));
  blas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, k, k, k, &one, v.data(), k,
                   p.data(), k, &zero, h.data(), k));
  blas(cublasDaxpy(cublas, k, &minus, mu.data(), 1, h.data(), k + 1));
  blas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, k, k, k, &one, v.data(), k,
                   v.data(), k, &zero, w.data(), k));
  blas(cublasDaxpy(cublas, k, &minus, ones.data(), 1, w.data(), k + 1));
  return {.symmetric = std::sqrt(squares(cublas, h)),
          .orthogonality = std::sqrt(squares(cublas, w)),
          .matrix = std::sqrt(a.squares),
          .basis = std::sqrt(squares(cublas, v)),
          .product = std::sqrt(squares(cublas, p))};
}

template ritz::enclosure_norms enclosure_norms_of<float>(
    const raft::handle_t&, const std::vector<float>&, const ritz::eigen_pairs&,
    int64_t);
template ritz::enclosure_norms enclosure_norms_of<double>(
    const raft::handle_t&, const std::vector<double>&, const ritz::eigen_pairs&,
    int64_t);

std::optional<ritz::spectrum_interval> window_spectrum(
    const raft::handle_t& handle, const std::vector<float>& m,
    const ritz::eigen_pairs& e, window_inputs w, window_errors r) {
  const auto d = static_cast<int64_t>(e.values.size());
  const int64_t k = w.window;
  const ritz::enclosure_norms gram = enclosure_norms_of(handle, m, e, d);
  const std::optional<ritz::enclosure_terms> t = ritz::terms_of(gram, d);
  const std::optional<ritz::spectrum_interval> s =
      ritz::enclosure(e.values, gram, d);
  const std::vector<double> h = ritz::symmetric_block(w.crossed, w.vectors, k);
  const std::optional<ritz::eigen_pairs> he = eigen_of(handle, h, k, true);
  if (!t || !s || !he) {
    return std::nullopt;
  }
  const std::optional<ritz::spectrum_interval> hs =
      ritz::enclosure(he->values, enclosure_norms_of(handle, h, *he, k), k);
  if (!hs) {
    return std::nullopt;
  }
  const rmm::cuda_stream_view stream = stream_of(handle);
  cublasHandle_t cublas = raft::resource::get_cublas_handle(handle);
  blas(cublasSetStream(cublas, stream.value()));
  blas(cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST));
  const std::vector<double> v(w.basis.begin(), w.basis.begin() + (k * d));
  const std::vector<double> av(w.applied.begin(), w.applied.begin() + (k * d));
  const double grown = (1.0 + rounded(static_cast<double>((d * k) + 2))) *
                       (1.0 + rounded(host_steps));
  const double vf = frobenius(v) * grown;
  const double af = frobenius(av) * grown;
  const double hf = frobenius(h) * grown;
  const double drift = r.applied * vf * grown;
  const double eta = t->eta;
  const double shift =
      ((std::sqrt(1.0 + eta) * drift) +
       (rounded(static_cast<double>(d)) * vf * af) + (rounded(1.0) * hf)) *
      grown;
  const double floor = ritz::outward(
      ritz::outward(windowed(hs->low - shift, eta), -infinite) - r.mean,
      -infinite);
  const double residual =
      std::sqrt(residual_squares(cublas, v, av, h, d, k, stream)) * grown;
  const double coupling = ritz::outward(
      (((residual + (rounded(static_cast<double>(k + 1)) * (af + (vf * hf)))) +
        drift) /
       std::sqrt(1.0 - eta) * grown) +
          r.mean,
      infinite);
  const ritz::spectrum_interval outer =
      ritz::widened({ritz::complement_floor(e.values, k, *t), s->high}, r.gram);
  return ritz::spectrum_interval{
      ritz::block_floor({floor, outer.low, coupling}), outer.high};
}

std::optional<ritz::spectrum_interval> enclosed_spectrum(
    const raft::handle_t& handle, const std::vector<float>& m,
    const ritz::eigen_pairs& e, int64_t d) {
  return ritz::enclosure(e.values, enclosure_norms_of(handle, m, e, d), d);
}

std::optional<double> enclosed_kappa(const raft::handle_t& handle,
                                     const std::vector<float>& m,
                                     const ritz::eigen_pairs& e,
                                     const ritz::gram_plan& plan,
                                     ritz::gram_shape shape, double limit,
                                     double gram) {
  const int64_t d = shape.columns;
  if (!ritz::enclosure_pays(e.values, gram, plan, shape, limit)) {
    return std::nullopt;
  }
  const std::optional<ritz::spectrum_interval> s =
      enclosed_spectrum(handle, m, e, d);
  if (!s) {
    return std::nullopt;
  }
  return ritz::certified_kappa(ritz::widened(*s, gram), limit);
}

}  // namespace cumlrkt::glm
