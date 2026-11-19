#include "ritz.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace cumlrkt::glm::ritz {

namespace {

constexpr int gap_bits = 20;

}  // namespace

double dot(const double* a, const double* b, int64_t d) {
  return std::transform_reduce(a, a + d, b, 0.0);
}

std::vector<double> orthonormal(const std::vector<double>& vectors, int64_t d,
                                int64_t k) {
  std::vector<double> v(vectors.begin(), vectors.begin() + (k * d));
  for (int64_t j = 0; j < k; ++j) {
    double* vj = v.data() + (j * d);
    for (int pass = 0; pass < 2; ++pass) {
      for (int64_t i = 0; i < j; ++i) {
        const double* vi = v.data() + (i * d);
        const double c = dot(vi, vj, d);
        std::transform(vj, vj + d, vi, vj,
                       [c](double a, double b) { return a - (c * b); });
      }
    }
    const double norm = std::sqrt(dot(vj, vj, d));
    std::transform(vj, vj + d, vj, [norm](double a) { return a / norm; });
  }
  return v;
}

std::vector<double> combined(const std::vector<double>& columns, int64_t d,
                             const double* c, int64_t k) {
  std::vector<double> y(d, 0.0);
  for (int64_t j = 0; j < k; ++j) {
    for (int64_t i = 0; i < d; ++i) {
      y[i] += c[j] * columns[(j * d) + i];
    }
  }
  return y;
}

std::vector<double> refined(const eigen_pairs& e, int64_t window,
                            const ritz_pair& p, int64_t d) {
  const double gap = std::ldexp(e.values.back(), -gap_bits);
  std::vector<double> v = p.vector;
  for (int64_t j = window; j < d; ++j) {
    const double apart = e.values[j] - p.value;
    if (!(apart > gap)) {
      continue;
    }
    const double* vj = e.vectors.data() + (j * d);
    double c = 0.0;
    for (int64_t k = 0; k < d; ++k) {
      c += vj[k] * (p.applied[k] - (p.value * p.vector[k]));
    }
    for (int64_t k = 0; k < d; ++k) {
      v[k] -= vj[k] * c / apart;
    }
  }
  const double norm = std::sqrt(dot(v.data(), v.data(), d));
  std::ranges::transform(v, v.begin(), [norm](double x) { return x / norm; });
  return v;
}

}  // namespace cumlrkt::glm::ritz
