#include "ritz.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace cumlrkt::glm::ritz {

namespace {

constexpr int gap_bits = 20;
constexpr double float32_roundoff = 0x1p-24;
constexpr double semicircle_edge = 2.0;
constexpr double window_reach = 16.0;
constexpr int64_t gram_share = 4;

double gram_error(int64_t d) {
  return semicircle_edge * std::sqrt(static_cast<double>(gram_block_rows * d)) *
         float32_roundoff;
}

}  // namespace

double dot(const double* a, const double* b, int64_t d) {
  return std::transform_reduce(a, a + d, b, 0.0);
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

std::vector<double> symmetric_block(const std::vector<double>& m, int64_t c,
                                    int64_t k) {
  std::vector<double> out(k * k);
  for (int64_t j = 0; j < k; ++j) {
    for (int64_t i = 0; i < k; ++i) {
      out[(j * k) + i] = 0.5 * (m[(j * c) + i] + m[(i * c) + j]);
    }
  }
  return out;
}

double reach_of(int64_t d) {
  return window_reach * gram_error(d);
}

int64_t window_of(const std::vector<double>& values, int64_t d) {
  const double reach = values.front() + reach_of(d);
  return std::max<int64_t>(1, std::ranges::count_if(values, [reach](double l) {
                             return l <= reach;
                           }));
}

bool exact_gram_cheaper(int64_t window, int64_t d) {
  return gram_share * window >= d;
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
