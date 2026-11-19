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
constexpr double eigen_error_factor = 16.0;
constexpr double window_reach = 16.0;
constexpr double edge_gap = 2.0;
constexpr double certified_margin = 2.0;
constexpr double pass_ns = 0.022;
constexpr double row_ns = 300.0;
constexpr double syrk_ns = 0.0035;
constexpr double eigen_square_ns = 20.0;
constexpr double eigen_column_ns = 32000.0;
constexpr double half = 0.5;

double root(int64_t k) {
  return std::sqrt(static_cast<double>(k));
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
      out[(j * k) + i] = half * (m[(j * c) + i] + m[(i * c) + j]);
    }
  }
  return out;
}

double spectrum_error(const std::vector<double>& values, int64_t d) {
  const double top =
      std::max(std::abs(values.front()), std::abs(values.back()));
  return (semicircle_edge * root(gram_block_rows * d) * float32_roundoff) +
         (eigen_error_factor * root(d) * float32_roundoff * top);
}

bool exact_gram_cheaper(int64_t vectors, gram_shape shape) {
  const auto rows = static_cast<double>(shape.rows);
  const auto cols = static_cast<double>(shape.columns);
  const double gram = (row_ns * rows) + (syrk_ns * rows * cols * cols);
  const double eigen =
      (eigen_square_ns * cols * cols) + (eigen_column_ns * cols);
  return pass_ns * static_cast<double>(vectors) * rows * cols >= gram + eigen;
}

gram_plan plan_of(const std::vector<double>& values, gram_shape shape,
                  double limit) {
  const int64_t d = shape.columns;
  const double error = spectrum_error(values, d);
  const double low = values.front();
  const double top = values.back();
  const double reach = window_reach * error;
  if (low > error) {
    const double upper = (top + error) / (low - error);
    if (certified_margin * upper < limit) {
      return {gram_path::certified, 0, reach, upper};
    }
  }
  const double edge = low + reach;
  const int64_t window = std::max<int64_t>(
      1, std::ranges::count_if(values, [edge](double l) { return l <= edge; }));
  const bool gap =
      window < d && low >= -error && values[window] >= edge_gap * edge;
  const bool exact = !gap || exact_gram_cheaper(window + 1, shape);
  return {exact ? gram_path::exact : gram_path::ritz, window, reach, 0.0};
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
