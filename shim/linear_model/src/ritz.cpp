#include "ritz.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

namespace cumlrkt::glm::ritz {

namespace {

constexpr int gap_bits = 20;
constexpr double float32_roundoff = 0x1p-24;
constexpr double float64_roundoff = 0x1p-53;
constexpr int64_t wide_run_ratio = 2;
constexpr int64_t wide_run_rows_min = 64;
constexpr int64_t square_roundings = 3;
constexpr int64_t staged_roundings = 5;
constexpr int64_t host_roundings = 32;
constexpr double expected_residual_factor = 0.5;
constexpr double eigen_error_factor = 16.0;
constexpr double window_reach = 16.0;
constexpr double edge_gap = 2.0;
constexpr double certified_margin = 2.0;
constexpr double pass_ns = 0.0208;
constexpr double vector_square_ns = 0.077;
constexpr double vector_ns = 98000.0;
constexpr double row_ns = 305.0;
constexpr double syrk_ns = 0.0022;
constexpr double eigen_cube_ns = 0.021;
constexpr double enclosure_cube_ns = 0.0085;
constexpr double enclosure_square_ns = 11.3;
constexpr double enclosure_ns = 126000.0;
constexpr double enclosure_steps = 16.0;
constexpr int outward_ulps = 4;
constexpr double half = 0.5;
constexpr double infinite = std::numeric_limits<double>::infinity();

double root(int64_t k) {
  return std::sqrt(static_cast<double>(k));
}

double rounding(double k) {
  const double ku = k * float64_roundoff;
  return ku / (1.0 - ku);
}

double single_rounding(double k) {
  const double ku = k * float32_roundoff;
  return ku / (1.0 - ku);
}

double compounded(double a, double b) {
  return a + b + (a * b);
}

double outward(double x, double toward) {
  for (int i = 0; i < outward_ulps; ++i) {
    x = std::nextafter(x, toward);
  }
  return x;
}

double exact_ns(gram_shape shape) {
  const auto rows = static_cast<double>(shape.rows);
  const auto cols = static_cast<double>(shape.columns);
  return (row_ns * rows) + (syrk_ns * rows * cols * cols) +
         (eigen_cube_ns * cols * cols * cols);
}

double pass_cost_ns(int64_t vectors, gram_shape shape) {
  const auto rows = static_cast<double>(shape.rows);
  const auto cols = static_cast<double>(shape.columns);
  return static_cast<double>(vectors) *
         ((pass_ns * rows * cols) + (vector_square_ns * cols * cols) +
          vector_ns);
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

int64_t gram_run_rows(int64_t d) {
  if (d <= narrow_gram_columns) {
    return narrow_run_rows;
  }
  const auto columns =
      static_cast<int64_t>(std::bit_ceil(static_cast<uint64_t>(d)));
  return std::clamp<int64_t>(wide_run_ratio * columns, wide_run_rows_min,
                             gram_block_rows);
}

gram_bound bound_of(const std::vector<double>& spread,
                    const std::vector<double>& scale, double alpha, int64_t n) {
  const auto d = static_cast<int64_t>(spread.size());
  const int64_t m = gram_run_rows(d);
  const double summed =
      1.0 - rounding(static_cast<double>(n + square_roundings));
  double trace = 0.0;
  double diagonal = 0.0;
  for (int64_t k = 0; k < d; ++k) {
    const double squares = scale[k] * scale[k] * spread[k] / summed;
    trace += squares;
    diagonal = std::max(diagonal, squares + (alpha * scale[k] * scale[k]));
  }
  const double host = 1.0 + rounding(static_cast<double>(d + host_roundings));
  return {.runs = ((n + m - 1) / m) + 1,
          .run_rows = m,
          .trace = trace * host,
          .diagonal = diagonal * host};
}

double gram_error(gram_bound b) {
  const double staged = compounded(
      float32_roundoff, rounding(static_cast<double>(staged_roundings)));
  const double runs =
      compounded(single_rounding(static_cast<double>(b.run_rows)),
                 rounding(static_cast<double>(b.runs)));
  const double entry = compounded(compounded(runs, compounded(staged, staged)),
                                  float32_roundoff);
  const double ridge =
      compounded(compounded(float32_roundoff, float32_roundoff), rounding(2.0));
  const double error = (entry * b.trace) + (ridge * (1.0 + entry) * b.diagonal);
  return error * (1.0 + rounding(static_cast<double>(host_roundings)));
}

double spectrum_error(const std::vector<double>& values, double gram) {
  const double top =
      std::max(std::abs(values.front()), std::abs(values.back()));
  const auto d = static_cast<int64_t>(values.size());
  return gram + (eigen_error_factor * root(d) * float32_roundoff * top);
}

double expected_residual(const std::vector<double>& values) {
  const double top =
      std::max(std::abs(values.front()), std::abs(values.back()));
  return expected_residual_factor * static_cast<double>(values.size()) *
         float32_roundoff * top;
}

std::optional<double> certified_kappa(spectrum_interval s, double limit) {
  if (!(s.low > 0)) {
    return std::nullopt;
  }
  const double upper = s.high / s.low;
  if (!(certified_margin * upper < limit)) {
    return std::nullopt;
  }
  return upper;
}

std::optional<spectrum_interval> enclosure(const std::vector<double>& values,
                                           enclosure_norms r, int64_t d) {
  const auto k = static_cast<double>(d);
  const double norms = 1.0 + rounding((k * k) + 2.0);
  const double products = rounding(k);
  const double basis = r.basis * norms;
  const double eta = (r.orthogonality * norms * (1.0 + rounding(1.0))) +
                     (products * basis * basis);
  if (!(eta < 1.0)) {
    return std::nullopt;
  }
  const double symmetric =
      (r.symmetric * norms * (1.0 + rounding(1.0))) +
      (products * basis * ((basis * r.matrix * norms) + (r.product * norms)));
  const double error = symmetric * (1.0 + rounding(enclosure_steps));

  const double low = values.front() - error;
  const double high = values.back() + error;
  return spectrum_interval{
      outward(low / (low > 0 ? 1.0 + eta : 1.0 - eta), -infinite),
      outward(high / (high > 0 ? 1.0 - eta : 1.0 + eta), infinite)};
}

bool exact_gram_cheaper(int64_t vectors, gram_shape shape) {
  return pass_cost_ns(vectors, shape) >= exact_ns(shape);
}

bool enclosure_pays(const std::vector<double>& values, const gram_plan& plan,
                    gram_shape shape, double limit, double gram) {
  const double reach = gram + expected_residual(values);

  if (!certified_kappa({values.front() - reach, values.back() + reach},
                       limit)) {
    return false;
  }
  if (plan.path == gram_path::certified) {
    return true;
  }
  const auto cols = static_cast<double>(shape.columns);
  const double cost = (enclosure_cube_ns * cols * cols * cols) +
                      (enclosure_square_ns * cols * cols) + enclosure_ns;
  const double saved = plan.path == gram_path::ritz
                           ? pass_cost_ns(plan.window + 1, shape)
                           : exact_ns(shape);
  return cost < saved;
}

gram_plan plan_of(const std::vector<double>& values, gram_shape shape,
                  double limit, double gram) {
  const int64_t d = shape.columns;
  const double error = spectrum_error(values, gram);
  const double low = values.front();
  const double top = values.back();
  const double reach = window_reach * error;
  if (const auto upper = certified_kappa({low - error, top + error}, limit)) {
    return {gram_path::certified, 0, reach, *upper};
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
