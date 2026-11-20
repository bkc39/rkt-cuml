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
constexpr int64_t mean_roundings = 6;
constexpr int64_t exact_staged_roundings = 10;
constexpr int64_t ridge_roundings = 3;
constexpr int64_t applied_roundings = 8;
constexpr int64_t floor_roundings = 4;
constexpr int64_t block_roundings = 6;
constexpr double expected_residual_factor = 0.5;
constexpr double eigen_error_factor = 16.0;
constexpr double window_reach = 16.0;
constexpr double edge_gap = 2.0;
constexpr double certified_margin = 2.0;
constexpr double pass_ns = 0.0208;
constexpr double vector_square_ns = 0.077;
constexpr double vector_ns = 98000.0;
constexpr double row_ns = 3.9;
constexpr double gemm_ns = 0.0049;
constexpr double eigen_square_ns = 74.0;
constexpr double exact_fixed_ns = 1.8e6;
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

gram_batch batch_of(gram_shape shape, int64_t run) {
  const auto [n, d] = shape;
  return {.run = run,
          .blocks = std::min({std::max(int64_t{1}, staged_floats / (run * d)),
                              std::max(int64_t{1}, block_floats / (d * d)),
                              (n + run - 1) / run})};
}

double exact_ns(gram_shape shape) {
  const auto rows = static_cast<double>(shape.rows);
  const auto cols = static_cast<double>(shape.columns);
  return (row_ns * rows) + (gemm_ns * rows * cols * cols) +
         (eigen_square_ns * cols * cols) + exact_fixed_ns;
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

gram_batch wide_batch(gram_shape shape) {
  return batch_of(shape, gram_run_rows(shape.columns));
}

gram_batch exact_batch(gram_shape shape) {
  return batch_of(shape, exact_run_rows);
}

gram_bound bound_of(int64_t n, const std::vector<double>& spread,
                    const std::vector<double>& scale, double alpha) {
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

double mean_error(int64_t n, const std::vector<double>& spread,
                  const std::vector<double>& scale, const centring& c) {
  if (c.mean.empty()) {
    return 0.0;
  }
  const auto rows = static_cast<double>(n);
  const auto d = static_cast<int64_t>(spread.size());
  const double summed = 1.0 - rounding(rows + square_roundings);
  const double total = c.total * (1.0 - rounding(rows));
  const double relative = rounding((2.0 * rows) + mean_roundings);
  double squares = 0.0;
  for (int64_t k = 0; k < d; ++k) {
    const double reach =
        relative * scale[k] *
        (std::sqrt(spread[k] / (total * summed)) + std::abs(c.mean[k]));
    squares += reach * reach;
  }
  return squares * c.total * (1.0 + rounding(rows)) *
         (1.0 + rounding(static_cast<double>(d + host_roundings)));
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
  const double error =
      (entry * b.trace) + (ridge * (1.0 + entry) * b.diagonal) + b.mean;
  return error * (1.0 + rounding(static_cast<double>(host_roundings)));
}

double exact_error(gram_bound b, int64_t n) {
  const int64_t runs = ((n + exact_run_rows - 1) / exact_run_rows) + 1;
  const double entry = compounded(
      rounding(static_cast<double>(exact_run_rows + exact_staged_roundings)),
      rounding(static_cast<double>(runs)));
  const double ridge = rounding(static_cast<double>(ridge_roundings));
  const double error =
      (entry * b.trace) + (ridge * (1.0 + entry) * b.diagonal) + b.mean;
  return error * (1.0 + rounding(static_cast<double>(host_roundings)));
}

double applied_error(gram_bound b, gram_shape shape) {
  const double products = rounding(
      static_cast<double>(shape.rows + shape.columns + applied_roundings));
  return products * (b.trace + b.diagonal) *
         (1.0 + rounding(static_cast<double>(host_roundings)));
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

std::optional<enclosure_terms> terms_of(enclosure_norms r, int64_t d) {
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
  return enclosure_terms{.error = symmetric * (1.0 + rounding(enclosure_steps)),
                         .eta = eta};
}

std::optional<spectrum_interval> enclosure(const std::vector<double>& values,
                                           enclosure_norms r, int64_t d) {
  const std::optional<enclosure_terms> t = terms_of(r, d);
  if (!t) {
    return std::nullopt;
  }
  const double low = values.front() - t->error;
  const double high = values.back() + t->error;
  return spectrum_interval{
      outward(low / (low > 0 ? 1.0 + t->eta : 1.0 - t->eta), -infinite),
      outward(high / (high > 0 ? 1.0 - t->eta : 1.0 + t->eta), infinite)};
}

double complement_floor(const std::vector<double>& values, int64_t window,
                        enclosure_terms t) {
  const double next = values[window];
  const double lean = (next - values.front()) * t.eta * t.eta;
  const double slack = rounding(static_cast<double>(floor_roundings)) *
                       (std::abs(next) + lean + t.error);
  const double b = next - lean - t.error - slack;
  return outward(b / (b >= 0 ? 1.0 + t.eta : 1.0 - t.eta), -infinite);
}

double block_floor(block_bound b) {
  const auto [window, complement, coupling] = b;
  const double low = std::min(window, complement);
  if (!(coupling > 0)) {
    return low;
  }
  const double apart =
      std::abs(complement - window) * half * (1.0 - rounding(2.0));
  const double squared = coupling * coupling;
  const double lean = squared / (apart + std::sqrt((apart * apart) + squared)) *
                      (1.0 + rounding(static_cast<double>(block_roundings)));
  return outward(low - lean, -infinite);
}

bool exact_gram_cheaper(int64_t vectors, gram_shape shape) {
  return pass_cost_ns(vectors, shape) >= exact_ns(shape);
}

bool enclosure_pays(const std::vector<double>& values, double gram,
                    const gram_plan& plan, gram_shape shape, double limit) {
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

gram_plan plan_of(const std::vector<double>& values, double gram,
                  gram_shape shape, double limit) {
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
