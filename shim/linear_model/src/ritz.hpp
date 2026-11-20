#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace cumlrkt::glm::ritz {

constexpr int64_t gram_block_rows = 1024;
constexpr int64_t narrow_gram_columns = 16;
constexpr int64_t narrow_run_rows = 32;
constexpr int64_t pass_chunk = 8;
constexpr int64_t exact_run_rows = 1024;
constexpr int64_t staged_floats = int64_t{1} << 23;
constexpr int64_t block_floats = int64_t{1} << 22;

enum class gram_path : std::uint8_t { none, certified, enclosed, ritz, exact };

struct gram_shape {
  int64_t rows;
  int64_t columns;
};

struct gram_plan {
  gram_path path;
  int64_t window;
  double reach;
  double kappa;
};

struct gram_bound {
  int64_t runs;
  int64_t run_rows;
  double trace;
  double diagonal;
  double mean = 0.0;
};

struct centring {
  std::vector<double> mean;
  double total;
  std::vector<double> residual;
};

struct gram_batch {
  int64_t run;
  int64_t blocks;
};

struct eigen_pairs {
  std::vector<double> values;
  std::vector<double> vectors;
};

struct enclosure_norms {
  double symmetric;
  double orthogonality;
  double matrix;
  double basis;
  double product;
};

struct enclosure_terms {
  double error;
  double eta;
};

struct block_bound {
  double window;
  double complement;
  double coupling;
};

struct spectrum_interval {
  double low;
  double high;
};

struct ritz_pair {
  double value;
  std::vector<double> vector;
  std::vector<double> applied;
};

double dot(const double* a, const double* b, int64_t d);

std::vector<double> combined(const std::vector<double>& columns, int64_t d,
                             const double* c, int64_t k);

std::vector<double> symmetric_block(const std::vector<double>& m, int64_t c,
                                    int64_t k);

int64_t gram_run_rows(int64_t d);

gram_batch wide_batch(gram_shape shape);

gram_batch exact_batch(gram_shape shape);

gram_bound bound_of(int64_t n, const std::vector<double>& spread,
                    const std::vector<double>& scale, double alpha);

double mean_error(int64_t n, const std::vector<double>& spread,
                  const std::vector<double>& scale, const centring& c);

double gram_error(gram_bound b);

double exact_error(gram_bound b, int64_t n);

double applied_error(gram_bound b, gram_shape shape);

double spectrum_error(const std::vector<double>& values, double gram);

double expected_residual(const std::vector<double>& values);

std::optional<double> certified_kappa(spectrum_interval s, double limit);

std::optional<enclosure_terms> terms_of(enclosure_norms r, int64_t d);

std::optional<spectrum_interval> enclosure(const std::vector<double>& values,
                                           enclosure_norms r, int64_t d);

double complement_floor(const std::vector<double>& values, int64_t window,
                        enclosure_terms t);

double block_floor(block_bound b);

bool exact_gram_cheaper(int64_t vectors, gram_shape shape);

bool enclosure_pays(const std::vector<double>& values, double gram,
                    const gram_plan& plan, gram_shape shape, double limit);

gram_plan plan_of(const std::vector<double>& values, double gram,
                  gram_shape shape, double limit);

std::vector<double> refined(const eigen_pairs& e, int64_t window,
                            const ritz_pair& p, int64_t d);

}  // namespace cumlrkt::glm::ritz
