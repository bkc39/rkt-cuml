#pragma once

#include <cstdint>
#include <vector>

namespace cumlrkt::glm::ritz {

constexpr int64_t gram_block_rows = 1024;
constexpr int64_t pass_chunk = 8;

enum class gram_path : std::uint8_t { certified, ritz, exact };

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

struct eigen_pairs {
  std::vector<double> values;
  std::vector<double> vectors;
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

double spectrum_error(const std::vector<double>& values, int64_t d);

bool exact_gram_cheaper(int64_t vectors, gram_shape shape);

gram_plan plan_of(const std::vector<double>& values, gram_shape shape,
                  double limit);

std::vector<double> refined(const eigen_pairs& e, int64_t window,
                            const ritz_pair& p, int64_t d);

}  // namespace cumlrkt::glm::ritz
