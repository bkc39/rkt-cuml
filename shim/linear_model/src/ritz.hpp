#pragma once

#include <cstdint>
#include <vector>

namespace cumlrkt::glm::ritz {

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

std::vector<double> orthonormal(const std::vector<double>& vectors, int64_t d,
                                int64_t k);

std::vector<double> combined(const std::vector<double>& columns, int64_t d,
                             const double* c, int64_t k);

std::vector<double> refined(const eigen_pairs& e, int64_t window,
                            const ritz_pair& p, int64_t d);

}  // namespace cumlrkt::glm::ritz
