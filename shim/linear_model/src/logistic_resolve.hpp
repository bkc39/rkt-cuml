#pragma once

#include <vector>

#include "logistic_device.hpp"
#include "logistic_fit.hpp"

namespace cumlrkt::glm::logistic {

struct penalties {
  double l1;
  double l2;
};

struct solve_check {
  qn_shape shape;
  std::vector<double> sigma;
  std::vector<double> mean;
  penalties pen;
  int n_iter;
};

struct resolution {
  double worst;
  double bound;
  double objective;

  bool resolved() const {
    return worst <= bound;
  }
};

resolution resolution_of(const residual_sums& sums,
                         const std::vector<double>& params,
                         const fit_options& options, const solve_check& c);

[[noreturn]] void refuse_cutoff(const resolution& r, const fit_options& options,
                                const solve_check& c);

}  // namespace cumlrkt::glm::logistic
