#include <cmath>
#include <format>
#include <type_traits>

#include "cumlrkt/linear_model.h"
#include "cumlrkt/support.hpp"
#include "glm_staging.hpp"
#include "gram_condition.hpp"

namespace cumlrkt::glm {

template <typename T>
double kappa_limit(glm_solver solver) {
  const double bound = std::ldexp(1.0, -tolerance_bits<T>) / unit_roundoff<T>;
  return solver.family == glm_family::ols ? bound * bound : bound;
}

template <typename T>
bool checks_conditioning(glm_solver solver, bool alpha_window) {
  if (solver.family == glm_family::ols) {
    return std::is_same_v<T, float> &&
           (solver.algorithm == CUML_LINEAR_MODEL_OLS_QR ||
            solver.algorithm == CUML_LINEAR_MODEL_OLS_SVD_QR);
  }
  return solver.family == glm_family::ridge && alpha_window &&
         solver.algorithm == CUML_LINEAR_MODEL_RIDGE_SVD;
}

template <typename T>
const char* ridge_advice() {
  return std::is_same_v<T, float>
             ? "X is ill-conditioned for this alpha (float64 or a larger "
               "alpha may fit)"
             : "X is ill-conditioned for this alpha (a larger alpha may "
               "fit)";
}

template <typename T>
void require_conditioned(double kappa, glm_solver solver) {
  const bool ols = solver.family == glm_family::ols;
  const double share = unit_roundoff<T> * (ols ? std::sqrt(kappa) : kappa);
  if (share < std::ldexp(1.0, -tolerance_bits<T>)) {
    return;
  }
  throw logic_error(std::format(
      "X: the {} solve's {} has u*kappa {} >= 2^-{}; {}", solver_name(solver),
      ols ? "X" : "regularised Gram matrix", printed(share), tolerance_bits<T>,
      ols ? "X may have collinear columns" : ridge_advice<T>()));
}

template bool checks_conditioning<float>(glm_solver, bool);
template bool checks_conditioning<double>(glm_solver, bool);
template void require_conditioned<float>(double, glm_solver);
template void require_conditioned<double>(double, glm_solver);
template double kappa_limit<float>(glm_solver);
template double kappa_limit<double>(glm_solver);
}  // namespace cumlrkt::glm
