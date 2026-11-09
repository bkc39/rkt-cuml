#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "fit_checks.hpp"
#include "glm_staging.hpp"
#include "residual_check.hpp"

namespace cumlrkt::glm {

inline constexpr int narrowest_spread_bits = 4;

struct cd_request {
  fit_request fit;
  double alpha;
  double l1_ratio;
  int32_t max_iter;
  double tol;
};

template <typename T>
struct cd_scale {
  int shared;
  T alpha;
};

struct cd_penalty {
  double l1;
  double l2;
};

struct cd_epochs {
  int n_iter;
  int64_t rows;
  double tol;
  double roundoff;
};

struct kkt_terms {
  std::optional<double> residual;
  double bound;
  double allowance;
};

template <typename T>
int cd_shared_cap(const staging& s, int64_t rows);

template <typename T>
cd_scale<T> cd_scale_of(const staging& s, int64_t rows, double alpha);

kkt_terms kkt_of(const residual_terms& t, const std::vector<double>& coef,
                 double weight, cd_penalty penalty, cd_epochs epochs);

void require_kkt(const kkt_terms& k, cd_epochs epochs);

bool absolute_stop(int n_iter, int max_iter, double largest, double tol);

void require_cd_rows(int64_t rows);

void require_cd_codes(const cd_request& request);

}  // namespace cumlrkt::glm
