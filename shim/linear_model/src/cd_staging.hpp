#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "fit_checks.hpp"
#include "glm_staging.hpp"
#include "residual_check.hpp"

namespace cumlrkt::glm {

inline constexpr int narrowest_column_slack_bits = 4;

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

struct zero_fit {
  double residual;
  double allowance;
  double alpha_max;
};

template <typename T>
int cd_shared_cap(const staging& s, int64_t rows);

template <typename T>
cd_scale<T> cd_scale_of(const staging& s, int64_t rows, double alpha);

kkt_terms kkt_of(const residual_terms& t, const std::vector<double>& coef,
                 double weight, cd_penalty penalty, cd_epochs epochs);

struct zero_request {
  double weight;
  double alpha;
  double l1_ratio;
  int64_t rows;
};

template <typename T>
zero_fit zero_fit_of(const residual_terms& t, const zero_request& r);

bool accepts_zero(const zero_fit& z);

void require_kkt(const kkt_terms& k, int n_iter, double tol);

struct unresolved_fit {
  double tol;
  double largest;
  double alpha;
  double alpha_max;
};

template <typename T>
void require_resolved(const kkt_terms& k, int n_iter, const unresolved_fit& f);

bool absolute_stop(int n_iter, int max_iter, double largest, double tol);

void require_cd_rows(int64_t rows);

void require_cd_codes(const cd_request& request);

}  // namespace cumlrkt::glm
