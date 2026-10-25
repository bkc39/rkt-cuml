#pragma once

#include <cstdint>
#include <string>

#include "cumlrkt/decomposition.h"
#include "decomposition_fixture.hpp"

namespace cumlrkt::test {

struct tsvd_fit_options {
  int64_t n_components = 1;
  int32_t solver = CUML_DECOMPOSITION_SOLVER_FULL;
  double tol = 0.0;
  int64_t iterations = default_iterations;
  int32_t with_transform = 0;
};

inline int tsvd_fit(const resources& res, const rr_view* x, record& out,
                    const tsvd_fit_options& o = {}) {
  return cuml_decomposition_tsvd_fit(res.handle, x, &out.r, o.n_components,
                                     o.solver, o.tol, o.iterations,
                                     o.with_transform);
}

inline std::string tsvd_fit_into(const resources& res, const rr_view* x,
                                 record& out, const tsvd_fit_options& o = {}) {
  return outcome(tsvd_fit(res, x, out, o));
}

using tsvd_apply = int (*)(void*, const rr_view*, const rr_view*, rr_deposit*);

inline std::string tsvd_apply_into(const resources& res, tsvd_apply entry,
                                   const rr_view* x, const rr_view* components,
                                   record& out) {
  return outcome(entry(res.handle, x, components, &out.r));
}

}  // namespace cumlrkt::test
