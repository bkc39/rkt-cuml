#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <raft/core/handle.hpp>
#include <vector>

#include "cumlrkt/support.hpp"
#include "glm_staging.hpp"
#include "logistic_device.hpp"
#include "logistic_fit.hpp"
#include "logistic_resolve.hpp"

namespace cumlrkt::glm::logistic {

template <typename T>
struct staged_fit {
  std::optional<buffer<T>> copied;
  T* source;
  buffer<T> y;
  std::optional<buffer<T>> w;
  std::vector<int64_t> kept;
  int shift;
  penalties pen;
  qn_shape shape;
  std::vector<double> sigma;
  std::vector<double> mean;

  T* x() {
    return copied ? copied->data() : source;
  }
  T* weights() {
    return w ? w->data() : nullptr;
  }
};

template <typename T>
struct qn_solution {
  buffer<T> params;
  int n_iter;
};

template <typename T>
struct attempt {
  qn_solution<T> solution;
  resolution r;
  solve_check check;
};

template <typename T>
qn_solution<T> qn_solved(const raft::handle_t& handle, T* x, T* y, T* w,
                         const qn_shape& shape, penalties pen,
                         const fit_options& options,
                         const std::vector<T>& start);

template <typename T>
using qn_solver = std::function<qn_solution<T>(
    const raft::handle_t&, T*, T*, T*, const qn_shape&, penalties,
    const fit_options&, const std::vector<T>&)>;

template <typename T>
attempt<T> attempted(const raft::handle_t& handle, T* x, T* y, T* w,
                     solve_check check, const fit_options& options,
                     const std::vector<T>& start = {});

template <typename T>
struct rescue {
  std::optional<qn_solution<T>> solution;
  int n_iter;
  std::optional<resolution> last;
};

template <typename T>
rescue<T> rescued(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                  row_selection rows, const column_stats& stats,
                  staged_fit<T>& st, const attempt<T>& plain,
                  const fit_options& options,
                  const qn_solver<T>& solve = qn_solved<T>);

}  // namespace cumlrkt::glm::logistic
