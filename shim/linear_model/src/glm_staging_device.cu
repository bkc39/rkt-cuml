#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <vector>

#include "cumlrkt/device.cuh"
#include "cumlrkt/scale.hpp"
#include "cumlrkt/support.hpp"
#include "glm_staging.hpp"
#include "glm_staging_device.hpp"

namespace cumlrkt::glm {

namespace {

template <typename T>
__device__ bool counted(const T* w, int64_t i) {
  return w == nullptr || w[i] > 0;
}

template <typename T>
__device__ double weight(const T* w, int64_t i) {
  return w == nullptr ? 1.0 : static_cast<double>(w[i]);
}

__device__ int64_t source_row(row_selection rows, int64_t i) {
  return rows.index == nullptr ? i : rows.index[i];
}

struct column_of {
  int64_t n;

  __device__ int64_t operator()(int64_t k) const {
    return k / n;
  }
};

template <typename F, typename V>
struct by_column {
  F f;
  int64_t n;

  __device__ V operator()(int64_t k) const {
    const int64_t j = k / n;
    return f(k - (j * n), j);
  }
};

template <typename V, typename F, typename Op>
std::vector<V> column_totals(rmm::cuda_stream_view stream, int64_t n, int64_t d,
                             F f, Op op) {
  buffer<V> totals = scratch<V>(d, stream);
  buffer<int64_t> columns = scratch<int64_t>(d, stream);
  const thrust::counting_iterator<int64_t> k(0);
  thrust::reduce_by_key(
      rmm::exec_policy(stream),
      thrust::make_transform_iterator(k, column_of{n}),
      thrust::make_transform_iterator(k + (n * d), column_of{n}),
      thrust::make_transform_iterator(k, by_column<F, V>{f, n}), columns.data(),
      totals.data(), thrust::equal_to<int64_t>{}, op);
  return to_host(totals, stream);
}

template <typename T>
struct positive_magnitude {
  const T* x;
  const T* w;
  int64_t d;

  __device__ double operator()(int64_t i, int64_t j) const {
    return counted(w, i) ? fabs(static_cast<double>(x[(i * d) + j])) : 0.0;
  }
};

struct larger {
  __device__ double operator()(double a, double b) const {
    return a < b ? b : a;
  }
};

template <typename T>
struct vector_magnitude {
  const T* v;
  const T* w;

  __device__ double operator()(int64_t i) const {
    return counted(w, i) ? fabs(static_cast<double>(v[i])) : 0.0;
  }
};

template <typename T>
struct unlike_reference {
  const T* x;
  const T* w;
  int64_t d;
  int64_t reference;
  int* flags;

  __device__ void operator()(int64_t k) const {
    const int64_t i = k / d;
    const int64_t j = k - (i * d);
    const T base = reference < 0 ? T{0} : x[(reference * d) + j];
    if (counted(w, i) && x[k] != base) {
      flags[j] = 1;
    }
  }
};

template <typename T>
struct positive {
  __device__ bool operator()(T w) const {
    return w > 0;
  }
};

template <typename T>
struct staged_column {
  const T* x;
  const T* w;
  row_selection rows;
  staged_columns columns;
  T* out;
  int64_t d;

  __device__ void operator()(int64_t o) const {
    const int64_t k = o / rows.count;
    const int64_t i = source_row(rows, o - (k * rows.count));
    out[o] = counted(w, i)
                 ? scaled(x[(i * d) + columns.kept[k]], columns.exponent[k])
                 : T{0};
  }
};

template <typename T>
struct staged_element {
  const T* v;
  const T* w;
  row_selection rows;
  T* out;
  int exponent;

  __device__ void operator()(int64_t o) const {
    const int64_t i = source_row(rows, o);
    out[o] = counted(w, i) ? scaled(v[i], exponent) : T{0};
  }
};

template <typename T>
struct weighted_value {
  const T* y;
  const T* w;

  __device__ double operator()(int64_t i) const {
    return weight(w, i) * static_cast<double>(y[i]);
  }
};

template <typename T>
struct weight_of {
  const T* w;

  __device__ double operator()(int64_t i) const {
    return weight(w, i);
  }
};

template <typename T>
double total(rmm::cuda_stream_view stream, int64_t n, T f) {
  const thrust::counting_iterator<int64_t> i(0);
  return thrust::transform_reduce(rmm::exec_policy(stream), i, i + n, f, 0.0,
                                  thrust::plus<double>{});
}

template <typename T>
struct unstaged {
  const T* staged;
  const int64_t* slot;
  const int* shift;
  T* coef;

  __device__ void operator()(int64_t j) const {
    const int64_t k = slot[j];
    coef[j] = k < 0 ? T{0} : scaled(staged[k], shift[k]);
  }
};

template <typename T>
struct residual_of {
  const T* x;
  const T* y;
  staged_columns columns;
  const T* coef;
  double intercept;
  int64_t d;
  double* r;

  __device__ void operator()(int64_t i) const {
    double fitted = intercept;
    for (int64_t k = 0; k < columns.count; ++k) {
      const T v = scaled(x[(i * d) + columns.kept[k]], columns.exponent[k]);
      fitted += static_cast<double>(v) * static_cast<double>(coef[k]);
    }
    r[i] = static_cast<double>(y[i]) - fitted;
  }
};

template <typename T>
struct column_weighted {
  const T* x;
  const T* w;
  staged_columns columns;
  int64_t d;

  __device__ double operator()(int64_t i, int64_t k) const {
    if (!counted(w, i)) {
      return 0.0;
    }
    return weight(w, i) *
           static_cast<double>(
               scaled(x[(i * d) + columns.kept[k]], columns.exponent[k]));
  }
};

struct add_moments {
  __device__ moment operator()(moment a, moment b) const {
    return {a.square + b.square, a.cross + b.cross};
  }
};

template <typename T>
struct column_moments {
  const T* x;
  const T* w;
  staged_columns columns;
  const double* mean;
  const double* r;
  int64_t d;

  __device__ moment operator()(int64_t i, int64_t k) const {
    if (!counted(w, i)) {
      return {0.0, 0.0};
    }
    const double v = static_cast<double>(scaled(x[(i * d) + columns.kept[k]],
                                                columns.exponent[k])) -
                     mean[k];
    const double wv = weight(w, i) * v;
    return {wv * v, wv * r[i]};
  }
};

template <typename T>
struct centred_square {
  const T* y;
  const T* w;
  double mean;

  __device__ double operator()(int64_t i) const {
    if (!counted(w, i)) {
      return 0.0;
    }
    const double v = static_cast<double>(y[i]) - mean;
    return weight(w, i) * v * v;
  }
};

}  // namespace

template <typename T>
std::vector<double> column_largest(const matrix<T, int64_t>& x, const T* w,
                                   rmm::cuda_stream_view stream) {
  return column_totals<double>(stream, x.n, x.d,
                               positive_magnitude<T>{x.data, w, x.d}, larger{});
}

template <typename T>
double largest_counted(const T* v, int64_t n, const T* w,
                       rmm::cuda_stream_view stream) {
  const thrust::counting_iterator<int64_t> i(0);
  return thrust::transform_reduce(rmm::exec_policy(stream), i, i + n,
                                  vector_magnitude<T>{v, w}, 0.0, larger{});
}

template <typename T>
int64_t first_positive(const T* w, int64_t n, rmm::cuda_stream_view stream) {
  if (w == nullptr) {
    return 0;
  }
  const int64_t found =
      thrust::find_if(rmm::exec_policy(stream), w, w + n, positive<T>{}) - w;
  return found == n ? -1 : found;
}

template <typename T>
int64_t positive_count(const T* w, int64_t n, rmm::cuda_stream_view stream) {
  if (w == nullptr) {
    return n;
  }
  return thrust::count_if(rmm::exec_policy(stream), w, w + n, positive<T>{});
}

template <typename T>
void positive_rows(const T* w, int64_t n, int64_t* rows,
                   rmm::cuda_stream_view stream) {
  const thrust::counting_iterator<int64_t> i(0);
  thrust::copy_if(rmm::exec_policy(stream), i, i + n, w, rows, positive<T>{});
}

template <typename T>
std::vector<char> constant_columns(const raft::handle_t& handle,
                                   const matrix<T, int64_t>& x, const T* w,
                                   bool fit_intercept) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const int64_t reference =
      fit_intercept ? first_positive(w, x.n, stream) : int64_t{-1};
  buffer<int> flags = scratch<int>(x.d, stream);
  raftrkt::cuda_check(
      cudaMemsetAsync(flags.data(), 0, flags.size() * sizeof(int),
                      stream.value()),
      "cudaMemsetAsync");
  each(handle, x.n * x.d,
       unlike_reference<T>{x.data, w, x.d, reference, flags.data()});
  const std::vector<int> varies = to_host(flags, stream);
  std::vector<char> constant(varies.size());
  std::ranges::transform(varies, constant.begin(),
                         [](int v) { return static_cast<char>(v == 0); });
  return constant;
}

template <typename T>
void stage_matrix(const raft::handle_t& handle, const matrix<T, int64_t>& x,
                  const T* w, row_selection rows, staged_columns columns,
                  T* out) {
  each(handle, rows.count * columns.count,
       staged_column<T>{x.data, w, rows, columns, out, x.d});
}

template <typename T>
void stage_elements(const raft::handle_t& handle, const T* v, const T* w,
                    row_selection rows, int exponent, T* out) {
  each(handle, rows.count, staged_element<T>{v, w, rows, out, exponent});
}

template <typename T>
double staged_mean(const raft::handle_t& handle, const T* y, const T* w,
                   int64_t n) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  return total(stream, n, weighted_value<T>{y, w}) /
         total(stream, n, weight_of<T>{w});
}

template <typename T>
void unstage(const raft::handle_t& handle, const T* staged, const int64_t* slot,
             const int* shift, int64_t d, T* coef) {
  each(handle, d, unstaged<T>{staged, slot, shift, coef});
}

template <typename T>
residual_terms residual_moments(const raft::handle_t& handle,
                                const matrix<T, int64_t>& x, const T* y,
                                const T* w, staged_columns columns,
                                const T* coef, double intercept,
                                bool fit_intercept) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  buffer<double> r = scratch<double>(x.n, stream);
  each(handle, x.n,
       residual_of<T>{x.data, y, columns, coef, intercept, x.d, r.data()});
  std::vector<double> mean(columns.count, 0.0);
  double y_mean = 0;
  if (fit_intercept) {
    const double weights = total(stream, x.n, weight_of<T>{w});
    const std::vector<double> sums = column_totals<double>(
        stream, x.n, columns.count, column_weighted<T>{x.data, w, columns, x.d},
        thrust::plus<double>{});
    std::ranges::transform(sums, mean.begin(),
                           [&](double v) { return v / weights; });
    y_mean = total(stream, x.n, weighted_value<T>{y, w}) / weights;
  }
  const buffer<double> means = to_device(mean, stream);
  return {column_totals<moment>(stream, x.n, columns.count,
                                column_moments<T>{x.data, w, columns,
                                                  means.data(), r.data(), x.d},
                                add_moments{}),
          std::sqrt(total(stream, x.n, centred_square<T>{y, w, y_mean}))};
}

#define CUMLRKT_GLM_STAGING_DEVICE(T)                                        \
  template std::vector<double> column_largest<T>(                            \
      const matrix<T, int64_t>&, const T*, rmm::cuda_stream_view);           \
  template double largest_counted<T>(const T*, int64_t, const T*,            \
                                     rmm::cuda_stream_view);                 \
  template int64_t first_positive<T>(const T*, int64_t,                      \
                                     rmm::cuda_stream_view);                 \
  template int64_t positive_count<T>(const T*, int64_t,                      \
                                     rmm::cuda_stream_view);                 \
  template void positive_rows<T>(const T*, int64_t, int64_t*,                \
                                 rmm::cuda_stream_view);                     \
  template std::vector<char> constant_columns<T>(                            \
      const raft::handle_t&, const matrix<T, int64_t>&, const T*, bool);     \
  template void stage_matrix<T>(const raft::handle_t&,                       \
                                const matrix<T, int64_t>&, const T*,         \
                                row_selection, staged_columns, T*);          \
  template void stage_elements<T>(const raft::handle_t&, const T*, const T*, \
                                  row_selection, int, T*);                   \
  template double staged_mean<T>(const raft::handle_t&, const T*, const T*,  \
                                 int64_t);                                   \
  template void unstage<T>(const raft::handle_t&, const T*, const int64_t*,  \
                           const int*, int64_t, T*);                         \
  template residual_terms residual_moments<T>(                               \
      const raft::handle_t&, const matrix<T, int64_t>&, const T*, const T*,  \
      staged_columns, const T*, double, bool);

CUMLRKT_GLM_STAGING_DEVICE(float)
CUMLRKT_GLM_STAGING_DEVICE(double)

#undef CUMLRKT_GLM_STAGING_DEVICE

}  // namespace cumlrkt::glm
