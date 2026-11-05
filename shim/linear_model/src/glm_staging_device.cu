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
#include "glm_device.cuh"
#include "glm_staging.hpp"
#include "glm_staging_device.hpp"

namespace cumlrkt::glm {

namespace {

template <typename T>
struct profile_entry {
  const T* x;
  const T* w;
  int64_t d;
  int64_t reference;

  __device__ column_profile operator()(int64_t i, int64_t j) const {
    if (!counted(w, i)) {
      return {0.0, 0.0};
    }
    const auto v = static_cast<double>(x[(i * d) + j]);
    const double base =
        reference < 0 ? 0.0 : static_cast<double>(x[(reference * d) + j]);
    return {fabs(v), fabs(v - base)};
  }
};

struct larger_profile {
  __device__ column_profile operator()(column_profile a,
                                       column_profile b) const {
    return {a.largest < b.largest ? b.largest : a.largest,
            a.varies < b.varies ? b.varies : a.varies};
  }
};

struct weighted_total {
  double weight;
  double value;
};

struct add_totals {
  __device__ weighted_total operator()(weighted_total a,
                                       weighted_total b) const {
    return {a.weight + b.weight, a.value + b.value};
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
struct normalised_entry {
  const T* x;
  const T* w;
  const int* exponent;
  int64_t d;

  __device__ double operator()(int64_t i, int64_t j) const {
    return counted(w, i)
               ? scalbn(static_cast<double>(x[(i * d) + j]), exponent[j])
               : 0.0;
  }
};

template <typename T>
struct weighted_entry {
  normalised_entry<T> v;
  const T* w;

  __device__ weighted_total operator()(int64_t i, int64_t j) const {
    if (!counted(w, i)) {
      return {0.0, 0.0};
    }
    return {weight(w, i), weight(w, i) * v(i, j)};
  }
};

template <typename T>
struct deviation_square {
  normalised_entry<T> v;
  const T* w;
  const double* mean;

  __device__ double operator()(int64_t i, int64_t j) const {
    if (!counted(w, i)) {
      return 0.0;
    }
    const double c = v(i, j) - mean[j];
    return weight(w, i) * c * c;
  }
};

}  // namespace

template <typename T>
std::vector<column_profile> column_profiles(const matrix<T, int64_t>& x,
                                            const T* w, row_selection rows,
                                            bool fit_intercept,
                                            rmm::cuda_stream_view stream) {
  const int64_t reference =
      fit_intercept ? first_positive(w, x.n, stream) : int64_t{-1};
  return column_reduce(stream, rows, x.d,
                       profile_entry<T>{x.data, w, x.d, reference},
                       larger_profile{}, column_profile{0.0, 0.0});
}

template <typename T>
std::vector<double> column_spread(const matrix<T, int64_t>& x, const T* w,
                                  row_selection rows,
                                  const std::vector<int>& exponent,
                                  bool fit_intercept,
                                  rmm::cuda_stream_view stream) {
  const buffer<int> e = to_device(exponent, stream);
  const normalised_entry<T> v{x.data, w, e.data(), x.d};
  std::vector<double> mean(x.d, 0.0);
  if (fit_intercept) {
    const std::vector<weighted_total> sums =
        column_reduce(stream, rows, x.d, weighted_entry<T>{v, w}, add_totals{},
                      weighted_total{0.0, 0.0});
    std::ranges::transform(sums, mean.begin(), [](weighted_total t) {
      return t.value / t.weight;
    });
  }
  const buffer<double> means = to_device(mean, stream);
  std::vector<double> spread =
      column_reduce(stream, rows, x.d, deviation_square<T>{v, w, means.data()},
                    thrust::plus<double>{}, 0.0);
  std::ranges::transform(spread, spread.begin(),
                         [](double s) { return std::sqrt(s); });
  return spread;
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

#define CUMLRKT_GLM_STAGING_DEVICE(T)                                        \
  template std::vector<column_profile> column_profiles<T>(                   \
      const matrix<T, int64_t>&, const T*, row_selection, bool,              \
      rmm::cuda_stream_view);                                                \
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
  template std::vector<double> column_spread<T>(                             \
      const matrix<T, int64_t>&, const T*, row_selection,                    \
      const std::vector<int>&, bool, rmm::cuda_stream_view);

CUMLRKT_GLM_STAGING_DEVICE(float)
CUMLRKT_GLM_STAGING_DEVICE(double)

#undef CUMLRKT_GLM_STAGING_DEVICE

}  // namespace cumlrkt::glm
