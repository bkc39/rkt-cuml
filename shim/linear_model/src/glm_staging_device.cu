#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
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
struct profile_sums {
  T largest;
  int varies;
  product3 sum;
};

template <typename T>
struct profile_entry {
  const T* x;
  const T* w;
  int64_t d;
  int64_t reference;

  __device__ profile_sums<T> operator()(int64_t i, int64_t j) const {
    if (!counted(w, i)) {
      return {T{0}, 0, no_product()};
    }
    const T v = x[(i * d) + j];
    const T base = reference < 0 ? T{0} : x[(reference * d) + j];
    return {fabs(v), v != base ? 1 : 0,
            product_of(weight(w, i), static_cast<double>(v))};
  }
};

template <typename T>
struct add_profiles {
  __device__ profile_sums<T> operator()(profile_sums<T> a,
                                        profile_sums<T> b) const {
    return {a.largest < b.largest ? b.largest : a.largest, a.varies | b.varies,
            add_scaled{}(a.sum, b.sum)};
  }
};

template <int Columns, typename T>
std::vector<profile_sums<T>> profile_totals(const T* x, int64_t n, int64_t d,
                                            const T* w, row_selection rows,
                                            bool fit_intercept,
                                            rmm::cuda_stream_view stream) {
  const int64_t reference =
      fit_intercept ? first_positive(w, n, stream) : int64_t{-1};
  return column_reduce<Columns>(
      stream, rows, d, profile_entry<T>{x, w, d, reference}, add_profiles<T>{},
      profile_sums<T>{T{0}, 0, no_product()});
}

template <typename T>
double normalised_mean(const profile_sums<T>& p, double weights) {
  return p.sum.exponent == no_exponent
             ? 0.0
             : std::ldexp(p.sum.significand / weights,
                          p.sum.exponent +
                              unit_exponent(static_cast<double>(p.largest)));
}

template <typename T>
struct vector_magnitude {
  const T* v;
  const T* w;

  __device__ double operator()(int64_t i) const {
    return counted(w, i) ? fabs(static_cast<double>(v[i])) : 0.0;
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
  const power_of_two<double>* unit;
  int64_t d;

  __device__ double operator()(int64_t i, int64_t j) const {
    return scaled_by(static_cast<double>(x[(i * d) + j]), unit[j]);
  }
};

struct deviation_sums {
  double square;
  double deviation;
};

struct add_deviations {
  __device__ deviation_sums operator()(deviation_sums a,
                                       deviation_sums b) const {
    return {a.square + b.square, a.deviation + b.deviation};
  }
};

// Σw·c / Σw is the part of the mean lost to its rounding to a double.
template <typename T>
struct deviation_terms {
  normalised_entry<T> v;
  const T* w;
  const double* mean;

  __device__ deviation_sums operator()(int64_t i, int64_t j) const {
    if (!counted(w, i)) {
      return {0.0, 0.0};
    }
    const double c = v(i, j) - mean[j];
    return w == nullptr
               ? deviation_sums{c * c, c}
               : deviation_sums{weight(w, i) * c * c, weight(w, i) * c};
  }
};

}  // namespace

template <typename T>
profiles column_profiles(const matrix<T, int64_t>& x, const T* w,
                         row_selection rows, bool fit_intercept,
                         rmm::cuda_stream_view stream) {
  const std::vector<profile_sums<T>> sums = profile_totals<column_lanes>(
      x.data, x.n, x.d, w, rows, fit_intercept, stream);
  const double weights = fit_intercept ? selected_weight(stream, rows, w) : 0;
  std::vector<column_profile> columns(x.d);
  std::ranges::transform(sums, columns.begin(), [&](const profile_sums<T>& p) {
    return column_profile{static_cast<double>(p.largest), p.varies != 0,
                          fit_intercept ? normalised_mean(p, weights) : 0.0};
  });
  return {std::move(columns), weights};
}

template <typename T>
std::vector<spread_terms> column_spread(const matrix<T, int64_t>& x, const T* w,
                                        row_selection rows,
                                        const std::vector<int>& exponent,
                                        const profiles& p,
                                        rmm::cuda_stream_view stream) {
  std::vector<power_of_two<double>> unit(exponent.size());
  std::ranges::transform(exponent, unit.begin(), power_of_two_of<double>);
  std::vector<double> mean(p.columns.size());
  std::ranges::transform(p.columns, mean.begin(),
                         [](const column_profile& c) { return c.mean; });
  const buffer<power_of_two<double>> units = to_device(unit, stream);
  const buffer<double> means = to_device(mean, stream);
  const std::vector<deviation_sums> sums = column_reduce(
      stream, rows, x.d,
      deviation_terms<T>{normalised_entry<T>{x.data, units.data(), x.d}, w,
                         means.data()},
      add_deviations{}, deviation_sums{0.0, 0.0});
  std::vector<spread_terms> spread(sums.size());
  std::ranges::transform(sums, spread.begin(), [&](const deviation_sums& s) {
    return spread_terms{std::sqrt(s.square),
                        p.weight > 0 ? s.deviation / p.weight : 0.0};
  });
  return spread;
}

template <typename T>
centre target_centre(const T* y, const T* w, row_selection rows, int y_exponent,
                     rmm::cuda_stream_view stream) {
  const staged_target<T> target{y, power_of_two_of<T>(y_exponent)};
  const weighted_total t =
      vector_reduce(stream, rows, weighted_target<T>{target, w}, add_totals{},
                    weighted_total{0.0, 0.0});
  const double mean = t.value / t.weight;
  const double deviation =
      vector_reduce(stream, rows, target_deviation<T>{target, w, mean},
                    thrust::plus<double>{}, 0.0);
  return {mean, deviation / t.weight};
}

template <typename T>
bool constant_vector(const raft::handle_t& handle, const T* v, int64_t n,
                     const T* w, bool fit_intercept) {
  return profile_totals<1>(v, n, 1, w, row_selection{nullptr, n}, fit_intercept,
                           stream_of(handle))[0]
             .varies == 0;
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
void unstage(const raft::handle_t& handle, const T* staged, const int64_t* slot,
             const int* shift, int64_t d, T* coef) {
  each(handle, d, unstaged<T>{staged, slot, shift, coef});
}

template profiles column_profiles<float>(const matrix<float, int64_t>&,
                                         const float*, row_selection, bool,
                                         rmm::cuda_stream_view);
template profiles column_profiles<double>(const matrix<double, int64_t>&,
                                          const double*, row_selection, bool,
                                          rmm::cuda_stream_view);
template std::vector<spread_terms> column_spread<float>(
    const matrix<float, int64_t>&, const float*, row_selection,
    const std::vector<int>&, const profiles&, rmm::cuda_stream_view);
template std::vector<spread_terms> column_spread<double>(
    const matrix<double, int64_t>&, const double*, row_selection,
    const std::vector<int>&, const profiles&, rmm::cuda_stream_view);
template centre target_centre<float>(const float*, const float*, row_selection,
                                     int, rmm::cuda_stream_view);
template centre target_centre<double>(const double*, const double*,
                                      row_selection, int,
                                      rmm::cuda_stream_view);
template bool constant_vector<float>(const raft::handle_t&, const float*,
                                     int64_t, const float*, bool);
template bool constant_vector<double>(const raft::handle_t&, const double*,
                                      int64_t, const double*, bool);
template double largest_counted<float>(const float*, int64_t, const float*,
                                       rmm::cuda_stream_view);
template double largest_counted<double>(const double*, int64_t, const double*,
                                        rmm::cuda_stream_view);
template int64_t first_positive<float>(const float*, int64_t,
                                       rmm::cuda_stream_view);
template int64_t first_positive<double>(const double*, int64_t,
                                        rmm::cuda_stream_view);
template int64_t positive_count<float>(const float*, int64_t,
                                       rmm::cuda_stream_view);
template int64_t positive_count<double>(const double*, int64_t,
                                        rmm::cuda_stream_view);
template void positive_rows<float>(const float*, int64_t, int64_t*,
                                   rmm::cuda_stream_view);
template void positive_rows<double>(const double*, int64_t, int64_t*,
                                    rmm::cuda_stream_view);
template void stage_matrix<float>(const raft::handle_t&,
                                  const matrix<float, int64_t>&, const float*,
                                  row_selection, staged_columns, float*);
template void stage_matrix<double>(const raft::handle_t&,
                                   const matrix<double, int64_t>&,
                                   const double*, row_selection, staged_columns,
                                   double*);
template void stage_elements<float>(const raft::handle_t&, const float*,
                                    const float*, row_selection, int, float*);
template void stage_elements<double>(const raft::handle_t&, const double*,
                                     const double*, row_selection, int,
                                     double*);
template void unstage<float>(const raft::handle_t&, const float*,
                             const int64_t*, const int*, int64_t, float*);
template void unstage<double>(const raft::handle_t&, const double*,
                              const int64_t*, const int*, int64_t, double*);

}  // namespace cumlrkt::glm
