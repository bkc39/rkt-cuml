#include <cmath>
#include <cstdint>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <vector>

#include "cd_device.hpp"
#include "cumlrkt/device.cuh"
#include "cumlrkt/scale.hpp"
#include "cumlrkt/support.hpp"
#include "glm_device.cuh"
#include "glm_staging.hpp"
#include "glm_staging_device.hpp"

namespace cumlrkt::glm {

namespace {

template <typename T>
struct root_weight {
  const T* w;
  double rows;
  double total;

  __device__ T operator()(int64_t i) const {
    return static_cast<T>(sqrt(static_cast<double>(w[i]) * rows / total));
  }
};

template <typename T>
struct weighted_column {
  const T* x;
  root_weight<T> root;
  row_selection rows;
  const int64_t* kept;
  power_of_two<T> stage;
  const double* mean;
  const double* remainder;
  int64_t d;
  T* out;

  __device__ void operator()(int64_t o) const {
    const int64_t k = o / rows.count;
    const int64_t i = source_row(rows, o - (k * rows.count));
    const double v =
        static_cast<double>(scaled_by(x[(i * d) + kept[k]], stage));
    out[o] = root(i) * static_cast<T>((v - mean[k]) - remainder[k]);
  }
};

template <typename T>
struct weighted_element {
  staged_target<T> y;
  root_weight<T> root;
  row_selection rows;
  centre c;
  T* out;

  __device__ void operator()(int64_t p) const {
    const int64_t i = source_row(rows, p);
    out[p] = root(i) * static_cast<T>((y(i) - c.mean) - c.remainder);
  }
};

}  // namespace

template <typename T>
cd_centres centres_of(const T* y, const T* w, row_selection rows,
                      const staging& s, bool fit_intercept,
                      rmm::cuda_stream_view stream) {
  const auto count = s.kept.size();
  cd_centres c{std::vector<double>(count, 0.0), std::vector<double>(count, 0.0),
               centre{0.0, 0.0}};
  if (!fit_intercept) {
    return c;
  }
  for (size_t k = 0; k < count; ++k) {
    const int64_t j = s.kept[k];
    const int shift = s.column_exponent[j] - unit_exponent(s.largest[j]);
    c.mean[k] = std::ldexp(s.normalised_mean[j], shift);
    c.remainder[k] = std::ldexp(s.normalised_remainder[j], shift);
  }
  c.y = target_centre(y, w, rows, s.y_exponent, stream);
  return c;
}

template <typename T>
double weight_total(const T* w, row_selection rows,
                    rmm::cuda_stream_view stream) {
  return selected_weight(stream, rows, w);
}

template <typename T>
staged_problem<T> stage_weighted(const raft::handle_t& handle,
                                 const matrix<T, int64_t>& x, const T* y,
                                 const T* w, row_selection rows,
                                 const staging& s, const cd_centres& c,
                                 double total) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  const auto count = static_cast<int64_t>(s.kept.size());
  staged_problem<T> p{scratch<T>(rows.count * count, stream),
                      scratch<T>(rows.count, stream), scratch<T>(0, stream)};
  const buffer<int64_t> kept = to_device(s.kept, stream);
  const buffer<double> means = to_device(c.mean, stream);
  const buffer<double> remainders = to_device(c.remainder, stream);
  const root_weight<T> root{w, static_cast<double>(rows.count), total};
  const power_of_two<T> stage = power_of_two_of<T>(s.y_exponent);
  each(handle, rows.count * count,
       weighted_column<T>{x.data, root, rows, kept.data(), stage, means.data(),
                          remainders.data(), x.d, p.x.data()});
  each(handle, rows.count,
       weighted_element<T>{staged_target<T>{y, stage}, root, rows, c.y,
                           p.y.data()});
  settle(handle);
  return p;
}

template cd_centres centres_of<float>(const float*, const float*, row_selection,
                                      const staging&, bool,
                                      rmm::cuda_stream_view);
template cd_centres centres_of<double>(const double*, const double*,
                                       row_selection, const staging&, bool,
                                       rmm::cuda_stream_view);
template double weight_total<float>(const float*, row_selection,
                                    rmm::cuda_stream_view);
template double weight_total<double>(const double*, row_selection,
                                     rmm::cuda_stream_view);
template staged_problem<float> stage_weighted<float>(
    const raft::handle_t&, const matrix<float, int64_t>&, const float*,
    const float*, row_selection, const staging&, const cd_centres&, double);
template staged_problem<double> stage_weighted<double>(
    const raft::handle_t&, const matrix<double, int64_t>&, const double*,
    const double*, row_selection, const staging&, const cd_centres&, double);

}  // namespace cumlrkt::glm
