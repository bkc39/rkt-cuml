#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "cluster_fixture.hpp"
#include "cumlrkt/cluster.h"
#include "kmeans_fixture.hpp"
#include "raftrkt/c_api.h"

namespace cumlrkt::test {
namespace {

constexpr int64_t grid_rows = 8;
constexpr int64_t grid_cols = 2;
constexpr int32_t far_k = 3;
constexpr int32_t random_restarts = 10;
constexpr double grid_inertia = 16.0;
constexpr double start_inertia = 20.0;
constexpr double weighted_inertia = 52.0;
constexpr double near_corner = 2.0;
constexpr double far_corner = 242.0;
constexpr double near_far_cell = 162.0;
constexpr int64_t far_row = 4;

const values grid = {0, 0, 0, 2, 2, 0, 2, 2, 10, 10, 10, 12, 12, 10, 12, 12};
const values start = {1, 0, 11, 11};
const values cell_means = {1, 1, 11, 11};
const values far_start = {1, 1, 11, 11, 1000, 1000};
const values weighted_grid = {0,  0,  0,  2,  2,  0,  2,  2,
                              10, 10, 10, 14, 14, 10, 14, 14};
const values weighted_start = {1, 0, 12, 11};
const values grid_weights = {1, 1, 1, 1, 3, 3, 3, 3};
const std::vector<int32_t> grid_labels = {0, 0, 0, 0, 1, 1, 1, 1};

template <typename T>
struct grid_case {
  grid_case(const resources& res, const values& data, const values& first,
            int64_t k = pair)
      : x(upload<T>(res, {grid_rows, grid_cols}, data)),
        init(upload<T>(res, {k, grid_cols}, first)) {}

  [[nodiscard]] fit_call call(record& out) const {
    return {.x = &x->view,
            .init = &init->view,
            .out = &out.r,
            .n_clusters = static_cast<int32_t>(init->view.shape[0])};
  }

  std::unique_ptr<device_array> x;
  std::unique_ptr<device_array> init;
};

constexpr int32_t centroids_region = 0;
constexpr int32_t labels_region = 1;

template <typename T>
void one_step_from_given_centroids() {
  SCOPED_TRACE(float_dtype<T>);
  const resources res;
  const grid_case<T> g(res, grid, start);
  record out(res);
  fit_call c = g.call(out);
  c.max_iter = 1;
  const fit_outcome r = fit(res, c);
  ASSERT_EQ(r.status, "accepted");
  EXPECT_EQ(shapes(out),
            (std::vector<int64_t>{2, float_dtype<T>, pair, grid_cols,
                                  RR_DTYPE_INT32, grid_rows}));
  EXPECT_EQ(download<T>(out, centroids_region), cell_means);
  EXPECT_EQ(std::pair(r.n_iter, r.inertia), std::pair(1, grid_inertia));
  EXPECT_EQ(out.template read<int32_t>(labels_region), grid_labels);
}

template <typename T>
void max_iter_zero_keeps_the_init() {
  SCOPED_TRACE(float_dtype<T>);
  const resources res;
  const grid_case<T> g(res, grid, start);
  record out(res);
  fit_call c = g.call(out);
  c.max_iter = 0;
  const fit_outcome r = fit(res, c);
  ASSERT_EQ(r.status, "accepted");
  EXPECT_EQ(download<T>(out, centroids_region), start);
  EXPECT_EQ(r.n_iter, 0);
  EXPECT_EQ(r.inertia, start_inertia);
}

template <typename T>
void leaves_the_init_untouched() {
  SCOPED_TRACE(float_dtype<T>);
  const resources res;
  const grid_case<T> g(res, grid, start);
  record out(res);
  ASSERT_EQ(fit(res, g.call(out)).status, "accepted");
  EXPECT_EQ(download<T>(*g.init), start);
  EXPECT_EQ(download<T>(out, centroids_region), cell_means);
}

template <typename T>
void every_init_fits_the_grid() {
  SCOPED_TRACE(float_dtype<T>);
  const resources res;
  const grid_case<T> g(res, grid, start);
  struct seeding {
    int32_t method;
    double factor;
    int32_t n_init;
  };
  for (const seeding s :
       {seeding{CUML_CLUSTER_KMEANS_INIT_KMEANS_PLUS_PLUS, 2.0, 1},
        seeding{CUML_CLUSTER_KMEANS_INIT_KMEANS_PLUS_PLUS, 0.0, 1},
        seeding{CUML_CLUSTER_KMEANS_INIT_RANDOM, 2.0, random_restarts}}) {
    SCOPED_TRACE(s.method);
    record out(res);
    fit_call c = g.call(out);
    c.init = nullptr;
    c.init_method = s.method;
    c.oversampling_factor = s.factor;
    c.n_init = s.n_init;
    c.seed = 1;
    const fit_outcome r = fit(res, c);
    ASSERT_EQ(r.status, "accepted");
    EXPECT_EQ(r.inertia, grid_inertia);
  }
}

template <typename T>
void weighted_inertia_is_normalised() {
  SCOPED_TRACE(float_dtype<T>);
  const resources res;
  const grid_case<T> g(res, weighted_grid, weighted_start);
  const auto w = upload<T>(res, {grid_rows}, grid_weights);
  record out(res);
  fit_call c = g.call(out);
  c.sample_weight = &w->view;
  const fit_outcome r = fit(res, c);
  ASSERT_EQ(r.status, "accepted");
  EXPECT_EQ(r.inertia, weighted_inertia);
}

template <typename T>
void empty_cluster_keeps_its_centroid() {
  SCOPED_TRACE(float_dtype<T>);
  const resources res;
  const grid_case<T> g(res, grid, far_start, far_k);
  record out(res);
  fit_call c = g.call(out);
  c.max_iter = 1;
  ASSERT_EQ(fit(res, c).status, "accepted");
  EXPECT_EQ(download<T>(out, centroids_region), far_start);
}

template <typename T>
void predict_labels() {
  SCOPED_TRACE(float_dtype<T>);
  const resources res;
  const auto x = upload<T>(res, {grid_rows, grid_cols}, grid);
  const auto c = upload<T>(res, {pair, grid_cols}, cell_means);
  record out(res);
  ASSERT_EQ(predict(res, &c->view, &x->view, &out.r), "accepted");
  EXPECT_EQ(shapes(out), (std::vector<int64_t>{1, RR_DTYPE_INT32, grid_rows}));
  EXPECT_EQ(out.read<int32_t>(0), grid_labels);
}

template <typename T>
void score_is_the_inertia() {
  SCOPED_TRACE(float_dtype<T>);
  const resources res;
  const auto x = upload<T>(res, {grid_rows, grid_cols}, grid);
  const auto c = upload<T>(res, {pair, grid_cols}, cell_means);
  double inertia = -1;
  ASSERT_EQ(
      outcome(cuml_cluster_kmeans_score(res.handle, &c->view, &x->view, nullptr,
                                        default_batch, &inertia)),
      "accepted");
  EXPECT_EQ(inertia, grid_inertia);
}

template <typename T>
void squared_distances() {
  SCOPED_TRACE(float_dtype<T>);
  const resources res;
  const auto x = upload<T>(res, {grid_rows, grid_cols}, grid);
  const auto c = upload<T>(res, {pair, grid_cols}, cell_means);
  record out(res);
  ASSERT_EQ(transform(res, &c->view, &x->view, &out.r), "accepted");
  ASSERT_EQ(shapes(out),
            (std::vector<int64_t>{1, float_dtype<T>, grid_rows, pair}));
  const values found = download<T>(out, 0);
  EXPECT_EQ((values{found[0], found[1], found[far_row * pair],
                    found[(far_row * pair) + 1]}),
            (values{near_corner, far_corner, near_far_cell, near_corner}));
}

TEST(Fit, OneStepFromGivenCentroidsIsTheCellMeans) {
  CUMLRKT_REQUIRE_GPU();
  one_step_from_given_centroids<float>();
  one_step_from_given_centroids<double>();
}

TEST(Fit, MaxIterZeroKeepsTheInit) {
  CUMLRKT_REQUIRE_GPU();
  max_iter_zero_keeps_the_init<float>();
  max_iter_zero_keeps_the_init<double>();
}

TEST(Fit, LeavesTheInitUntouched) {
  CUMLRKT_REQUIRE_GPU();
  leaves_the_init_untouched<float>();
  leaves_the_init_untouched<double>();
}

TEST(Fit, EveryInitFitsTheGrid) {
  CUMLRKT_REQUIRE_GPU();
  every_init_fits_the_grid<float>();
  every_init_fits_the_grid<double>();
}

TEST(Fit, WeightedInertiaIsNormalised) {
  CUMLRKT_REQUIRE_GPU();
  weighted_inertia_is_normalised<float>();
  weighted_inertia_is_normalised<double>();
}

TEST(Fit, EmptyClusterKeepsItsCentroid) {
  CUMLRKT_REQUIRE_GPU();
  empty_cluster_keeps_its_centroid<float>();
  empty_cluster_keeps_its_centroid<double>();
}

TEST(Predict, LabelsOfTheGrid) {
  CUMLRKT_REQUIRE_GPU();
  predict_labels<float>();
  predict_labels<double>();
}

TEST(Score, IsTheInertiaOfTheGrid) {
  CUMLRKT_REQUIRE_GPU();
  score_is_the_inertia<float>();
  score_is_the_inertia<double>();
}

TEST(Transform, SquaredDistances) {
  CUMLRKT_REQUIRE_GPU();
  squared_distances<float>();
  squared_distances<double>();
}

}  // namespace
}  // namespace cumlrkt::test
