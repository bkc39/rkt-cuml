#include "logistic_softmax.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <raft/core/handle.hpp>
#include <random>
#include <vector>

#include "cumlrkt/support.hpp"
#include "glm_staging_device.hpp"
#include "linear_model_fixture.hpp"
#include "raftrkt/view.hpp"

namespace {

using cumlrkt::buffer;
using cumlrkt::scratch;
using cumlrkt::glm::to_device;
using cumlrkt::glm::to_host;
using cumlrkt::glm::logistic::softmax_loss_grad;
using cumlrkt::glm::logistic::softmax_problem;
using cumlrkt::test::resources;

constexpr int64_t rows = 32;
constexpr int64_t columns = 3;
constexpr unsigned seed = 124;
constexpr double least_weight = 0.2;
constexpr double most_weight = 3.0;
constexpr double float64_tol = 1e-12;
constexpr double float32_tol = 2e-5;

struct problem {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> w;
  std::vector<double> params;
  softmax_problem shape;
};

problem drawn(int64_t k, bool equal) {
  std::mt19937 gen(seed + static_cast<unsigned>(k));
  std::normal_distribution<double> normal;
  std::uniform_int_distribution<int64_t> label(0, k - 1);
  std::uniform_real_distribution<double> weight(least_weight, most_weight);
  const softmax_problem shape{rows, columns, k, true};
  problem p{std::vector<double>(rows * columns), std::vector<double>(rows),
            std::vector<double>(rows, 1.0),
            std::vector<double>((columns + 1) * k), shape};
  std::ranges::generate(p.x, [&] { return normal(gen); });
  std::ranges::generate(p.y, [&] { return static_cast<double>(label(gen)); });
  std::ranges::generate(p.params, [&] { return normal(gen); });
  if (!equal) {
    std::ranges::generate(p.w, [&] { return weight(gen); });
    double total = 0;
    for (const double v : p.w) {
      total += v;
    }
    std::ranges::transform(p.w, p.w.begin(), [&](double v) {
      return v * static_cast<double>(rows) / total;
    });
  }
  return p;
}

struct loss_grad {
  double loss;
  std::vector<double> grad;
};

loss_grad host_reference(const problem& p) {
  const int64_t k = p.shape.k;
  const auto m = static_cast<double>(rows);
  loss_grad out{0.0, std::vector<double>(p.params.size(), 0.0)};
  for (int64_t i = 0; i < rows; ++i) {
    std::vector<double> z(k);
    for (int64_t c = 0; c < k; ++c) {
      z[c] = p.params[(columns * k) + c];
      for (int64_t j = 0; j < columns; ++j) {
        z[c] += p.x[(i * columns) + j] * p.params[(j * k) + c];
      }
    }
    const double top = std::ranges::max(z);
    double total = 0;
    for (const double v : z) {
      total += std::exp(v - top);
    }
    const double lse = top + std::log(total);
    const auto label = static_cast<int64_t>(p.y[i]);
    out.loss += p.w[i] * (lse - z[label]) / m;
    for (int64_t c = 0; c < k; ++c) {
      const double r =
          p.w[i] * (std::exp(z[c] - lse) - (c == label ? 1.0 : 0.0)) / m;
      for (int64_t j = 0; j < columns; ++j) {
        out.grad[(j * k) + c] += r * p.x[(i * columns) + j];
      }
      out.grad[(columns * k) + c] += r;
    }
  }
  return out;
}

template <typename T>
std::vector<T> narrowed(const std::vector<double>& v) {
  return {v.begin(), v.end()};
}

template <typename T>
loss_grad device_loss_grad(const problem& p, bool weighted) {
  const resources res;
  const raft::handle_t& handle = raftrkt::handle_of(res.handle);
  const auto stream = handle.get_stream();
  buffer<T> x = to_device(narrowed<T>(p.x), stream);
  buffer<T> y = to_device(narrowed<T>(p.y), stream);
  buffer<T> w = to_device(narrowed<T>(p.w), stream);
  buffer<T> params = to_device(narrowed<T>(p.params), stream);
  buffer<T> grad = scratch<T>(static_cast<int64_t>(p.params.size()), stream);
  const T loss = softmax_loss_grad<T>(handle, x.data(), y.data(),
                                      weighted ? w.data() : nullptr,
                                      params.data(), p.shape, grad.data());
  const std::vector<T> g = to_host(grad, stream);
  return {static_cast<double>(loss), std::vector<double>(g.begin(), g.end())};
}

template <typename T>
void expect_reference(int64_t k, double tol) {
  const problem p = drawn(k, false);
  const loss_grad want = host_reference(p);
  const loss_grad got = device_loss_grad<T>(p, true);
  EXPECT_NEAR(got.loss, want.loss, tol * std::abs(want.loss)) << "k " << k;
  const double scale =
      std::ranges::max(want.grad, {}, [](double v) { return std::abs(v); });
  for (std::size_t q = 0; q < want.grad.size(); ++q) {
    EXPECT_NEAR(got.grad[q], want.grad[q], tol * std::abs(scale))
        << "k " << k << " entry " << q;
  }
}

template <typename T>
void expect_cuml_softmax(int64_t k) {
  const problem p = drawn(k, true);
  const loss_grad ours = device_loss_grad<T>(p, true);
  const loss_grad theirs = device_loss_grad<T>(p, false);
  EXPECT_EQ(ours.loss, theirs.loss) << "k " << k;
  EXPECT_EQ(ours.grad, theirs.grad) << "k " << k;
}

TEST(WeightedSoftmax, MatchesAFloat64HostReference) {
  CUMLRKT_REQUIRE_GPU();
  for (const int64_t k : {2, 3, 7}) {
    expect_reference<double>(k, float64_tol);
    expect_reference<float>(k, float32_tol);
  }
}

TEST(WeightedSoftmax, EqualWeightsAreCumlsSoftmaxBitForBit) {
  CUMLRKT_REQUIRE_GPU();
  for (const int64_t k : {2, 3, 7}) {
    expect_cuml_softmax<double>(k);
    expect_cuml_softmax<float>(k);
  }
}

}  // namespace
