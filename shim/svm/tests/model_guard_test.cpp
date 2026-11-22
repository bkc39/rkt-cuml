#include <cuml/svm/svm_model.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cuml/matrix/kernel_params.hpp>
#include <cuml/svm/svc.hpp>
#include <raft/core/handle.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "cumlrkt/support.hpp"
#include "svm_fixture.hpp"
#include "svm_model.hpp"

namespace cumlrkt::test {
namespace {

using cumlrkt::svm::model_guard;
using cumlrkt::svm::stall_translated;

constexpr int rows = 300;
constexpr int cols = 4;
constexpr unsigned blob_seed = 5;

template <typename T>
void guarded_fit(const raft::handle_t& h, T* x, T* y, int n) {
  ML::matrix::KernelParams k{ML::matrix::KernelType::RBF, 3, 1.0 / cols, 0.0};
  const ML::SVM::SvmParameter param = defaults();
  ML::SVM::SvmModel<T> model{};
  model.support_matrix.nnz = -1;
  const model_guard<T> guard{h, model};
  ML::SVM::svcFit(h, x, n, cols, y, param, k, model,
                  static_cast<const T*>(nullptr));
  settle(h);
  throw std::runtime_error("after the fit");
}

template <typename T>
std::string fit_failure(const raft::handle_t& h, T* x, T* y) {
  std::string what;
  try {
    guarded_fit(h, x, y, rows);
  } catch (const std::exception& e) {
    what = e.what();
  }
  settle(h);
  return what;
}

template <typename T>
void expect_flat() {
  const allocation_meter meter;
  const raft::handle_t h;
  problem<T> p = blobs<T>(rows, cols, blob_seed);
  auto x = on_device(p.columns, h);
  auto y = on_device(p.labels, h);
  std::vector<T> one(p.labels.size(), static_cast<T>(label_low));
  auto single = on_device(one, h);
  settle(h);
  EXPECT_EQ(fit_failure(h, x.data(), y.data()), "after the fit");
  const int64_t before = meter.current();
  EXPECT_EQ(fit_failure(h, x.data(), y.data()), "after the fit");
  EXPECT_EQ(meter.current(), before);
  EXPECT_FALSE(fit_failure(h, x.data(), single.data()).empty());
  EXPECT_EQ(meter.current(), before);
}

TEST(model_guard, O3_a_thrown_fit_frees_every_buffer_float) {
  CUMLRKT_REQUIRE_GPU();
  expect_flat<float>();
}

TEST(model_guard, O3_a_thrown_fit_frees_every_buffer_double) {
  CUMLRKT_REQUIRE_GPU();
  expect_flat<double>();
}

TEST(model_guard, a_stalled_working_set_names_its_remedy) {
  try {
    stall_translated([]() -> int {
      throw std::runtime_error(
          "exception occurred! file=ws_util.cuh Working set has already been "
          "initialized!");
    });
    FAIL() << "accepted";
  } catch (const cumlrkt::logic_error& e) {
    EXPECT_STREQ(e.what(),
                 "the SMO solver stopped making progress; raise #:tol");
  }
}

TEST(model_guard, other_failures_pass_through) {
  try {
    stall_translated(
        []() -> int { throw std::runtime_error("another failure"); });
    FAIL() << "accepted";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ(e.what(), "another failure");
  } catch (...) {
    FAIL() << "rethrown as another type";
  }
}

TEST(model_guard, results_pass_through) {
  EXPECT_EQ(stall_translated([] { return 3; }), 3);
}

}  // namespace
}  // namespace cumlrkt::test
