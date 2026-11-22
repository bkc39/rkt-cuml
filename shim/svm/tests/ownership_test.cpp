#include <cuda_runtime_api.h>
#include <cuml/svm/svm_model.h>
#include <cuml/svm/svm_parameter.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <cuml/common/logger.hpp>
#include <cuml/matrix/kernel_params.hpp>
#include <cuml/svm/svc.hpp>
#include <optional>
#include <raft/core/handle.hpp>
#include <random>
#include <rmm/device_uvector.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "cumlrkt/support.hpp"
#include "cumlrkt/transpose.hpp"
#include "svm_fixture.hpp"
#include "svm_model.hpp"

namespace cumlrkt::test {
namespace {

using cumlrkt::svm::copy_out;
using cumlrkt::svm::model_guard;
using cumlrkt::svm::rebuilt;

template <typename T>
std::vector<T> predicted(const raft::handle_t& h, T* columns, int n, int d,
                         ML::matrix::KernelParams& k,
                         const ML::SVM::SvmModel<T>& m, bool labels) {
  rmm::device_uvector<T> out(static_cast<std::size_t>(n), stream_of(h));
  ML::SVM::svcPredict(h, columns, n, d, k, m, out.data(), T{1024}, labels);
  return on_host(out.data(), out.size(), h);
}

template <typename T>
void expect_rebuilt_bitwise(int kernel, int n, int d) {
  const raft::handle_t h;
  problem<T> p = blobs<T>(n, d, static_cast<unsigned>(n + d + kernel));
  auto x = on_device(p.columns, h);
  auto y = on_device(p.labels, h);
  ML::matrix::KernelParams k{static_cast<ML::matrix::KernelType>(kernel), 3,
                             1.0 / d, 0.5};
  const ML::SVM::SvmParameter param = defaults();
  std::vector<T> decisions;
  std::vector<T> labels;
  std::vector<T> unique;
  std::optional<cumlrkt::svm::svc_parts<T>> kept;
  {
    ML::SVM::SvmModel<T> model{};
    model.support_matrix.nnz = -1;
    const model_guard<T> guard{h, model};
    ML::SVM::svcFit(h, x.data(), n, d, y.data(), param, k, model,
                    static_cast<const T*>(nullptr));
    settle(h);
    decisions = predicted(h, x.data(), n, d, k, model, false);
    labels = predicted(h, x.data(), n, d, k, model, true);
    unique = on_host(model.unique_labels, 2, h);
    kept.emplace(copy_out(h, model));
  }
  cumlrkt::svm::svc_parts<T>& parts = *kept;
  const int ns = static_cast<int>(parts.n_support);
  ASSERT_GT(ns, 0);
  rmm::device_uvector<T> columns(static_cast<std::size_t>(ns) * d,
                                 stream_of(h));
  to_columns(h, parts.support_vectors.data(), ns, d,
             static_cast<const T*>(nullptr), columns.data(), 0);
  const T b = on_host(parts.intercept.data(), 1, h)[0];
  ML::SVM::SvmModel<T> again =
      rebuilt<T>(ns, d, b, parts.dual.data(), columns.data());
  EXPECT_TRUE(
      bitwise(predicted(h, x.data(), n, d, k, again, false), decisions));
  auto classes = on_device(unique, h);
  again.n_classes = 2;
  again.unique_labels = classes.data();
  EXPECT_TRUE(bitwise(predicted(h, x.data(), n, d, k, again, true), labels));
  EXPECT_EQ(parts.n_negative + parts.n_positive, ns);
}

class ownership : public ::testing::TestWithParam<std::tuple<int, bool, bool>> {
};

constexpr int small_rows = 200;
constexpr int small_cols = 4;
constexpr int large_rows = 3000;
constexpr int large_cols = 12;

TEST_P(ownership, O1_rebuilt_predicts_bitwise) {
  CUMLRKT_REQUIRE_GPU();
  const auto [kernel, wide, large] = GetParam();
  const int n = large ? large_rows : small_rows;
  const int d = large ? large_cols : small_cols;
  if (wide) {
    expect_rebuilt_bitwise<double>(kernel, n, d);
  } else {
    expect_rebuilt_bitwise<float>(kernel, n, d);
  }
}

INSTANTIATE_TEST_SUITE_P(svc, ownership,
                         ::testing::Combine(::testing::Values(0, 1, 2, 3),
                                            ::testing::Bool(),
                                            ::testing::Bool()),
                         [](const auto& info) {
                           return std::string("kernel") +
                                  std::to_string(std::get<0>(info.param)) +
                                  (std::get<1>(info.param) ? "_f64" : "_f32") +
                                  (std::get<2>(info.param) ? "_large"
                                                           : "_small");
                         });

TEST(ownership_null, O2_free_of_a_zeroed_model_is_a_no_op) {
  CUMLRKT_REQUIRE_GPU();
  const raft::handle_t h;
  ML::SVM::SvmModel<float> narrow{};
  narrow.support_matrix.nnz = -1;
  ML::SVM::SvmModel<double> wide{};
  wide.support_matrix.nnz = -1;
  EXPECT_NO_THROW(ML::SVM::svmFreeBuffers(h, narrow));
  EXPECT_NO_THROW(ML::SVM::svmFreeBuffers(h, wide));
  settle(h);
}

}  // namespace
}  // namespace cumlrkt::test
