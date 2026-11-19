#include "symmetric_eigen.hpp"

#include <cusolverDn.h>

#include <cstdint>
#include <optional>
#include <raft/core/cusolver_macros.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cusolver_dn_handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <type_traits>
#include <vector>

#include "cumlrkt/support.hpp"
#include "glm_staging_device.hpp"
#include "ritz.hpp"

namespace cumlrkt::glm {

namespace {

constexpr auto fill = CUBLAS_FILL_MODE_LOWER;

void solved(cusolverStatus_t result) {
  RAFT_CUSOLVER_TRY(result);
}

template <typename S>
std::optional<ritz::eigen_pairs> decomposed(rmm::cuda_stream_view stream,
                                            const buffer<S>& values,
                                            const buffer<int>& info,
                                            const buffer<S>& a) {
  if (to_host(info, stream)[0] != 0) {
    return std::nullopt;
  }
  const std::vector<S> l = to_host(values, stream);
  const std::vector<S> v = to_host(a, stream);
  return ritz::eigen_pairs{{l.begin(), l.end()}, {v.begin(), v.end()}};
}

}  // namespace

template <typename S>
std::optional<ritz::eigen_pairs> eigen_of(const raft::handle_t& handle,
                                          const std::vector<S>& m, int64_t d,
                                          bool vectors) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  buffer<S> a = to_device(m, stream);
  buffer<S> values = scratch<S>(d, stream);
  cusolverDnHandle_t solver = raft::resource::get_cusolver_dn_handle(handle);
  solved(cusolverDnSetStream(solver, stream.value()));
  const auto dim = static_cast<int>(d);
  const auto mode =
      vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
  int lwork = 0;
  if constexpr (std::is_same_v<S, float>) {
    solved(cusolverDnSsyevd_bufferSize(solver, mode, fill, dim, a.data(), dim,
                                       values.data(), &lwork));
  } else {
    solved(cusolverDnDsyevd_bufferSize(solver, mode, fill, dim, a.data(), dim,
                                       values.data(), &lwork));
  }
  buffer<S> work = scratch<S>(lwork, stream);
  buffer<int> info = scratch<int>(1, stream);
  if constexpr (std::is_same_v<S, float>) {
    solved(cusolverDnSsyevd(solver, mode, fill, dim, a.data(), dim,
                            values.data(), work.data(), lwork, info.data()));
  } else {
    solved(cusolverDnDsyevd(solver, mode, fill, dim, a.data(), dim,
                            values.data(), work.data(), lwork, info.data()));
  }
  return decomposed(stream, values, info, a);
}

std::optional<ritz::eigen_pairs> pencil_of(const raft::handle_t& handle,
                                           const std::vector<double>& h,
                                           const std::vector<double>& s,
                                           int64_t d) {
  const rmm::cuda_stream_view stream = stream_of(handle);
  buffer<double> a = to_device(h, stream);
  buffer<double> b = to_device(s, stream);
  buffer<double> values = scratch<double>(d, stream);
  cusolverDnHandle_t solver = raft::resource::get_cusolver_dn_handle(handle);
  solved(cusolverDnSetStream(solver, stream.value()));
  const auto dim = static_cast<int>(d);
  constexpr auto type = CUSOLVER_EIG_TYPE_1;
  constexpr auto mode = CUSOLVER_EIG_MODE_VECTOR;
  int lwork = 0;
  solved(cusolverDnDsygvd_bufferSize(solver, type, mode, fill, dim, a.data(),
                                     dim, b.data(), dim, values.data(),
                                     &lwork));
  buffer<double> work = scratch<double>(lwork, stream);
  buffer<int> info = scratch<int>(1, stream);
  solved(cusolverDnDsygvd(solver, type, mode, fill, dim, a.data(), dim,
                          b.data(), dim, values.data(), work.data(), lwork,
                          info.data()));
  return decomposed(stream, values, info, a);
}

template std::optional<ritz::eigen_pairs> eigen_of<float>(
    const raft::handle_t&, const std::vector<float>&, int64_t, bool);
template std::optional<ritz::eigen_pairs> eigen_of<double>(
    const raft::handle_t&, const std::vector<double>&, int64_t, bool);

}  // namespace cumlrkt::glm
