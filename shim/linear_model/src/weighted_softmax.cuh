// Adapted from cuML 26.08.00 (Apache-2.0, Copyright (c) 2018-2025 NVIDIA
// CORPORATION), cpp/src/glm/qn/glm_softmax.cuh lines 33-181: logSoftmaxKernel,
// launchLogsoftmax and Softmax. The change: each row's loss term and its dZ
// column are multiplied by the row's weight, which cuML's softmax never reads.
#pragma once

#include <cub/cub.cuh>
#include <cuda/functional>
#include <cuml/common/utils.hpp>
#include <glm/qn/glm_base.cuh>
#include <glm/qn/simple_mat.cuh>
#include <limits>
#include <raft/core/handle.hpp>
#include <raft/core/interruptible.hpp>
#include <raft/util/cuda_utils.cuh>

namespace cumlrkt::glm::logistic {

template <typename T, int BX, int BY>
CUML_KERNEL void weighted_log_softmax(T* out, T* dZ, const T* in,
                                      const T* labels, const T* weights, int C,
                                      int N) {
  using warp_reduce = cub::WarpReduce<T, BX>;
  using block_reduce =
      cub::BlockReduce<T, BX, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BY>;

  __shared__ union {
    typename warp_reduce::TempStorage warp[BY];
    typename block_reduce::TempStorage block;
    T value[BY];
  } shm;

  const int y = threadIdx.y + (blockIdx.x * BY);
  const int len = C * N;

  bool delta = false;
  if (threadIdx.x == 0) {
    shm.value[threadIdx.y] =
        y < N ? labels[y] : std::numeric_limits<T>::lowest();
  }
  __syncthreads();
  const T label = shm.value[threadIdx.y];
  __syncthreads();
  if (threadIdx.x == 0) {
    shm.value[threadIdx.y] = y < N ? weights[y] : T(0);
  }
  __syncthreads();
  const T weight = shm.value[threadIdx.y];
  __syncthreads();
  T eta_y = 0;
  T my_eta = 0;
  T eta_max = -1e9;
  T lse = 0;
  for (int x = threadIdx.x; x < C; x += BX) {
    const int idx = x + (y * C);
    if (idx < len) {
      my_eta = in[idx];
      if (x == label) {
        delta = true;
        eta_y = my_eta;
      }
      eta_max = raft::max<T>(my_eta, eta_max);
    }
  }
  const T top =
      warp_reduce(shm.warp[threadIdx.y]).Reduce(eta_max, cuda::maximum{});
  if (threadIdx.x == 0) {
    shm.value[threadIdx.y] = top;
  }
  __syncthreads();
  eta_max = shm.value[threadIdx.y];
  __syncthreads();

  if (C <= BX) {
    const int idx = threadIdx.x + (y * C);
    if (threadIdx.x < C && idx < len) {
      lse = raft::exp<T>(my_eta - eta_max);
    }
  } else {
    for (int x = threadIdx.x; x < C; x += BX) {
      const int idx = x + (y * C);
      if (idx < len) {
        lse += raft::exp<T>(in[idx] - eta_max);
      }
    }
  }
  const T total = warp_reduce(shm.warp[threadIdx.y]).Sum(lse);
  if (threadIdx.x == 0) {
    shm.value[threadIdx.y] = eta_max + raft::log<T>(total);
  }
  __syncthreads();
  lse = shm.value[threadIdx.y];
  __syncthreads();

  if (C <= BX) {
    const int idx = threadIdx.x + (y * C);
    if (threadIdx.x < C && idx < len) {
      dZ[idx] = weight * (raft::exp<T>(my_eta - lse) - (threadIdx.x == label));
    }
  } else {
    for (int x = threadIdx.x; x < C; x += BX) {
      const int idx = x + (y * C);
      if (idx < len) {
        dZ[idx] = weight * (raft::exp<T>(in[idx] - lse) - (x == label));
      }
    }
  }

  T loss = 0;
  if (delta) {
    loss = (weight * (lse - eta_y)) / N;
  }
  const T sum = block_reduce(shm.block).Sum(loss);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    raft::myAtomicAdd(out, sum);
  }
}

template <typename T, int BX, int BY>
void launch_weighted_log_softmax(T* loss, T* dZ, const T* Z, const T* labels,
                                 const T* weights, int C, int N,
                                 cudaStream_t stream) {
  const dim3 block(BX, BY);
  const dim3 grid(raft::ceildiv(N, BY));
  weighted_log_softmax<T, BX, BY>
      <<<grid, block, 0, stream>>>(loss, dZ, Z, labels, weights, C, N);
}

template <typename T>
void weighted_log_softmax_of(T* loss, T* dZ, const T* Z, const T* labels,
                             const T* weights, int C, int N,
                             cudaStream_t stream) {
  RAFT_CUDA_TRY(cudaMemsetAsync(loss, 0, sizeof(T), stream));
  raft::interruptible::synchronize(stream);
  if (C <= 4) {
    launch_weighted_log_softmax<T, 4, 64>(loss, dZ, Z, labels, weights, C, N,
                                          stream);
  } else if (C <= 8) {
    launch_weighted_log_softmax<T, 8, 32>(loss, dZ, Z, labels, weights, C, N,
                                          stream);
  } else if (C <= 16) {
    launch_weighted_log_softmax<T, 16, 16>(loss, dZ, Z, labels, weights, C, N,
                                           stream);
  } else {
    launch_weighted_log_softmax<T, 32, 8>(loss, dZ, Z, labels, weights, C, N,
                                          stream);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// The weights are the L4-b w~ (they sum to the row count), so the loss is
// (1/N) sum w~ l and qn's backward pass, which divides by N, is its gradient.
template <typename T>
struct weighted_softmax : ML::GLM::detail::GLMBase<T, weighted_softmax<T>> {
  using base = ML::GLM::detail::GLMBase<T, weighted_softmax<T>>;

  weighted_softmax(const raft::handle_t& handle, int D, int C,
                   bool fit_intercept, T* weights)
      : base(handle, D, C, fit_intercept) {
    this->sample_weights = weights;
  }

  void getLossAndDZ(T* loss, ML::SimpleDenseMat<T>& Z,
                    const ML::SimpleVec<T>& y, cudaStream_t stream) {
    weighted_log_softmax_of(loss, Z.data, Z.data, y.data, this->sample_weights,
                            Z.m, Z.n, stream);
  }

  T gradNorm(const ML::SimpleVec<T>& grad, T* dev_scalar, cudaStream_t stream) {
    return ML::nrmMax(grad, dev_scalar, stream);
  }
};

}  // namespace cumlrkt::glm::logistic
