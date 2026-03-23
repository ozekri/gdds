#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr unsigned kWarpSize = 32;
constexpr int kVecWidth = 4;

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* ptr) {
  return static_cast<float>(*ptr);
}

template <>
__device__ __forceinline__ float load_as_float<c10::Half>(const c10::Half* ptr) {
  return __half2float(*reinterpret_cast<const __half*>(ptr));
}

template <>
__device__ __forceinline__ float load_as_float<c10::BFloat16>(const c10::BFloat16* ptr) {
#if __CUDA_ARCH__ >= 800
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(ptr));
#else
  return static_cast<float>(*ptr);
#endif
}

__device__ __forceinline__ void update_best(float score, int idx, float& best_score, int& best_idx) {
  if (score > best_score) {
    best_score = score;
    best_idx = idx;
  }
}

__device__ __forceinline__ uint32_t mix32(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return static_cast<uint32_t>(x);
}

__device__ __forceinline__ float uniform_from_counter(uint64_t seed, int row, int col) {
  uint64_t key = seed;
  key ^= static_cast<uint64_t>(row) * 0x9E3779B185EBCA87ULL;
  key ^= static_cast<uint64_t>(col) * 0xC2B2AE3D27D4EB4FULL;
  uint32_t bits = mix32(key);
  // Map to (0, 1], avoiding exact zero for the Gumbel transform.
  return (static_cast<float>(bits) + 1.0f) * 2.3283064365386963e-10f;
}

template <typename scalar_t>
__global__ void sample_block_gumbel_argmax_kernel_generic(
    const scalar_t* __restrict__ logr,
    const scalar_t* __restrict__ exponents,
    float* __restrict__ out_scores,
    int64_t* __restrict__ out_indices,
    int rows,
    int cols,
    uint64_t seed) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  float local_best = -FLT_MAX;
  int local_best_idx = 0;
  float exponent = load_as_float(exponents + row);

  int col = threadIdx.x;
  int stride = blockDim.x;
  int base = row * cols;

  for (; col + kVecWidth * stride <= cols; col += kVecWidth * stride) {
    float u0 = uniform_from_counter(seed, row, col);
    float u1 = uniform_from_counter(seed, row, col + stride);
    float u2 = uniform_from_counter(seed, row, col + 2 * stride);
    float u3 = uniform_from_counter(seed, row, col + 3 * stride);

    float s0 = exponent * load_as_float(logr + base + col) + (-logf(-logf(u0)));
    float s1 = exponent * load_as_float(logr + base + col + stride) + (-logf(-logf(u1)));
    float s2 = exponent * load_as_float(logr + base + col + 2 * stride) + (-logf(-logf(u2)));
    float s3 = exponent * load_as_float(logr + base + col + 3 * stride) + (-logf(-logf(u3)));

    update_best(s0, col, local_best, local_best_idx);
    update_best(s1, col + stride, local_best, local_best_idx);
    update_best(s2, col + 2 * stride, local_best, local_best_idx);
    update_best(s3, col + 3 * stride, local_best, local_best_idx);
  }

  for (; col < cols; col += stride) {
    float u = uniform_from_counter(seed, row, col);
    float score = exponent * load_as_float(logr + base + col) + (-logf(-logf(u)));
    update_best(score, col, local_best, local_best_idx);
  }

  int lane = threadIdx.x & (kWarpSize - 1);
  int warp_id = threadIdx.x / kWarpSize;

  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    float other_score = __shfl_down_sync(0xffffffff, local_best, offset);
    int other_index = __shfl_down_sync(0xffffffff, local_best_idx, offset);
    if (other_score > local_best) {
      local_best = other_score;
      local_best_idx = other_index;
    }
  }

  __shared__ float shared_scores[kThreads / kWarpSize];
  __shared__ int shared_indices[kThreads / kWarpSize];

  if (lane == 0) {
    shared_scores[warp_id] = local_best;
    shared_indices[warp_id] = local_best_idx;
  }
  __syncthreads();

  if (warp_id == 0) {
    local_best = lane < (kThreads / kWarpSize) ? shared_scores[lane] : -FLT_MAX;
    local_best_idx = lane < (kThreads / kWarpSize) ? shared_indices[lane] : 0;

    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      float other_score = __shfl_down_sync(0xffffffff, local_best, offset);
      int other_index = __shfl_down_sync(0xffffffff, local_best_idx, offset);
      if (other_score > local_best) {
        local_best = other_score;
        local_best_idx = other_index;
      }
    }

    if (lane == 0) {
      out_scores[row] = local_best;
      out_indices[row] = static_cast<int64_t>(local_best_idx);
    }
  }
}

template <typename scalar_t, int kCols>
__global__ void sample_block_gumbel_argmax_kernel_fixed_cols(
    const scalar_t* __restrict__ logr,
    const scalar_t* __restrict__ exponents,
    float* __restrict__ out_scores,
    int64_t* __restrict__ out_indices,
    int rows,
    uint64_t seed) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  float local_best = -FLT_MAX;
  int local_best_idx = 0;
  float exponent = load_as_float(exponents + row);

  constexpr int kStride = kThreads;
  constexpr int kBaseStep = kVecWidth * kStride;
  constexpr int kNumSteps = kCols / kBaseStep;
  static_assert(kCols % kBaseStep == 0, "fixed-width sampler expects a multiple of 4 * kThreads");

  int base = row * kCols;
  int col = threadIdx.x;

#pragma unroll
  for (int step = 0; step < kNumSteps; ++step) {
    int offset = col + step * kBaseStep;

    float u0 = uniform_from_counter(seed, row, offset);
    float u1 = uniform_from_counter(seed, row, offset + kStride);
    float u2 = uniform_from_counter(seed, row, offset + 2 * kStride);
    float u3 = uniform_from_counter(seed, row, offset + 3 * kStride);

    float s0 = exponent * load_as_float(logr + base + offset) + (-logf(-logf(u0)));
    float s1 = exponent * load_as_float(logr + base + offset + kStride) + (-logf(-logf(u1)));
    float s2 = exponent * load_as_float(logr + base + offset + 2 * kStride) + (-logf(-logf(u2)));
    float s3 = exponent * load_as_float(logr + base + offset + 3 * kStride) + (-logf(-logf(u3)));

    update_best(s0, offset, local_best, local_best_idx);
    update_best(s1, offset + kStride, local_best, local_best_idx);
    update_best(s2, offset + 2 * kStride, local_best, local_best_idx);
    update_best(s3, offset + 3 * kStride, local_best, local_best_idx);
  }

  int lane = threadIdx.x & (kWarpSize - 1);
  int warp_id = threadIdx.x / kWarpSize;

  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    float other_score = __shfl_down_sync(0xffffffff, local_best, offset);
    int other_index = __shfl_down_sync(0xffffffff, local_best_idx, offset);
    if (other_score > local_best) {
      local_best = other_score;
      local_best_idx = other_index;
    }
  }

  __shared__ float shared_scores[kThreads / kWarpSize];
  __shared__ int shared_indices[kThreads / kWarpSize];

  if (lane == 0) {
    shared_scores[warp_id] = local_best;
    shared_indices[warp_id] = local_best_idx;
  }
  __syncthreads();

  if (warp_id == 0) {
    local_best = lane < (kThreads / kWarpSize) ? shared_scores[lane] : -FLT_MAX;
    local_best_idx = lane < (kThreads / kWarpSize) ? shared_indices[lane] : 0;

    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      float other_score = __shfl_down_sync(0xffffffff, local_best, offset);
      int other_index = __shfl_down_sync(0xffffffff, local_best_idx, offset);
      if (other_score > local_best) {
        local_best = other_score;
        local_best_idx = other_index;
      }
    }

    if (lane == 0) {
      out_scores[row] = local_best;
      out_indices[row] = static_cast<int64_t>(local_best_idx);
    }
  }
}

template <typename scalar_t>
void launch_sample_block_gumbel_argmax_kernel(
    const scalar_t* logr,
    const scalar_t* exponents,
    float* out_scores,
    int64_t* out_indices,
    int rows,
    int cols,
    uint64_t seed,
    cudaStream_t stream) {
  switch (cols) {
    case 1024:
      sample_block_gumbel_argmax_kernel_fixed_cols<scalar_t, 1024><<<rows, kThreads, 0, stream>>>(
          logr, exponents, out_scores, out_indices, rows, seed);
      return;
    case 2048:
      sample_block_gumbel_argmax_kernel_fixed_cols<scalar_t, 2048><<<rows, kThreads, 0, stream>>>(
          logr, exponents, out_scores, out_indices, rows, seed);
      return;
    case 4096:
      sample_block_gumbel_argmax_kernel_fixed_cols<scalar_t, 4096><<<rows, kThreads, 0, stream>>>(
          logr, exponents, out_scores, out_indices, rows, seed);
      return;
    default:
      sample_block_gumbel_argmax_kernel_generic<scalar_t><<<rows, kThreads, 0, stream>>>(
          logr, exponents, out_scores, out_indices, rows, cols, seed);
      return;
  }
}

}  // namespace


std::vector<torch::Tensor> sample_block_gumbel_argmax_cuda(
    const torch::Tensor& chunk_logr,
    const torch::Tensor& chunk_exp,
    int64_t seed) {
  TORCH_CHECK(chunk_logr.is_cuda(), "chunk_logr must be CUDA");
  TORCH_CHECK(chunk_exp.is_cuda(), "chunk_exp must be CUDA");

  auto logr = chunk_logr.contiguous();
  auto exponents = chunk_exp.contiguous();
  auto rows = static_cast<int>(logr.size(0));
  auto cols = static_cast<int>(logr.size(1));

  auto out_scores = torch::empty({rows}, logr.options().dtype(torch::kFloat32));
  auto out_indices = torch::empty({rows}, logr.options().dtype(torch::kInt64));

  auto stream = at::cuda::getDefaultCUDAStream();

  switch (logr.scalar_type()) {
    case at::ScalarType::Float:
      launch_sample_block_gumbel_argmax_kernel<float>(
          logr.data_ptr<float>(),
          exponents.data_ptr<float>(),
          out_scores.data_ptr<float>(),
          out_indices.data_ptr<int64_t>(),
          rows,
          cols,
          static_cast<uint64_t>(seed),
          stream);
      break;
    case at::ScalarType::Half:
      launch_sample_block_gumbel_argmax_kernel<c10::Half>(
          logr.data_ptr<c10::Half>(),
          exponents.data_ptr<c10::Half>(),
          out_scores.data_ptr<float>(),
          out_indices.data_ptr<int64_t>(),
          rows,
          cols,
          static_cast<uint64_t>(seed),
          stream);
      break;
    case at::ScalarType::BFloat16:
      launch_sample_block_gumbel_argmax_kernel<c10::BFloat16>(
          logr.data_ptr<c10::BFloat16>(),
          exponents.data_ptr<c10::BFloat16>(),
          out_scores.data_ptr<float>(),
          out_indices.data_ptr<int64_t>(),
          rows,
          cols,
          static_cast<uint64_t>(seed),
          stream);
      break;
    default:
      TORCH_CHECK(false, "Unsupported dtype for CUDA block sampler: ", logr.scalar_type());
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_scores, out_indices};
}

template <typename scalar_t>
__global__ void sample_block_gumbel_argmax_indexed_kernel_generic(
    const scalar_t* __restrict__ unique_logr,
    const int64_t* __restrict__ row_index,
    const scalar_t* __restrict__ exponents,
    float* __restrict__ out_scores,
    int64_t* __restrict__ out_indices,
    int rows,
    int cols,
    uint64_t seed) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  int64_t src_row = row_index[row];
  float local_best = -FLT_MAX;
  int local_best_idx = 0;
  float exponent = load_as_float(exponents + row);

  int col = threadIdx.x;
  int stride = blockDim.x;
  int base = static_cast<int>(src_row) * cols;

  for (; col + kVecWidth * stride <= cols; col += kVecWidth * stride) {
    float u0 = uniform_from_counter(seed, row, col);
    float u1 = uniform_from_counter(seed, row, col + stride);
    float u2 = uniform_from_counter(seed, row, col + 2 * stride);
    float u3 = uniform_from_counter(seed, row, col + 3 * stride);

    float s0 = exponent * load_as_float(unique_logr + base + col) + (-logf(-logf(u0)));
    float s1 = exponent * load_as_float(unique_logr + base + col + stride) + (-logf(-logf(u1)));
    float s2 = exponent * load_as_float(unique_logr + base + col + 2 * stride) + (-logf(-logf(u2)));
    float s3 = exponent * load_as_float(unique_logr + base + col + 3 * stride) + (-logf(-logf(u3)));

    update_best(s0, col, local_best, local_best_idx);
    update_best(s1, col + stride, local_best, local_best_idx);
    update_best(s2, col + 2 * stride, local_best, local_best_idx);
    update_best(s3, col + 3 * stride, local_best, local_best_idx);
  }

  for (; col < cols; col += stride) {
    float u = uniform_from_counter(seed, row, col);
    float score = exponent * load_as_float(unique_logr + base + col) + (-logf(-logf(u)));
    update_best(score, col, local_best, local_best_idx);
  }

  int lane = threadIdx.x & (kWarpSize - 1);
  int warp_id = threadIdx.x / kWarpSize;

  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    float other_score = __shfl_down_sync(0xffffffff, local_best, offset);
    int other_index = __shfl_down_sync(0xffffffff, local_best_idx, offset);
    if (other_score > local_best) {
      local_best = other_score;
      local_best_idx = other_index;
    }
  }

  __shared__ float shared_scores[kThreads / kWarpSize];
  __shared__ int shared_indices[kThreads / kWarpSize];

  if (lane == 0) {
    shared_scores[warp_id] = local_best;
    shared_indices[warp_id] = local_best_idx;
  }
  __syncthreads();

  if (warp_id == 0) {
    local_best = lane < (kThreads / kWarpSize) ? shared_scores[lane] : -FLT_MAX;
    local_best_idx = lane < (kThreads / kWarpSize) ? shared_indices[lane] : 0;

    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      float other_score = __shfl_down_sync(0xffffffff, local_best, offset);
      int other_index = __shfl_down_sync(0xffffffff, local_best_idx, offset);
      if (other_score > local_best) {
        local_best = other_score;
        local_best_idx = other_index;
      }
    }

    if (lane == 0) {
      out_scores[row] = local_best;
      out_indices[row] = static_cast<int64_t>(local_best_idx);
    }
  }
}

template <typename scalar_t, int kCols>
__global__ void sample_block_gumbel_argmax_indexed_kernel_fixed_cols(
    const scalar_t* __restrict__ unique_logr,
    const int64_t* __restrict__ row_index,
    const scalar_t* __restrict__ exponents,
    float* __restrict__ out_scores,
    int64_t* __restrict__ out_indices,
    int rows,
    uint64_t seed) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  int64_t src_row = row_index[row];
  float local_best = -FLT_MAX;
  int local_best_idx = 0;
  float exponent = load_as_float(exponents + row);

  constexpr int kStride = kThreads;
  constexpr int kBaseStep = kVecWidth * kStride;
  constexpr int kNumSteps = kCols / kBaseStep;
  static_assert(kCols % kBaseStep == 0, "fixed-width sampler expects a multiple of 4 * kThreads");

  int base = static_cast<int>(src_row) * kCols;
  int col = threadIdx.x;

#pragma unroll
  for (int step = 0; step < kNumSteps; ++step) {
    int offset = col + step * kBaseStep;

    float u0 = uniform_from_counter(seed, row, offset);
    float u1 = uniform_from_counter(seed, row, offset + kStride);
    float u2 = uniform_from_counter(seed, row, offset + 2 * kStride);
    float u3 = uniform_from_counter(seed, row, offset + 3 * kStride);

    float s0 = exponent * load_as_float(unique_logr + base + offset) + (-logf(-logf(u0)));
    float s1 = exponent * load_as_float(unique_logr + base + offset + kStride) + (-logf(-logf(u1)));
    float s2 = exponent * load_as_float(unique_logr + base + offset + 2 * kStride) + (-logf(-logf(u2)));
    float s3 = exponent * load_as_float(unique_logr + base + offset + 3 * kStride) + (-logf(-logf(u3)));

    update_best(s0, offset, local_best, local_best_idx);
    update_best(s1, offset + kStride, local_best, local_best_idx);
    update_best(s2, offset + 2 * kStride, local_best, local_best_idx);
    update_best(s3, offset + 3 * kStride, local_best, local_best_idx);
  }

  int lane = threadIdx.x & (kWarpSize - 1);
  int warp_id = threadIdx.x / kWarpSize;

  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    float other_score = __shfl_down_sync(0xffffffff, local_best, offset);
    int other_index = __shfl_down_sync(0xffffffff, local_best_idx, offset);
    if (other_score > local_best) {
      local_best = other_score;
      local_best_idx = other_index;
    }
  }

  __shared__ float shared_scores[kThreads / kWarpSize];
  __shared__ int shared_indices[kThreads / kWarpSize];

  if (lane == 0) {
    shared_scores[warp_id] = local_best;
    shared_indices[warp_id] = local_best_idx;
  }
  __syncthreads();

  if (warp_id == 0) {
    local_best = lane < (kThreads / kWarpSize) ? shared_scores[lane] : -FLT_MAX;
    local_best_idx = lane < (kThreads / kWarpSize) ? shared_indices[lane] : 0;

    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      float other_score = __shfl_down_sync(0xffffffff, local_best, offset);
      int other_index = __shfl_down_sync(0xffffffff, local_best_idx, offset);
      if (other_score > local_best) {
        local_best = other_score;
        local_best_idx = other_index;
      }
    }

    if (lane == 0) {
      out_scores[row] = local_best;
      out_indices[row] = static_cast<int64_t>(local_best_idx);
    }
  }
}

template <typename scalar_t>
void launch_sample_block_gumbel_argmax_indexed_kernel(
    const scalar_t* unique_logr,
    const int64_t* row_index,
    const scalar_t* exponents,
    float* out_scores,
    int64_t* out_indices,
    int rows,
    int cols,
    uint64_t seed,
    cudaStream_t stream) {
  switch (cols) {
    case 1024:
      sample_block_gumbel_argmax_indexed_kernel_fixed_cols<scalar_t, 1024><<<rows, kThreads, 0, stream>>>(
          unique_logr, row_index, exponents, out_scores, out_indices, rows, seed);
      return;
    case 2048:
      sample_block_gumbel_argmax_indexed_kernel_fixed_cols<scalar_t, 2048><<<rows, kThreads, 0, stream>>>(
          unique_logr, row_index, exponents, out_scores, out_indices, rows, seed);
      return;
    case 4096:
      sample_block_gumbel_argmax_indexed_kernel_fixed_cols<scalar_t, 4096><<<rows, kThreads, 0, stream>>>(
          unique_logr, row_index, exponents, out_scores, out_indices, rows, seed);
      return;
    default:
      sample_block_gumbel_argmax_indexed_kernel_generic<scalar_t><<<rows, kThreads, 0, stream>>>(
          unique_logr, row_index, exponents, out_scores, out_indices, rows, cols, seed);
      return;
  }
}

std::vector<torch::Tensor> sample_block_gumbel_argmax_indexed_cuda(
    const torch::Tensor& unique_logr,
    const torch::Tensor& row_index,
    const torch::Tensor& chunk_exp,
    int64_t seed) {
  TORCH_CHECK(unique_logr.is_cuda(), "unique_logr must be CUDA");
  TORCH_CHECK(row_index.is_cuda(), "row_index must be CUDA");
  TORCH_CHECK(chunk_exp.is_cuda(), "chunk_exp must be CUDA");

  auto logr = unique_logr.contiguous();
  auto rows_idx = row_index.contiguous();
  auto exponents = chunk_exp.contiguous();
  auto rows = static_cast<int>(rows_idx.size(0));
  auto cols = static_cast<int>(logr.size(1));

  auto out_scores = torch::empty({rows}, logr.options().dtype(torch::kFloat32));
  auto out_indices = torch::empty({rows}, logr.options().dtype(torch::kInt64));
  auto stream = at::cuda::getDefaultCUDAStream();

  switch (logr.scalar_type()) {
    case at::ScalarType::Float:
      launch_sample_block_gumbel_argmax_indexed_kernel<float>(
          logr.data_ptr<float>(),
          rows_idx.data_ptr<int64_t>(),
          exponents.data_ptr<float>(),
          out_scores.data_ptr<float>(),
          out_indices.data_ptr<int64_t>(),
          rows,
          cols,
          static_cast<uint64_t>(seed),
          stream);
      break;
    case at::ScalarType::Half:
      launch_sample_block_gumbel_argmax_indexed_kernel<c10::Half>(
          logr.data_ptr<c10::Half>(),
          rows_idx.data_ptr<int64_t>(),
          exponents.data_ptr<c10::Half>(),
          out_scores.data_ptr<float>(),
          out_indices.data_ptr<int64_t>(),
          rows,
          cols,
          static_cast<uint64_t>(seed),
          stream);
      break;
    case at::ScalarType::BFloat16:
      launch_sample_block_gumbel_argmax_indexed_kernel<c10::BFloat16>(
          logr.data_ptr<c10::BFloat16>(),
          rows_idx.data_ptr<int64_t>(),
          exponents.data_ptr<c10::BFloat16>(),
          out_scores.data_ptr<float>(),
          out_indices.data_ptr<int64_t>(),
          rows,
          cols,
          static_cast<uint64_t>(seed),
          stream);
      break;
    default:
      TORCH_CHECK(false, "Unsupported dtype for CUDA indexed block sampler: ", logr.scalar_type());
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_scores, out_indices};
}
