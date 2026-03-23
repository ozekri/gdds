#include <torch/extension.h>

#include <cstdint>
#include <vector>

std::vector<torch::Tensor> sample_block_gumbel_argmax_cpu(
    const torch::Tensor& chunk_logr,
    const torch::Tensor& chunk_exp,
    int64_t /* seed */) {
  auto logr = chunk_logr.to(torch::kFloat32);
  auto exponents = chunk_exp.to(torch::kFloat32).unsqueeze(1);
  auto uniform = torch::rand_like(logr).clamp_min_(1.0e-7);
  auto gumbel = -(-uniform.log()).log();
  auto scores = exponents * logr + gumbel;
  auto result = scores.max(1);
  return {std::get<0>(result), std::get<1>(result)};
}

std::vector<torch::Tensor> sample_block_gumbel_argmax_indexed_cpu(
    const torch::Tensor& unique_logr,
    const torch::Tensor& row_index,
    const torch::Tensor& chunk_exp,
    int64_t seed) {
  auto gathered = unique_logr.index_select(0, row_index.to(torch::kLong));
  return sample_block_gumbel_argmax_cpu(gathered, chunk_exp, seed);
}

#ifdef WITH_CUDA
std::vector<torch::Tensor> sample_block_gumbel_argmax_cuda(
    const torch::Tensor& chunk_logr,
    const torch::Tensor& chunk_exp,
    int64_t seed);
std::vector<torch::Tensor> sample_block_gumbel_argmax_indexed_cuda(
    const torch::Tensor& unique_logr,
    const torch::Tensor& row_index,
    const torch::Tensor& chunk_exp,
    int64_t seed);
#endif


std::vector<torch::Tensor> sample_block_gumbel_argmax(
    const torch::Tensor& chunk_logr,
    const torch::Tensor& chunk_exp,
    int64_t seed) {
  TORCH_CHECK(chunk_logr.dim() == 2, "chunk_logr must be 2D");
  TORCH_CHECK(chunk_exp.dim() == 1, "chunk_exp must be 1D");
  TORCH_CHECK(chunk_logr.size(0) == chunk_exp.size(0), "row mismatch between chunk_logr and chunk_exp");
  TORCH_CHECK(chunk_logr.is_contiguous(), "chunk_logr must be contiguous");
  TORCH_CHECK(chunk_exp.is_contiguous(), "chunk_exp must be contiguous");

  if (chunk_logr.is_cuda()) {
#ifdef WITH_CUDA
    return sample_block_gumbel_argmax_cuda(chunk_logr, chunk_exp, seed);
#else
    TORCH_CHECK(false, "CUDA support is not available in this build");
#endif
  }

  return sample_block_gumbel_argmax_cpu(chunk_logr, chunk_exp, seed);
}

std::vector<torch::Tensor> sample_block_gumbel_argmax_indexed(
    const torch::Tensor& unique_logr,
    const torch::Tensor& row_index,
    const torch::Tensor& chunk_exp,
    int64_t seed) {
  TORCH_CHECK(unique_logr.dim() == 2, "unique_logr must be 2D");
  TORCH_CHECK(row_index.dim() == 1, "row_index must be 1D");
  TORCH_CHECK(chunk_exp.dim() == 1, "chunk_exp must be 1D");
  TORCH_CHECK(row_index.size(0) == chunk_exp.size(0), "row mismatch between row_index and chunk_exp");
  TORCH_CHECK(unique_logr.is_contiguous(), "unique_logr must be contiguous");
  TORCH_CHECK(row_index.is_contiguous(), "row_index must be contiguous");
  TORCH_CHECK(chunk_exp.is_contiguous(), "chunk_exp must be contiguous");

  if (unique_logr.is_cuda()) {
#ifdef WITH_CUDA
    return sample_block_gumbel_argmax_indexed_cuda(unique_logr, row_index, chunk_exp, seed);
#else
    TORCH_CHECK(false, "CUDA support is not available in this build");
#endif
  }

  return sample_block_gumbel_argmax_indexed_cpu(unique_logr, row_index, chunk_exp, seed);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "sample_block_gumbel_argmax",
      &sample_block_gumbel_argmax,
      "Sample exact Gumbel-max over a dense [B, V] block");
  m.def(
      "sample_block_gumbel_argmax_indexed",
      &sample_block_gumbel_argmax_indexed,
      "Sample exact Gumbel-max over a dense [U, V] block with row indices");
}
