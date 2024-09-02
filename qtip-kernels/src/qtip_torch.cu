#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <torch/types.h>
#include <torch/extension.h>
#include "inference.cu"

using namespace torch::indexing;

template <uint32_t L, uint32_t S, uint32_t R, uint32_t V, uint32_t M, uint32_t N, uint32_t K>
__host__ static void decompress_matvec(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    CHECK_INPUT(out);
    TORCH_CHECK(out.dim() == 2);
    TORCH_CHECK(out.scalar_type() == torch::kFloat32);

    CHECK_INPUT(compressed);
    TORCH_CHECK(compressed.dim() == 1);
    TORCH_CHECK(compressed.scalar_type() == torch::kInt32);

    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2);
    TORCH_CHECK(x.scalar_type() == torch::kFloat16);

    CHECK_INPUT(codebook);
    TORCH_CHECK(codebook.dim() == 1);
    TORCH_CHECK(codebook.scalar_type() == torch::kFloat16);

    size_t m = out.size(0);
    size_t n = out.size(1);
    size_t k = x.size(0);

    TORCH_CHECK(m == M);
    //TORCH_CHECK(n == N);
    TORCH_CHECK(k == K);
    TORCH_CHECK(compressed.numel() * 32 == R * m * k);
    TORCH_CHECK(x.size(1) == n);
    TORCH_CHECK(codebook.size(0) == 1<<(S+V));

    decompress_matvec_ptr<L, S, R, V, M, N, K>(
            reinterpret_cast<float *>(out.data_ptr<float>()),
            reinterpret_cast<const uint32_t *>(compressed.data_ptr<int32_t>()),
            reinterpret_cast<const half2 *>(x.data_ptr<c10::Half>()),
            reinterpret_cast<const half2 *>(codebook.data_ptr<c10::Half>()),
            at::cuda::getCurrentCUDAStream()
    );
}


__host__ extern void decompress_matvec_16_9_2_1_256_1_256(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 2U, 1U, 256U, 1U, 256U>(out, compressed, x, codebook);
}

__host__ extern void decompress_matvec_16_9_2_1_4096_1_4096(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 2U, 1U, 4096U, 1U, 4096U>(out, compressed, x, codebook);
}


__host__ extern void decompress_matvec_16_9_2_1_4096_1_11008(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 2U, 1U, 4096U, 1U, 11008U>(out, compressed, x, codebook);
}


__host__ extern void decompress_matvec_16_9_2_1_11008_1_4096(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 2U, 1U, 11008, 1U, 4096U>(out, compressed, x, codebook);
}


__host__ extern void decompress_matvec_16_9_2_1_12288_1_4096(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 2U, 1U, 12288, 1U, 4096U>(out, compressed, x, codebook);
}

__host__ extern void decompress_matvec_16_9_2_1_22016_1_4096(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 2U, 1U, 22016, 1U, 4096U>(out, compressed, x, codebook);
}



__host__ extern void decompress_matvec_16_9_2_1_8192_1_8192(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 2U, 1U, 8192U, 1U, 8192U>(out, compressed, x, codebook);
}

__host__ extern void decompress_matvec_16_9_2_1_10240_1_8192(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 2U, 1U, 10240U, 1U, 8192U>(out, compressed, x, codebook);
}

__host__ extern void decompress_matvec_16_9_3_1_10240_1_8192(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 3U, 1U, 10240U, 1U, 8192U>(out, compressed, x, codebook);
}

__host__ extern void decompress_matvec_16_9_4_1_10240_1_8192(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 4U, 1U, 10240U, 1U, 8192U>(out, compressed, x, codebook);
}


__host__ extern void decompress_matvec_16_9_2_1_57344_1_8192(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 2U, 1U, 57344U, 1U, 8192U>(out, compressed, x, codebook);
}

__host__ extern void decompress_matvec_16_9_3_1_57344_1_8192(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 3U, 1U, 57344U, 1U, 8192U>(out, compressed, x, codebook);
}

__host__ extern void decompress_matvec_16_9_4_1_57344_1_8192(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 4U, 1U, 57344U, 1U, 8192U>(out, compressed, x, codebook);
}



__host__ extern void decompress_matvec_16_9_2_1_8192_1_1024(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 2U, 1U, 8192U, 1U, 1024U>(out, compressed, x, codebook);
}


__host__ extern void decompress_matvec_16_9_2_1_8192_1_28672(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 2U, 1U, 8192U, 1U, 28672U>(out, compressed, x, codebook);
}


__host__ extern void decompress_matvec_16_9_2_1_28672_1_8192(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 2U, 1U, 28672U, 1U, 8192U>(out, compressed, x, codebook);
}

__host__ extern void decompress_matvec_16_9_2_1_1024_1_8192(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 2U, 1U, 1024, 1U, 8192U>(out, compressed, x, codebook);
}


__host__ extern void decompress_matvec_16_9_3_1_256_1_256(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 3U, 1U, 256U, 1U, 256U>(out, compressed, x, codebook);
}

__host__ extern void decompress_matvec_16_9_3_1_4096_1_4096(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 3U, 1U, 4096U, 1U, 4096U>(out, compressed, x, codebook);
}


__host__ extern void decompress_matvec_16_9_3_1_4096_1_11008(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 3U, 1U, 4096U, 1U, 11008U>(out, compressed, x, codebook);
}


__host__ extern void decompress_matvec_16_9_3_1_11008_1_4096(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 3U, 1U, 11008, 1U, 4096U>(out, compressed, x, codebook);
}


__host__ extern void decompress_matvec_16_9_3_1_12288_1_4096(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 3U, 1U, 12288, 1U, 4096U>(out, compressed, x, codebook);
}

__host__ extern void decompress_matvec_16_9_3_1_22016_1_4096(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 3U, 1U, 22016, 1U, 4096U>(out, compressed, x, codebook);
}

__host__ extern void decompress_matvec_16_9_3_1_8192_1_8192(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 3U, 1U, 8192U, 1U, 8192U>(out, compressed, x, codebook);
}


__host__ extern void decompress_matvec_16_9_3_1_8192_1_1024(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 3U, 1U, 8192U, 1U, 1024U>(out, compressed, x, codebook);
}


__host__ extern void decompress_matvec_16_9_3_1_8192_1_28672(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 3U, 1U, 8192U, 1U, 28672U>(out, compressed, x, codebook);
}


__host__ extern void decompress_matvec_16_9_3_1_28672_1_8192(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 3U, 1U, 28672U, 1U, 8192U>(out, compressed, x, codebook);
}

__host__ extern void decompress_matvec_16_9_3_1_1024_1_8192(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 3U, 1U, 1024, 1U, 8192U>(out, compressed, x, codebook);
}



__host__ extern void decompress_matvec_16_9_4_1_256_1_256(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 4U, 1U, 256U, 1U, 256U>(out, compressed, x, codebook);
}

__host__ extern void decompress_matvec_16_9_4_1_4096_1_4096(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 4U, 1U, 4096U, 1U, 4096U>(out, compressed, x, codebook);
}


__host__ extern void decompress_matvec_16_9_4_1_4096_1_11008(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 4U, 1U, 4096U, 1U, 11008U>(out, compressed, x, codebook);
}


__host__ extern void decompress_matvec_16_9_4_1_11008_1_4096(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 4U, 1U, 11008, 1U, 4096U>(out, compressed, x, codebook);
}


__host__ extern void decompress_matvec_16_9_4_1_12288_1_4096(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 4U, 1U, 12288, 1U, 4096U>(out, compressed, x, codebook);
}

__host__ extern void decompress_matvec_16_9_4_1_22016_1_4096(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 4U, 1U, 22016, 1U, 4096U>(out, compressed, x, codebook);
}

__host__ extern void decompress_matvec_16_9_4_1_8192_1_8192(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 4U, 1U, 8192U, 1U, 8192U>(out, compressed, x, codebook);
}


__host__ extern void decompress_matvec_16_9_4_1_8192_1_1024(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 4U, 1U, 8192U, 1U, 1024U>(out, compressed, x, codebook);
}


__host__ extern void decompress_matvec_16_9_4_1_8192_1_28672(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 4U, 1U, 8192U, 1U, 28672U>(out, compressed, x, codebook);
}


__host__ extern void decompress_matvec_16_9_4_1_28672_1_8192(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 4U, 1U, 28672U, 1U, 8192U>(out, compressed, x, codebook);
}

__host__ extern void decompress_matvec_16_9_4_1_1024_1_8192(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_matvec<16U, 9U, 4U, 1U, 1024, 1U, 8192U>(out, compressed, x, codebook);
}
