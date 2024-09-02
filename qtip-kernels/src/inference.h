#pragma once

#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


#define gpuErrchk(ans)      do { gpuAssert((ans), __FILE__, __LINE__); } while (false)

__host__ static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert[%s:%d]: %s\n", file, line, cudaGetErrorString(code));
        if (abort) exit(code);
    }
}


typedef union ditto {
    uint32_t u32;
    half2 f16x2;
} ditto;


typedef union ditto2 {
    unsigned long long ull;
    uint64_t u64;
    uint2 u32x2;
    float2 f32x2;
    ushort4 u16x4;
    uint32_t u32[2];
    half2 f16x2[2];
    half2 *ptr2f16x2;
} ditto2;


typedef union ditto4 {
    uint4 u32x4;
    uint32_t u32[4];
    float4 f32x4;
    half2 f16x2[4];
    uint16_t u16[8];
    float f32[4];
} ditto4;

template <uint32_t L, uint32_t S, uint32_t R, uint32_t V, uint32_t M, uint32_t N, uint32_t K>
__host__ static void decompress_matvec_ptr(
        float *__restrict__ out,                    // m-by-n
        const uint32_t *__restrict__ compressed,    // m-by-k
        const half2 * __restrict__ x,               // k-by-n
        const half2 * __restrict__ codebook,
        CUstream_st *stream
);
