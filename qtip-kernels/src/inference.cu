#include <cstdio>
#include <cassert>
#include <climits>

#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <mma.h>
#include <c10/cuda/CUDAStream.h>

#include "inference.h"

using namespace nvcuda;


#define CHECK_CUDA(x)           TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)     TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)          do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while (false)

#define BLOCKS_PER_SM 1
#define MMA_M                   16
#define MMA_N                   8
#define MMA_K                   16

#define BLOCK_COUNT             128
//#define MAX_THREADS_PER_SM      2048
#define WARP_SIZE               32
#define BLOCK_SIZE              1024
#define WARPS_PER_BLOCK         (BLOCK_SIZE/WARP_SIZE)

#define PREFETCHW               4
#define PREFETCHX               4
#define BLOCKS_PER_SM           1

#define FULL_MASK               0xFFFFFFFFU

__inline__ __device__ uint32_t ld_cs(const uint32_t* p)
{
    uint32_t out;
    asm("ld.global.cs.u32 %0, [%1];" : "=r"(out) : "l"(p));
    return out;
}

__inline__ __device__ uint2 ld_cs(const uint2* p)
{
    uint2 out;
    asm("ld.global.cs.v2.u32 {%0, %1}, [%2];" : "=r"(out.x), "=r"(out.y) : "l"(p));
    //asm("ld.weak.global.cs.L2::256B.v2.u32 {%0, %1}, [%2];" : "=r"(out.x), "=r"(out.y) : "l"(p));
    // the compiler doesn't know how to infer load(p) and load(p+4096) from loop unrolling with this :(
    return out;
}
__inline__ __device__ uint3 ld_cs(const uint3* p)
{
    uint3 out;
    asm("ld.global.cs.u32 %0, [%1];" : "=r"(out.x) : "l"(p));
    asm("ld.global.cs.u32 %0, [%1+4];" : "=r"(out.y) : "l"(p));
    asm("ld.global.cs.u32 %0, [%1+8];" : "=r"(out.z) : "l"(p));
    return out;
}
__inline__ __device__ uint4 ld_cs(const uint4* p)
{
    uint4 out;
    asm("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w) : "l"(p));
    return out;
}
__inline__ __device__ uint2 ld_x(const uint32_t* p, uint32_t x_idx, int subki)
{
    uint2 out;
    // the indexing is written as int32 math instead of lsu constant offset because
    // apparently using lsu offset adds lots of MIO pressure!
    if (subki == 0) {
        asm("ld.global.L1::evict_last.u32 %0, [%1];" : "=r"(out.x) : "l"(p+x_idx));
        asm("ld.global.L1::evict_last.u32 %0, [%1];" : "=r"(out.y) : "l"(p+(x_idx+4)));
    } else {
        asm("ld.global.L1::evict_last.u32 %0, [%1];" : "=r"(out.x) : "l"(p+(x_idx+8)));
        asm("ld.global.L1::evict_last.u32 %0, [%1];" : "=r"(out.y) : "l"(p+(x_idx+12)));
    }
    return out;
}
__inline__ __device__ uint32_t ld_x(const uint32_t* p)
{
    uint32_t out;
    asm("ld.global.L1::evict_last.u32 %0, [%1];" : "=r"(out) : "l"(p));
    return out;
}

__inline__ __device__ void prefetch(uint32_t *a){
    asm("prefetch.global.L1 [%0];"::"l"(a));
}

#define LD_CS
template <uint32_t R>
__device__ inline void load_reg_cs(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
    if constexpr(R == 2) {
#ifdef LD_CS
        ditto2 reg_load = {.u32x2 = ld_cs((uint2 *) &compressed[weight_idx])};
#else
        ditto2 reg_load = {.u16x4 = *((ushort4 * )(compressed + weight_idx))};
#endif
        uint32_t next1 = __shfl_sync(FULL_MASK, reg_load.u32x2.x, laneId + 1);
        uint32_t next2 = __shfl_sync(FULL_MASK, reg_load.u32x2.y, laneId + 1);
        reg_cs_next.x = __byte_perm(next1, reg_load.u32x2.x, 0x5410);
        reg_cs_next.y = __byte_perm(next1, reg_load.u32x2.x, 0x7632);
        reg_cs_next.z = __byte_perm(next2, reg_load.u32x2.y, 0x5410);
        reg_cs_next.w = __byte_perm(next2, reg_load.u32x2.y, 0x7632);
    } else if constexpr(R == 3) {
#ifdef LD_CS
        uint3 reg_load = ld_cs((uint3 *) &compressed[weight_idx]);
        uint32_t reg_load1 = reg_load.x, reg_load2 = reg_load.y, reg_load3 = reg_load.z;
#else
        uint32_t reg_load1 = *((uint32_t *) &compressed[weight_idx]);
        uint32_t reg_load2 = *((uint32_t *) &compressed[weight_idx + 2]);
        uint32_t reg_load3 = *((uint32_t *) &compressed[weight_idx + 4]);
#endif

        uint32_t reg_24_1 = reg_load1 & 0xffffff;
        uint32_t reg_24_2 = ((reg_load1 >> 24) | (reg_load2 << 8)) & 0xffffff;
        uint32_t reg_24_3 = ((reg_load2 >> 16) | (reg_load3 << 16)) & 0xffffff;
        uint32_t reg_24_4 = (reg_load3 >> 8) & 0xffffff;

        // send high 16 bits to prev thread
        uint32_t pack1 = (reg_24_1 >> 8) | ((reg_24_2 << 8) & 0xffff0000);
        uint32_t pack3 = (reg_24_3 >> 8) | ((reg_24_4 << 8) & 0xffff0000);

        // receive high 16 bits from next thread
        uint32_t next1 = __shfl_sync(FULL_MASK, pack1, laneId + 1);
        uint32_t next3 = __shfl_sync(FULL_MASK, pack3, laneId + 1);

        reg_cs_next.x = __byte_perm(next1, reg_24_1, 0x6541);
        reg_cs_next.y = __byte_perm(next1, reg_24_2, 0x6543);
        reg_cs_next.z = __byte_perm(next3, reg_24_3, 0x6541);
        reg_cs_next.w = __byte_perm(next3, reg_24_4, 0x6543);

        reg_cs2_next.x = ((next1 >> 6) & 0b11'1111'1111) | (reg_24_1 << 10);
        reg_cs2_next.y = ((next1 >> (6 + 16) & 0b11'1111'1111)) | (reg_24_2 << 10);
        reg_cs2_next.z = ((next3 >> 6) & 0b11'1111'1111) | (reg_24_3 << 10);
        reg_cs2_next.w = ((next3 >> (6 + 16) & 0b11'1111'1111)) | (reg_24_4 << 10);
    } else if constexpr(R == 4) {
#ifdef LD_CS
        uint4 reg_load = ld_cs((uint4 *) &compressed[weight_idx]);
#else
        uint4 reg_load = *((uint4 *) &compressed[weight_idx]);
#endif
        uint32_t reg_load1 = reg_load.x, reg_load2 = reg_load.y, reg_load3 = reg_load.z, reg_load4 = reg_load.w;

        // send high 16 bits to prev thread
        uint32_t pack1 = (reg_load1 >> 16) | (reg_load2 & 0xffff0000);
        uint32_t pack3 = (reg_load3 >> 16) | (reg_load4 & 0xffff0000);

        uint32_t next1 = __shfl_sync(FULL_MASK, pack1, laneId + 1);
        uint32_t next3 = __shfl_sync(FULL_MASK, pack3, laneId + 1);

        reg_cs_next.x = reg_load1;
        reg_cs_next.y = reg_load2;
        reg_cs_next.z = reg_load3;
        reg_cs_next.w = reg_load4;

        reg_cs2_next.x = __byte_perm(next1, reg_load1, 0x0041);
        reg_cs2_next.y = __byte_perm(next1, reg_load2, 0x0043);
        reg_cs2_next.z = __byte_perm(next3, reg_load3, 0x0041);
        reg_cs2_next.w = __byte_perm(next3, reg_load4, 0x0043);
    }

}

template <uint32_t L, uint32_t S, uint32_t R, uint32_t V, uint32_t M, uint32_t N, uint32_t K>
__global__ static void
__launch_bounds__(BLOCK_SIZE, 1)
kernel_decompress_matvec(
    float *__restrict__ out,
    const uint32_t *__restrict__ compressed,
    const half2 *__restrict__ x,
    const half2 *__restrict__ codebook
) {
        // ** load codebook **
    extern __shared__ __align__(1<<(5+V+1)) half2 smem_codebook[];

    // ** cursed indexing math **

    uint32_t threadId = threadIdx.x;
    uint32_t laneId = threadIdx.x % WARP_SIZE;
    uint32_t warpId = threadId / WARP_SIZE;
    uint32_t blockId = blockIdx.x;

    constexpr uint32_t tileCountM = M / MMA_M;
    constexpr uint32_t tileCountK = K / MMA_K;

    constexpr uint32_t warps_per_block = BLOCK_SIZE / WARP_SIZE;

#define ROUND_UP(a, b) ((a + b - 1) / b)

    static_assert (tileCountM % 2 == 0);
    constexpr uint32_t m_per_block = ROUND_UP(tileCountM, (2 * BLOCK_COUNT));
    // tiles are iterated along k in groups of 2
    //static_assert (tileCountK >= warps_per_block * 2);
    constexpr uint32_t k_per_block = tileCountK / (warps_per_block * 4) * 2;
    // we sync at ki%4==0, make sure this is safe
    //constexpr bool enable_kim4_sync = !(M == 4096 && K==4096) && (tileCountK % (warps_per_block * 2) == 0 || k_per_block % 4 != 0);
    // some warps have more k tiles
    static_assert((tileCountK % (warps_per_block * 4)) % 4 == 0);
    uint32_t this_warp_k = (warpId < (tileCountK % (warps_per_block * 4)) / 4) ? k_per_block + 2 : k_per_block;

    constexpr uint32_t u16_per_compressed_tile = MMA_M * MMA_K * R / 16;
    static_assert((MMA_M * MMA_K * R) % 16 == 0);
    constexpr uint32_t f16x2_per_x_tile = MMA_K / 2;
    constexpr uint32_t f32_per_out_tile = MMA_M;

    uint32_t tileIdM = m_per_block * blockId;

    constexpr uint32_t weight_block = 4;
    constexpr uint32_t u16_per_tile_block = u16_per_compressed_tile * weight_block; // one tile block per warp at a time
    constexpr uint32_t weight_step = warps_per_block * u16_per_tile_block;
    constexpr uint32_t weight_row_step = tileCountK * u16_per_compressed_tile * 2;  // 2 rows of tiles



    for (uint32_t mi = 0; mi < m_per_block; mi+=1) {
        if (tileIdM * 2 >= tileCountM) return;
        // ** load weight, start loop **
        int weight_idx = tileIdM * weight_row_step + warpId * u16_per_tile_block * 2 + laneId * (u16_per_tile_block / WARP_SIZE);
        uint4 reg_cs_next = {};
        uint4 reg_cs2_next = {};
        load_reg_cs<R>((const uint16_t * __restrict__) compressed, weight_idx, laneId, reg_cs_next, reg_cs2_next);
        uint4 reg_cs;
        uint4 reg_cs2;

        // define acc
        float4 reg_p[2] = {};

#define LOAD_X_BUFFERED
#ifdef PERMUTE_K
        uint32_t x_idx = warpId * f16x4_per_x_tile*2 + laneId;
        uint32_t x_idx_step = warps_per_block * f16x4_per_x_tile * 2;
#else
#if !defined(LOAD_X_SHUFFLE) && !defined(LOAD_X_BUFFERED)
        uint32_t x_idx = warpId * f16x2_per_x_tile * 2 + laneId;  // every warp does 2 k tiles per iteration
        uint32_t x_idx_step = warps_per_block * f16x2_per_x_tile * 2;
#else
        uint32_t x_idx = warpId * f16x2_per_x_tile * 4 + laneId;  // every warp does 4 k tiles per iteration
        uint32_t x_idx_step = warps_per_block * f16x2_per_x_tile * 4;
#endif
#endif
        if (mi == 0) {
#define DO_LOAD_CODEBOOK
#ifdef DO_LOAD_CODEBOOK
            uint32_t my_cb_idx = threadIdx.x & 0x1ff;
            half2 my_codebook_element = codebook[my_cb_idx];
            for (uint32_t i = 0; i < 32; i+= 2) {
                smem_codebook[(my_cb_idx << 5)|(i ^ (threadIdx.x & 0x1f) ^ (threadIdx.x >> 9))] = my_codebook_element;
            }
            // for (uint32_t i = 0; i < 32; i+= 1) { assert(smem_codebook[(my_cb_idx << 5) + i] == my_codebook_element); }
            __syncthreads();
#endif
        }

        __shared__ ditto2 x_buf[2][BLOCK_SIZE / WARP_SIZE][4][4];
        uint32_t x_line;
#pragma unroll 4
        for (uint32_t ki = 0; ki < this_warp_k; ki += 1) {
            // load this 2x2 block of weight tiles
            if (ki + 1 != this_warp_k && ki % 2 == 1) weight_idx += weight_step * 2; // fixme: this costs 10GB/s
            reg_cs = reg_cs_next;
            reg_cs2 = reg_cs2_next;
            load_reg_cs<R>((const uint16_t * __restrict__) compressed, weight_idx + (1 - ki % 2) * u16_per_tile_block, laneId, reg_cs_next, reg_cs2_next);

#define LOAD_X
#ifdef LOAD_X
#ifdef LOAD_X_BUFFERED
            if (ki % 2 == 0) {
                __syncwarp();
                x_buf[0][warpId][laneId / 8][laneId % 4].u32[(laneId % 8) / 4] = ld_x(reinterpret_cast<const uint32_t *>(x) + x_idx);
                __syncwarp();
                x_idx += x_idx_step;
            }
#else
#ifdef LOAD_X_SHUFFLE
            if (ki % 2 == 0) {
                x_line = ld_x(((uint32_t *) x) + x_idx);
                x_idx += x_idx_step;
            }
#endif
#endif
#endif

#pragma unroll 2
            for (uint32_t subki = 0; subki < 2; subki += 1) {
                // load activations
                // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
                ditto2 reg_a;
#define LD_X
#ifdef LOAD_X
#ifdef LOAD_X_SHUFFLE
                uint32_t x_subki = (ki % 2 * 2 + subki);
                if (x_subki != 0) {
                    reg_a.u32x2.x = __shfl_sync(FULL_MASK, x_line, (laneId & 3) | (8 * x_subki));
                    reg_a.u32x2.y = __shfl_sync(FULL_MASK, x_line, (laneId & 3) | (4 | (8 * x_subki)));
                } else {
                    reg_a.u32x2.x = x_line;
                    reg_a.u32x2.y = __shfl_sync(FULL_MASK, x_line, (laneId & 3) | 4);
                }
#else
                if (laneId < 4) {
#ifdef LOAD_X_BUFFERED
                    reg_a.u32x2 = x_buf[0][warpId][ki % 2 * 2 + subki][laneId].u32x2;
#endif
                }
#endif
#endif

#pragma unroll 2
                for (uint32_t submi = 0; submi < 2; submi++) {
                    uint32_t reg_c, reg_c2;
                    if (submi == 0 && subki == 0) reg_c = reg_cs.x;
                    else if (submi == 1 && subki == 0) reg_c = reg_cs.y;
                    else if (submi == 0 && subki == 1) reg_c = reg_cs.z;
                    else if (submi == 1 && subki == 1) reg_c = reg_cs.w;
                    if (submi == 0 && subki == 0) reg_c2 = reg_cs2.x;
                    else if (submi == 1 && subki == 0) reg_c2 = reg_cs2.y;
                    else if (submi == 0 && subki == 1) reg_c2 = reg_cs2.z;
                    else if (submi == 1 && subki == 1) reg_c2 = reg_cs2.w;

                    // ** decode weights **

#define DO_MMA
#ifdef DO_MMA
                    // at R = 2, 16 bit -> 8 weights -> 4 half2
                    ditto4 reg_w;
                    #pragma unroll
                    for (uint32_t j = 0; j < 4; j += 1) {
#define DO_LOOKUP
#ifndef DO_LOOKUP
                        reg_w.u32[0] = reg_c;
                        reg_w.u32[1] = reg_c;
                        reg_w.u32[2] = reg_c;
                        reg_w.u32[3] = reg_c;
#else
                        uint32_t idx;
                        if constexpr(R == 2) {
                            idx = reg_c >> (4 * (4-j));
                        } else if constexpr(R == 3) {
                            idx = (j < 3) ? (reg_c >> (6 * (2-j) + 4)) : reg_c2;
                        } else if constexpr(R == 4) {
                            idx = (j < 3) ? (reg_c >> (8 * (2-j))) : reg_c2;
                        }

                        static_assert(L==16);
                        idx = idx * (idx+1);
                        uint32_t masked_idx = ((idx & 0b0111111111000000) | (laneId << 1)); // this /2 will not be elided automatically
                        __builtin_assume(masked_idx % 2 == 0);
#define DO_LUT
#ifdef DO_LUT
                        reg_w.f16x2[j] = smem_codebook[masked_idx/2];
                        //asm("ld.shared.u32 %0, [%1];" : "=r"(reg_w.u32[j]) : "r"((masked_idx * 2 + (uint16_t) smem_codebook)));
#endif
                        // sign flip
                        uint32_t selector = 0b00000000'00000000'10000000'00000000;
                        reg_w.u32[j] = reg_w.u32[j] ^ (selector & idx);
#endif
                    }

                    //printf("%u: %f %f %f %f\n", tileIdK, __half2float(reg_w.f16x2[0].x),__half2float(reg_w.f16x2[0].y), __half2float(reg_w.f16x2[1].x),__half2float(reg_w.f16x2[1].y));
                    asm volatile (
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                            " {%0, %1, %2, %3},"
                            " {%4, %5, %6, %7},"
                            " {%8, %9},"
                            " {%0, %1, %2, %3};"
                            : "+f"(reg_p[submi].x), "+f"(reg_p[submi].y), "+f"(reg_p[submi].z), "+f"(reg_p[submi].w)
                            :  "r"(reg_w.u32[0]), "r"(reg_w.u32[1]), "r"(reg_w.u32[2]), "r"(reg_w.u32[3]),
                            "r"(reg_a.u32[0]), "r"(reg_a.u32[1])
                    );
                    //printf("%u %u %u: %f %f %f %f\n", tileIdM, warpId, laneId, reg_p.x, reg_p.y, reg_p.z, reg_p.w);
#else
#ifdef LOAD_X
                    reg_p.x += reg_c * reg_a.u32[0];
                    reg_p.y += reg_c * reg_a.u32[1];
                    reg_p.z += reg_c * reg_a.u32[0];
                    reg_p.w += reg_c * reg_a.u32[1];
#else
                    reg_p.x += reg_c;
                    reg_p.y += reg_c;
                    reg_p.z += reg_c;
                    reg_p.w += reg_c;
#endif
#endif
                }

            }
            //if constexpr(enable_kim4_sync) {if (ki % 4 == 0) __syncthreads();} // slower with 7b even with this if constexpr thing fsr
#define PREFETCH_X
#ifdef LOAD_X
#ifdef PREFETCH_X
            if (ki % 2 == 0) {
                prefetch((uint32_t *) (x + x_idx + x_idx_step*4));
            }
#endif
#endif
        }

        __shared__ __align__(16 * 8*32) float reduce_gather[BLOCK_SIZE / WARP_SIZE][2][16];
        if (laneId % 4 == 0) {
            for (int pi = 0; pi < 2; pi++) {
                reduce_gather[warpId][pi][laneId / 4] = reg_p[pi].x;
                reduce_gather[warpId][pi][laneId / 4 + 8] = reg_p[pi].z;
            }
        }
        __syncthreads();
        float reduced = 0.0;
        if (warpId < 1) {
            int pi = laneId / 16;
            for (int warpi = 0; warpi < BLOCK_SIZE / WARP_SIZE; warpi++) {
                reduced += reduce_gather[warpi][pi][laneId % 16];
            }

            // TODO: https://forums.developer.nvidia.com/t/can-float4-be-used-for-atomicadd-efficiently/215692
            // two rows at a time
            float *out_tile = out + (tileIdM * 2) * f32_per_out_tile;
            out_tile[laneId] = reduced;
        }
        if constexpr(m_per_block > 1) __syncthreads();
        tileIdM += 1;
    }
}


// L: shift register bit-width
// S: codebook index bit-width
// R: bits per weight
// V: log2(VQ dimension)
template <uint32_t L, uint32_t S, uint32_t R, uint32_t V, uint32_t M, uint32_t N, uint32_t K>
__host__ static void decompress_matvec_ptr(
    float *__restrict__ out,                    // m-by-n
    const uint32_t *__restrict__ compressed,    // m-by-k
    const half2 * __restrict__ x,               // k-by-n
    const half2 * __restrict__ codebook,
    CUstream_st *stream
) {
    static_assert(L <= 16, "Shift register should fit in uint16_t");
    static_assert(L >= S, "Shift register state space must not be smaller than codebook size");
    static_assert(S + V >= 3, "Codebook must have at least eight float16 elements as smem copy operates on uint4");
    static_assert(S + 5 + V + 1 <= 16, "We can only use 64 KiB shared memory"); // warpSize is 1<<5, sizeof(half) is 1<<1
    static_assert(R == 2 || R == 3 || R == 4, "Quantization rate = 2 or 3 or 4 for now");
    static_assert(V == 1, "Always quantize two weights at a time");

    static_assert(M % MMA_M == 0);
    static_assert(N == 1);
    static_assert(K % MMA_K == 0);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    //assert(deviceProp.multiProcessorCount == SM_COUNT);
    //assert(deviceProp.maxThreadsPerMultiProcessor == MAX_THREADS_PER_SM);
    assert(deviceProp.warpSize == WARP_SIZE);

    //static_assert(MAX_THREADS_PER_SM % BLOCK_SIZE == 0);
    static_assert(BLOCK_SIZE % WARP_SIZE == 0);

    constexpr uint32_t gridSize = BLOCK_COUNT;
    constexpr uint32_t blockSize = BLOCK_SIZE;
    constexpr uint32_t smemCodebookSize = 1<<(S+5+V+1);
    constexpr uint32_t smemReduceGatherSize = 2 * BLOCK_SIZE * sizeof(float4);
    cudaFuncSetAttribute(kernel_decompress_matvec<L, S, R, V, M, N, K>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smemCodebookSize);

    kernel_decompress_matvec<L, S, R, V, M, N, K><<<gridSize, blockSize, smemCodebookSize, stream>>>(out, compressed, x, codebook);
    
    gpuErrchk(cudaPeekAtLastError());
}

