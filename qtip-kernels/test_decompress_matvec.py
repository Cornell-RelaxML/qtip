import os

import qtip_kernels
import torch
from cuda import cuda

#from tinygrad import Device

kernels = {
    2: {
        (256, 1, 256): qtip_kernels.decompress_matvec_16_9_2_1_256_1_256,
        (4096, 1, 4096): qtip_kernels.decompress_matvec_16_9_2_1_4096_1_4096,
        (4096, 1, 11008): qtip_kernels.decompress_matvec_16_9_2_1_4096_1_11008,
        (11008, 1, 4096): qtip_kernels.decompress_matvec_16_9_2_1_11008_1_4096,
        (8192, 1, 8192): qtip_kernels.decompress_matvec_16_9_2_1_8192_1_8192,
        (1024, 1, 8192): qtip_kernels.decompress_matvec_16_9_2_1_1024_1_8192,
        (8192, 1, 28672): qtip_kernels.decompress_matvec_16_9_2_1_8192_1_28672,
        (28672, 1, 8192): qtip_kernels.decompress_matvec_16_9_2_1_28672_1_8192,
    },
    3: {
        (4096, 1, 4096): qtip_kernels.decompress_matvec_16_9_3_1_4096_1_4096,
        (4096, 1, 11008): qtip_kernels.decompress_matvec_16_9_3_1_4096_1_11008,
        (11008, 1, 4096): qtip_kernels.decompress_matvec_16_9_3_1_11008_1_4096,
        (8192, 1, 8192): qtip_kernels.decompress_matvec_16_9_3_1_8192_1_8192,
        (1024, 1, 8192): qtip_kernels.decompress_matvec_16_9_3_1_1024_1_8192,
        (8192, 1, 28672): qtip_kernels.decompress_matvec_16_9_3_1_8192_1_28672,
        (28672, 1, 8192): qtip_kernels.decompress_matvec_16_9_3_1_28672_1_8192,
    },
    4: {
        (4096, 1, 4096): qtip_kernels.decompress_matvec_16_9_4_1_4096_1_4096,
        (4096, 1, 11008): qtip_kernels.decompress_matvec_16_9_4_1_4096_1_11008,
        (11008, 1, 4096): qtip_kernels.decompress_matvec_16_9_4_1_11008_1_4096,
        (8192, 1, 8192): qtip_kernels.decompress_matvec_16_9_4_1_8192_1_8192,
        (1024, 1, 8192): qtip_kernels.decompress_matvec_16_9_4_1_1024_1_8192,
        (8192, 1, 28672): qtip_kernels.decompress_matvec_16_9_4_1_8192_1_28672,
        (28672, 1, 8192): qtip_kernels.decompress_matvec_16_9_4_1_28672_1_8192,
    },
}


#dev = Device[Device.DEFAULT]
def time_kernel(kernel):
    ITER = int(os.getenv("ITER", "100"))
    zero_buf = torch.empty(128 * (1024**2), dtype=torch.int8, device='cuda')
    # capture CUDA graph
    start_events = [cuda.cuEventCreate(0)[1] for _ in range(ITER)]
    end_events = [cuda.cuEventCreate(0)[1] for _ in range(ITER)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stream = torch.cuda.current_stream().cuda_stream
        cuda.cuEventRecordWithFlags(start_events[0], stream, 1)
        kernel()
        cuda.cuEventRecordWithFlags(end_events[0], stream, 1)
        zero_buf.zero_()

    elapsed_time_ms = 0.0
    for _ in range(ITER):
        graph.replay()
        torch.cuda.synchronize()
        #        dev.invalidate_caches()
        torch.cuda.synchronize()
        elapsed_time_ms += cuda.cuEventElapsedTime(start_events[0],
                                                   end_events[0])[1]

    #elapsed_time_ms = sum(cuda.cuEventElapsedTime(se, ee)[1] for se, ee in zip(start_events, end_events))
    return elapsed_time_ms / ITER


def time_kernel_dual(kernel1, kernel2):
    ITER = int(os.getenv("ITER", "100"))
    # capture CUDA graph
    start_events = [cuda.cuEventCreate(0)[1] for _ in range(ITER)]
    end_events = [cuda.cuEventCreate(0)[1] for _ in range(ITER)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stream = torch.cuda.current_stream().cuda_stream
        for i in range(ITER):
            cuda.cuEventRecordWithFlags(start_events[i], stream, 1)
            if i % 2 == 0:
                kernel1()
            else:
                kernel2()
            cuda.cuEventRecordWithFlags(end_events[i], stream, 1)

    graph.replay()
    torch.cuda.synchronize()

    elapsed_time_ms = sum(
        cuda.cuEventElapsedTime(se, ee)[1]
        for se, ee in zip(start_events, end_events))
    return elapsed_time_ms / ITER


def quipsharp_time(M, N, K):
    import quiptools_cuda
    x = torch.randn((K, ), dtype=torch.float16, device="cuda")
    x2 = torch.randn((K, ), dtype=torch.float16, device="cuda")
    Qidxs = torch.randint(0x7FFFFFFFFFFFFFFF, (M // 16, K // 64, 8, 4),
                          dtype=torch.int64,
                          device="cuda")
    Qidxs2 = torch.randint(0x7FFFFFFFFFFFFFFF, (M // 16, K // 64, 8, 4),
                           dtype=torch.int64,
                           device="cuda")
    codebook = torch.randint(0x7fffffff, (256, ),
                             dtype=torch.int32,
                             device="cuda")
    codebook2 = torch.randint(0x7fffffff, (256, ),
                              dtype=torch.int32,
                              device="cuda")

    elapsed_time_ms = time_kernel_dual(
        lambda: quiptools_cuda.decode_matvec_e8p(x, Qidxs, codebook),
        lambda: quiptools_cuda.decode_matvec_e8p(x2, Qidxs2, codebook2))
    gbps = (Qidxs.nbytes / (10**9)) / (elapsed_time_ms / 1000)
    print(f"quip# {(M, N, K)}: {elapsed_time_ms * 1000:.2f}us {gbps:.1f} GBps")


def time_qs_kernels():
    for M, N, K in kernels[2]:
        quipsharp_time(M, N, K)


def decompress_matvec_time(R, args1, args2):
    out, compressed, x, codebook = args1
    m, n = out.shape
    k, n = x.shape
    kernel = kernels[R][(m, n, k)]

    elapsed_time_ms = time_kernel_dual(lambda: kernel(*args1),
                                       lambda: kernel(*args2))

    gbps = (compressed.nbytes / (10**9)) / (elapsed_time_ms / 1000)
    print(
        f"{R}bit {(m, n, k)}: {elapsed_time_ms * 1000:.2f}us {gbps:.1f} GBps")


def decompress_matvec(R, out, compressed, x, codebook):
    m, n = out.shape
    k, n = x.shape
    kernel = kernels[R][(m, n, k)]
    kernel(out, compressed, x, codebook)


def prepare_arguments_sanity(L, S, R, V, m, n, k):
    out = torch.zeros((m, n), dtype=torch.float32,
                      device="cuda")  # we require zero-initialization
    # NOTE: all zero so no top bit flips
    compressed = torch.full((R * m * k // 32, ),
                            0,
                            dtype=torch.int32,
                            device="cuda")
    """
    compressed = torch.randint(torch.iinfo(torch.int32).min,
                               torch.iinfo(torch.int32).max+1,
                               (R * m * k // 32,),
                               dtype=torch.int32,
                               device="cpu").cuda()
                               """
    x = torch.ones((k, n), dtype=torch.float16, device="cpu").cuda()
    codebook = torch.full((1 << (S + V), ),
                          1 / k / n,
                          dtype=torch.float16,
                          device="cpu").cuda()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return out, compressed, x, codebook


def sanity_check(L, S, R, V):
    for m, n, k in kernels[R]:
        out, compressed, x, codebook = prepare_arguments_sanity(
            L, S, R, V, m, n, k)
        decompressed = torch.ones(m, k, dtype=torch.float16,
                                  device="cpu").cuda() / k / n
        decompress_matvec(R, out, compressed, x, codebook)
        print("sanity check", m, n, k,
              torch.sum(out).item(), "=",
              torch.sum(decompressed @ x).item())


def quantlut_sym(tlut, L, nbits):
    with torch.no_grad():
        lut = torch.arange(1 << L, device=tlut.device)
        lut = (lut + 1) * lut
        sflp = 1 - ((lut >> 15) & 1) * 2
        lut = (lut >> (16 - nbits - 1)) & ((1 << nbits) - 1)
    lut = tlut[lut]
    lut[:, 0] = lut[:, 0] * sflp
    return lut


def decode_compressed(L, S, R, V, m, k, compressed, codebook):
    if compressed.dtype != torch.uint16:
        compressed = compressed.view(torch.uint16)

    assert compressed.shape == (R * m * k // 16, )

    BLOCK_SIZE = 16 * 16

    BITS_PER_BLOCK = R * 16 * 16  # R bits * f16 mma tile A size

    compressed = (
        compressed.view(torch.uint8).reshape(m // 16 // 2, k // 16 // 2,
                                             BLOCK_SIZE // 8, 2, 2,
                                             R).permute(0, -2, 1, -3, 2, -1)
        # big endian across words, little endian within words ...
        .flip((-1, )).reshape(m // 16, k // 16, BITS_PER_BLOCK // 16, 2).flip(
            (-1, )).view(torch.uint16).reshape(m // 16, k // 16,
                                               BITS_PER_BLOCK // 16))
    '''
    # unswizzle interleaved blocks

    compressed = (compressed.reshape(m // 16 // 2, k // 16 // 2,
                                     BITS_PER_BLOCK // 16, 2, 2).permute(
                                         0, -1, 1, -2,
                                         2).reshape(m // 16, k // 16,
                                                    BITS_PER_BLOCK // 16))
    '''
    # decode block

    assert L == 16

    blocked = compressed.reshape(R * m * k // BITS_PER_BLOCK,
                                 BITS_PER_BLOCK // 16, 1)
    blocked_roll = torch.roll(blocked.cpu(), -1, -2).cuda()
    blocked32 = torch.cat((blocked_roll, blocked),
                          dim=-1).reshape(blocked.shape[0],
                                          -1).contiguous().view(torch.uint32)
    # blocked32 is 16bits[-1]||16bits[0] 16bits[0]||16bits[1] ... 16bits[-2]||16bits[-1]

    expanded32 = blocked32.reshape(*blocked32.shape,
                                   1).expand(*blocked32.shape,
                                             16).view(torch.int32)
    shifts = (torch.arange(0, 16, dtype=torch.int32, device="cuda")).to(
        torch.int32).reshape(1, 1, -1).expand(expanded32.shape)
    shifted = expanded32 >> (16 - shifts)
    indices = torch.bitwise_and(
        shifted.reshape(shifted.shape[0], -1)[:, 16 - L::R << V], (1 << L) - 1)

    # decode lut
    expanded_lut = quantlut_sym(codebook, L, S)
    mma_swizzled = expanded_lut[indices]

    # deswizzle m16n8k16 mma pattern
    decompressed = (mma_swizzled.reshape(m // 16, k // 16, 16, 16).reshape(
        m // 16, k // 16, 8, 4, 2, 2, 2).permute(0, -2, 2, 1, -3, 3,
                                                 -1).reshape(m, k))
    return decompressed


def prepare_arguments(L, S, R, V, m, n, k):
    out = torch.zeros((m, n), dtype=torch.float32,
                      device="cuda")  # we require zero-initialization
    #codebook = torch.full((1<<(S+V),), 1, dtype=torch.float16, device="cpu").cuda()
    codebook = (torch.randn(
        (1 << (S + V)), dtype=torch.float16, device="cpu").cuda() / 16).clamp(
            -1, 1)
    compressed = torch.randint(torch.iinfo(torch.int32).min,
                               torch.iinfo(torch.int32).max + 1,
                               (R * m * k // 32, ),
                               dtype=torch.int32,
                               device="cpu").cuda()
    x = (torch.randn(
        (k, n), dtype=torch.float16, device="cpu").cuda() / 16).clamp(-1, 1)
    #x = torch.zeros((k, n), dtype=torch.float16, device="cuda")
    #x[4,0] = 1.0
    x = x.contiguous()

    decompressed = decode_compressed(L, S, R, V, m, k, compressed,
                                     codebook.reshape(1 << S, 1 << V))

    return out, compressed, x, codebook, decompressed


def test_kernels(L, S, R, V):
    torch.set_printoptions(threshold=10_000)
    for m, n, k in kernels[R]:
        out, compressed, x, codebook, decompressed = prepare_arguments(
            L, S, R, V, m, n, k)
        if not os.getenv("TIMING"):
            decompress_matvec(R, out, compressed, x, codebook)
        else:
            decompress_matvec_time(R, (out, compressed, x, codebook),
                                   prepare_arguments(L, S, R, V, m, n, k)[:4])
        if not os.getenv("NOCHECK"):
            ref = (decompressed @ x)
            allclose = torch.allclose(out.half(), ref, atol=1e-5, rtol=0.01)
            if not allclose:
                print(torch.stack((ref[:16], out[:16]), dim=-1))
            try:
                torch.testing.assert_allclose(out.half(),
                                              ref,
                                              atol=1e-5,
                                              rtol=0.01)
            except:
                import traceback
                traceback.print_exc()
                exit()
            print("real test", m, n, k,
                  torch.sum(out).item(), "=",
                  torch.sum(decompressed @ x).item(), "allclose", allclose)


if __name__ == "__main__":
    torch.manual_seed(42)
    L, S, V = 16, 9, 1
    for R in range(2, 5):
        print(R)
        #sanity_check(L, S, R, V)
        test_kernels(L, S, R, V)
    if os.getenv("QS"): time_qs_kernels()
