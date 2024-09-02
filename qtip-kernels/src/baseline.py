import torch


def benchmark():
    m, n, k = 8192, 8, 8192
    print(f"{m = }, {n = }, {k = }")

    decompressed = torch.randn((m, k), dtype=torch.float16,
                               device="cpu").cuda()
    x = torch.randn((n, k), dtype=torch.float16, device="cpu").cuda()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    result = decompressed @ x.T
    dummy = torch.sum(result)
    print(dummy.item())


if __name__ == "__main__":
    torch.manual_seed(42)
    benchmark()
