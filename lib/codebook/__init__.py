import torch
import qtip_kernels

kernels = [
    (4096, 1, 4096, 2),
    (4096, 1, 11008, 2),
    (11008, 1, 4096, 2),
    (12288, 1, 4096, 2),
    (22016, 1, 4096, 2),
    (8192, 1, 8192, 2),
    (10240, 1, 8192, 2),
    (10240, 1, 8192, 3),
    (10240, 1, 8192, 4),
    (57344, 1, 8192, 2),
    (57344, 1, 8192, 3),
    (57344, 1, 8192, 4),
    (8192, 1, 1024, 2),
    (8192, 1, 28672, 2),
    (28672, 1, 8192, 2),
    (1024, 1, 8192, 2),
    (4096, 1, 4096, 3),
    (4096, 1, 11008, 3),
    (11008, 1, 4096, 3),
    (12288, 1, 4096, 3),
    (22016, 1, 4096, 3),
    (8192, 1, 8192, 3),
    (8192, 1, 1024, 3),
    (8192, 1, 28672, 3),
    (28672, 1, 8192, 3),
    (1024, 1, 8192, 3),
    (4096, 1, 4096, 4),
    (4096, 1, 11008, 4),
    (11008, 1, 4096, 4),
    (12288, 1, 4096, 4),
    (22016, 1, 4096, 4),
    (8192, 1, 8192, 4),
    (8192, 1, 1024, 4),
    (8192, 1, 28672, 4),
    (28672, 1, 8192, 4),
    (1024, 1, 8192, 4),
]

kdict = {}

for m, n, k, bitrate in kernels:
    torch.library.define(
        f"quip_lib::decompress_matvec_qtip_{m}_{n}_{k}_{bitrate}",
        "(Tensor compressed, Tensor x, Tensor codebook) -> Tensor")

    name = f"decompress_matvec_qtip_{m}_{n}_{k}_{bitrate}"
    kernel_name = f"qtip_kernels.decompress_matvec_16_9_{bitrate}_1_{m}_{n}_{k}"
    exec(f"""\
@torch.library.register_fake("quip_lib::{name}")
def {name}_abstract(
        compressed: torch.Tensor,
        x: torch.Tensor,
        codebook: torch.Tensor) -> torch.Tensor:
    return torch.zeros(1, {m}, dtype=torch.float32, device=x.device)

@torch.library.impl("quip_lib::{name}", "cuda")
def {name}_cuda(
        compressed: torch.Tensor,
        x: torch.Tensor,
        codebook: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(({m}, 1), dtype=torch.float32, device="cuda")
    {kernel_name}(out, compressed.reshape(-1).view(torch.int32), x.T.to(torch.float16), codebook.reshape(-1))
    return out.T
    """)

