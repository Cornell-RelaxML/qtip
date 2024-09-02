import math

import torch


def flat_to_sym(V, N):
    A = torch.zeros(N, N, dtype=V.dtype, device=V.device)
    idxs = torch.tril_indices(N, N, device=V.device)
    A[idxs.unbind()] = V
    A[idxs[1, :], idxs[0, :]] = V
    return A


def block_LDL(H, b, check_nan=True):
    n = H.shape[0]
    assert (n % b == 0)
    m = n // b

    try:
        L = torch.linalg.cholesky(H)
    except:
        return None

    DL = torch.diagonal(L.reshape(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)
    D = (DL @ DL.permute(0, 2, 1)).cpu()
    DL = torch.linalg.inv(DL)
    L = L.view(n, m, b)
    for i in range(m):
        L[:, i, :] = L[:, i, :] @ DL[i, :, :]

    if check_nan and L.isnan().any():
        return None

    L = L.reshape(n, n).contiguous()

    L.view(m, b, m,
           b).permute(0, 2, 1,
                      3)[torch.arange(m), torch.arange(m)] = torch.stack(
                          [torch.eye(b, device=L.device, dtype=H.dtype)] * m)

    return (L, D.to(DL.device))


def regularize_H(H, sigma_reg):
    diagmean = torch.diag(H).mean()
    H /= diagmean
    idx = torch.arange(len(H))
    H[idx, idx] += sigma_reg
    return H * diagmean
