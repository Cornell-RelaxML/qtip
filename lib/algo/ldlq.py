import copy
import os

import glog
import torch
from tqdm import tqdm

from lib import utils

_PERMUTE = torch.arange(256).reshape(2, 8, 2, 4, 2).permute(1, 3, 2, 0,
                                                            4).flatten()
_INV_PERMUTE = torch.zeros(256, dtype=torch.int64)
_INV_PERMUTE[_PERMUTE] = torch.arange(256)


def LDLQ(Wr, L, cb, args, buf_cols=128, for_kernel=True):
    if for_kernel:
        assert args.td_x == 16 and args.td_y == 16
    buf_cols = max(buf_cols, args.td_y)
    trellissz = args.td_x * args.td_y
    (m, n) = Wr.shape
    assert buf_cols % args.td_y == 0
    assert n % buf_cols == 0
    assert args.td_y % args.V == 0
    buf_size = buf_cols // args.td_y

    hatWr_T = torch.zeros(n, m, dtype=L.dtype, device=L.device)
    Qidxs_T = torch.zeros(n // args.V, m, dtype=cb.idx_dtype, device=L.device)

    device = Wr.device
    Wr = Wr.cpu()
    utils.clean()
    Wr_T = Wr.T.contiguous().to(device)

    # quip
    prod_cache = torch.zeros(n, m, dtype=Wr_T.dtype, device=Wr_T.device)
    for cur_col in tqdm(range(n // args.td_y, 0, -buf_size)):
        b_Wr_T = Wr_T[args.td_y * (cur_col - buf_size):args.td_y * cur_col]
        b_hatWr_T = hatWr_T[args.td_y * (cur_col - buf_size):args.td_y *
                            cur_col]
        b_L = L[args.td_y * (cur_col - buf_size):args.td_y *
                cur_col].contiguous()
        b_prod = prod_cache[args.td_y * (cur_col - buf_size):args.td_y *
                            cur_col]
        b_Qidxs_T = Qidxs_T[args.td_y * (cur_col - buf_size) //
                            args.V:args.td_y * cur_col // args.V]
        L_offset = args.td_y * (cur_col - buf_size)
        for i in reversed(range(buf_size)):
            WXWX = b_Wr_T[args.td_y * i : args.td_y * (i + 1)] + \
                b_L[args.td_y * (i + 1):, L_offset + args.td_y * i : L_offset + args.td_y * (i + 1)].T @ \
                (b_Wr_T[args.td_y * (i + 1):] - b_hatWr_T[args.td_y * (i + 1):]) + \
                b_prod[args.td_y * i : args.td_y * (i + 1)]
            if trellissz > -1:
                WXWXshape = WXWX.shape
                thing = WXWX.T.reshape(-1, trellissz)
                if for_kernel:
                    thing = thing[..., _PERMUTE]
                q_out = cb.quantize(thing)
                if for_kernel:
                    thing = q_out[0][..., _INV_PERMUTE].reshape(
                        WXWXshape[1], WXWXshape[0])
                else:
                    thing = q_out[0].reshape(WXWXshape[1], WXWXshape[0])
                idxs = q_out[1].reshape(WXWXshape[1], WXWXshape[0] // args.V)
                b_hatWr_T[args.td_y * i:args.td_y * (i + 1)] = thing.T
                b_Qidxs_T[args.td_y // args.V * i:args.td_y // args.V *
                          (i + 1)] = idxs.T
            else:
                q_out = cb.quantize(WXWX.T)
                b_hatWr_T[args.td_y * i:args.td_y * (i + 1)] = q_out[0].T
                b_Qidxs_T[args.td_y // args.V * i:args.td_y // args.V *
                          (i + 1)] = q_out[1].T

        prod_cache += b_L.T @ (b_Wr_T - b_hatWr_T)
        hatWr_T[args.td_y * (cur_col - buf_size):args.td_y *
                cur_col] = b_hatWr_T

    del b_Wr_T, b_hatWr_T, b_L, b_prod, L_offset, prod_cache
    utils.clean()
    return hatWr_T.T.contiguous(), Qidxs_T.T.contiguous()
