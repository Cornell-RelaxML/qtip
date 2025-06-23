import math
import time

import torch
import torch.nn as nn

from lib.codebook import bitshift
from lib.utils import (clean, dtype_from_str, get_hadK, has_kernel,
                       matmul_hadU_cuda)


class QuantizedLinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        td_x,
        td_y,
        L,  # trellis window
        K,  # bpw
        V,  # vq dim
        tlut_bits,  # tunable LUT bits
        decode_mode,
        bias=False,
        dtype=torch.float16,
        mode='eval',
        use_prev_kernel=True,
        grad_ckpt=False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.td_x = td_x
        self.td_y = td_y
        self.L = L
        self.K = K
        self.V = V
        self.tlut_bits = tlut_bits
        self.decode_mode = decode_mode
        self.register_buffer('rcp', torch.tensor(0))
        # TP rank, not used unless rcp != 0
        self.register_buffer('tp_rank', torch.tensor(8))
        self.dtype = dtype
        # packed into int16
        self.register_buffer(
            'trellis',
            torch.zeros((out_features // td_x) * (in_features // td_y),
                        math.ceil((td_x * td_y) * K / 16),
                        dtype=torch.int16))

        if decode_mode in ['lut', 'quantlut', 'quantlut_sym']:
            self.tlut = nn.Parameter(torch.zeros(2**tlut_bits,
                                                 V,
                                                 dtype=torch.float16),
                                     requires_grad=False)
        else:
            self.tlut = None

        if bias:
            self.register_buffer('bias', torch.ones(out_features))
        else:
            self.bias = None

        self.register_buffer("SU", torch.ones(in_features, dtype=self.dtype))
        self.register_buffer("SV", torch.ones(out_features,
                                              dtype=torch.float32))

        self.built_codebook_class = False
        self.built_graph = False

        had_left, K_left = get_hadK(in_features)
        had_right, K_right = get_hadK(out_features)
        self.register_buffer('had_left', had_left, persistent=False)
        self.register_buffer('had_right', had_right, persistent=False)
        self.K_left = K_left
        self.K_right = K_right
        self.mode = mode
        self.use_prev_kernel = use_prev_kernel
        self.grad_ckpt = grad_ckpt
        self.has_kernel = has_kernel(decode_mode, L, K, V, tlut_bits, td_x,
                                     td_y)

    def forward(self, input):
        if self.grad_ckpt:
            return self.ckpt_forward(input)
        return self.no_ckpt_forward(input)

    def ckpt_forward(self, input):
        return torch.utils.checkpoint.checkpoint(self.no_ckpt_forward,
                                                 input,
                                                 use_reentrant=True)

    def no_ckpt_forward(self, input):
        if not self.built_codebook_class:
            self.codebook_class = bitshift.BitshiftLinear(
                self.td_x,
                self.td_y,
                self.L,
                self.K,
                self.V,
                self.tlut_bits,
                self.decode_mode,
                dtype=self.dtype,
                tlut=self.tlut,
                has_kernel=self.has_kernel)

            rcp = self.rcp.item()
            del self.rcp
            self.rcp = rcp

            if self.mode == 'eval':
                pass
            elif self.mode == 'train-recons':
                if not self.has_kernel:
                    self.packed_trellis = self.trellis.cpu()
                    unpacked_trellis = self.codebook_class.cb.unpack_trellis(
                        self.trellis, self.td_x * self.td_y)
                    self.trellis = unpacked_trellis
                    clean()
            elif self.mode == 'train-fixW':
                self.codebook_class.cache_hatW(self.trellis, self.had_left,
                                               self.had_right, self.K_left,
                                               self.K_right, len(self.SV),
                                               len(self.SU), self.rcp,
                                               self.tp_rank)
                self.trellis = self.trellis.cpu()
                del self.had_left, self.had_right, self.K_left, self.K_right
                clean()
                self.had_left = None
                self.had_right = None
                self.K_left = None
                self.K_right = None
            else:
                raise Exception

            self.built_codebook_class = True

        result = self.codebook_class(input,
                                     self.trellis,
                                     self.SU,
                                     self.SV,
                                     self.had_left,
                                     self.had_right,
                                     self.K_left,
                                     self.K_right,
                                     self.rcp,
                                     self.tp_rank,
                                     mode=self.mode,
                                     use_prev_kernel=self.use_prev_kernel) + 0
        if self.bias is not None:
            return result + self.bias
        return result
