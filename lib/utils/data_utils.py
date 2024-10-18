import multiprocessing as mp

import glog
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from lib import codebook

from .matmul_had import matmul_hadU
from .misc import clean


def flat_to_sym(V, N):
    A = torch.zeros(N, N, dtype=V.dtype, device=V.device)
    idxs = torch.tril_indices(N, N, device=V.device)
    A[idxs.unbind()] = V
    A[idxs[1, :], idxs[0, :]] = V
    return A


def sym_to_flat(A):
    N = A.shape[-1]
    idxs = torch.tril_indices(N, N, device=A.device)
    return A[idxs.unbind()]


def register_input_H_hook(module, save_pfx, device):
    n = module.in_features
    H = torch.zeros(n, n, dtype=torch.float64, device=device)
    ct = 0

    def H_hook(module, x):
        nonlocal H, ct, n
        x = x[0].reshape(-1, n).to(torch.float64)
        H.addmm_(x.T, x)
        ct += len(x)

    hook = module.register_forward_pre_hook(H_hook)

    def done():
        nonlocal H, ct, hook
        save_path = f"{save_pfx}_{device}.pt"
        torch.save({'H': H, 'n': H.shape[0], 'ct': ct}, save_path)
        del H, ct
        hook.remove()
        del hook
        clean()

    return done


def register_output_H_hook(module, *args, **kwargs):
    device = module.self_attn.q_proj.weight.device
    dtype = torch.float32
    H_out = None
    ct_out = 0

    bw_hook = None

    def forward_hook(module, input, output):
        nonlocal H_out, ct_out, bw_hook

        @torch.no_grad
        def backward_hook(grad):
            nonlocal H_out, ct_out
            eps = grad.view(-1, grad.shape[-1]).float()
            if H_out is None:
                H_out = sym_to_flat(
                    torch.zeros(eps.shape[-1],
                                eps.shape[-1],
                                dtype=dtype,
                                device=device))
            H_out += sym_to_flat((eps.T @ eps).to(device).to(dtype))
            ct_out += eps.shape[0]

        if not output[0].requires_grad:
            output[0].requires_grad = True
        bw_hook = output[0].register_hook(backward_hook)

    fw_hook = module.register_forward_hook(forward_hook)

    def done():
        nonlocal H_out, ct_out, bw_hook, fw_hook
        bw_hook.remove()
        fw_hook.remove()
        return H_out.cpu(), ct_out

    return done


def register_output_from_in_H_hook(module, H_in):
    device = 'cpu'  #module.weight.device
    dtype = torch.float32
    n = module.in_features
    m = module.out_features
    H_out = sym_to_flat(torch.zeros(m, m, dtype=dtype, device=device))
    ct_out = 0
    H_in_norm = H_in.norm().cpu()

    bw_hook = None

    def forward_hook(module, input, output):
        nonlocal H_out, ct_out, H_in, bw_hook
        with torch.no_grad():
            input = input[0].view(-1, input[0].shape[-1]).to(dtype)
            xHx = ((input @ flat_to_sym(H_in.to(input.device).to(dtype), n)) *
                   input).sum(dim=-1, keepdims=True).cpu()

        @torch.no_grad
        def backward_hook(grad):
            nonlocal H_out, ct_out
            eps = grad.view(-1, grad.shape[-1]).to(dtype)
            H_out += sym_to_flat(
                (eps.T @ (eps * xHx.to(eps.device))).to(device).to(dtype))
            ct_out += eps.shape[0]

        if not output.requires_grad:
            output.requires_grad = True
        bw_hook = output.register_hook(backward_hook)

    fw_hook = module.register_forward_hook(forward_hook)

    def done():
        nonlocal H_out, ct_out, bw_hook, fw_hook
        bw_hook.remove()
        fw_hook.remove()
        return H_out.cpu() / (H_in_norm**2), ct_out, H_in_norm

    return done


def register_input_from_out_H_hook(module, H_out):
    device = 'cpu'  #module.weight.device
    dtype = torch.float32
    n = module.in_features
    m = module.out_features
    H_in = torch.zeros(n, n, dtype=dtype, device=device)
    ct_in = 0
    H_out_norm = H_out.norm().cpu()

    bw_hook = None

    def forward_hook(module, input, output):
        nonlocal H_in, ct_in, H_out, bw_hook, m
        with torch.no_grad():
            input = input[0].view(-1, input[0].shape[-1]).to(dtype)

        @torch.no_grad()
        def backward_hook(grad):
            nonlocal H_in, ct_in, input, m
            eps = grad.view(-1, grad.shape[-1]).to(dtype)
            H_in.add_((input.T @ (input * (
                (eps @ flat_to_sym(H_out.to(eps.device).to(dtype), m)) *
                eps).sum(dim=-1, keepdims=True))).to(device).to(dtype))
            ct_in += eps.shape[0]

        if not output.requires_grad:
            output.requires_grad = True
        bw_hook = output.register_hook(backward_hook)

    fw_hook = module.register_forward_hook(forward_hook)

    def done():
        nonlocal H_in, ct_in, bw_hook, fw_hook
        bw_hook.remove()
        fw_hook.remove()
        return H_in.cpu() / (H_out_norm**2), ct_in, H_out_norm

    return done


def register_detach_hook(module):

    def forward_hook(module, input, output):

        @torch.no_grad
        def backward_hook(grad):
            return grad * 0

        if not output.requires_grad:
            output.requires_grad = True
        output.register_hook(backward_hook)

    return module.register_forward_hook(forward_hook)


def wrap_tokenizer(tokenizer, x, ctx_size, truncate=True):
    return tokenizer(x,
                     return_tensors='pt',
                     truncation=truncate,
                     padding=True,
                     max_length=ctx_size)


def sample_rp1t(tokenizer, size=128, ctx_size=2048, nproc=1):
    dataset = load_dataset('togethercomputer/RedPajama-Data-1T-Sample',
                           split='train')
    devset = torch.zeros((size, ctx_size), dtype=torch.int64)
    saved = 0
    if nproc > 1:
        p = mp.Pool(nproc)
        while saved < size:
            seqs = [(tokenizer, dataset[torch.randint(len(dataset),
                                                      (size, ))]['text'],
                     ctx_size) for _ in range(nproc)]
            tokens = p.starmap(wrap_tokenizer, seqs)
            for i in range(len(tokens)):
                lens = tokens[i].attention_mask.sum(dim=-1)
                good = torch.where(lens == ctx_size)[0]
                if len(good) > 0:
                    if saved + len(good) > size:
                        good = good[:size - saved]
                    devset[saved:saved + len(good)] = tokens[i].input_ids[good]
                    saved += len(good)
                    print(saved)
    else:
        while saved < size:
            tokens = tokenizer(dataset[torch.randint(len(dataset),
                                                     (size, ))]['text'],
                               return_tensors='pt',
                               truncation=True,
                               padding=True,
                               max_length=ctx_size)
            lens = tokens.attention_mask.sum(dim=-1)
            good = torch.where(lens == ctx_size)[0]
            if len(good) > 0:
                if saved + len(good) > size:
                    good = good[:size - saved]
                devset[saved:saved + len(good)] = tokens.input_ids[good]
                saved += len(good)
    return devset


def sample_rp1t_concat(tokenizer, size=128, ctx_size=2048, nproc=1):
    dataset = load_dataset('togethercomputer/RedPajama-Data-1T-Sample',
                           split='train')
    devset = torch.zeros((size, ctx_size), dtype=torch.int64)
    concat = []
    p = mp.Pool(nproc)
    while len(concat) < ctx_size * size:
        seqs = [(tokenizer, dataset[torch.randint(len(dataset),
                                                  (128, ))]['text'], -1, False)
                for _ in range(nproc)]
        tokens = p.starmap(wrap_tokenizer, seqs)
        for i in range(len(tokens)):
            lens = tokens[i].attention_mask.sum(dim=-1)
            for j in range(len(tokens[i].input_ids)):
                concat += tokens[i].input_ids[j][:lens[j]]
        print(len(concat), ctx_size * size)
    concat = torch.tensor(concat)[:ctx_size * size]
    return concat.reshape(size, ctx_size).contiguous()


def sample_falcon_refinedweb(tokenizer, size=128, ctx_size=2048, nproc=1):
    dataset = load_dataset('tiiuae/falcon-refinedweb',
                           streaming=True,
                           split='train')
    dataset = dataset.shuffle(buffer_size=100000, seed=0)
    iter_dataset = iter(dataset)

    devset = torch.zeros((size, ctx_size), dtype=torch.int64)
    saved = 0

    p = mp.Pool(nproc)
    while saved < size:
        seqs = [(tokenizer,
                 [next(iter_dataset)['content']
                  for _ in range(size)], ctx_size) for _ in range(nproc)]
        tokens = p.starmap(wrap_tokenizer, seqs)
        for token in tokens:
            good = torch.where(token.attention_mask.sum(dim=-1) == ctx_size)[0]
            if saved + len(good) > size:
                good = good[:size - saved]
            devset[saved:saved + len(good)] = token.input_ids[good]
            saved += len(good)
    p.close()
    return devset


def unpack_quip(module, saved_layer):
    module.trellis.copy_(saved_layer['trellis'])
    if module.tlut is not None:
        module.tlut.copy_(saved_layer['tlut'].float().to(torch.float16))
    if 'rcp' in saved_layer:
        rcp = saved_layer['rcp']
        SU = saved_layer['SU'].float()
        SV = saved_layer['SV'].float()
        Wscale = saved_layer['Wscale'].float()
        module.rcp.copy_(rcp)
        module.tp_rank.copy_(saved_layer['tp_rank'])
        if rcp == 1:
            # row
            module.SU.copy_(
                (SU.reshape(8, -1) * Wscale.unsqueeze(-1)).reshape(SU.shape))
            module.SV.copy_(SV)
        elif rcp == 2:
            module.SU.copy_(SU)
            module.SV.copy_(
                (SV.reshape(8, -1) * Wscale.unsqueeze(-1)).reshape(SV.shape))
        else:
            module.SU.copy_(SU)
            module.SV.copy_(SV * Wscale)
    else:
        module.SU.copy_(saved_layer['SU'])
        module.SV.copy_(saved_layer['SV'].float() *
                        saved_layer['Wscale'].float())


def dtype_from_str(str):
    dtype_map = {
        'torch.int64': torch.int64,
        'torch.int32': torch.int32,
        'torch.int16': torch.int16,
        'torch.uint8': torch.uint8,
        'torch.int8': torch.int8,
    }
    return dtype_map[str]


class SimpleDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def split_data(X, Y, args):
    split = int(len(X) - args.ft_valid_size)
    glog.info(f'using {split} training seqs, {len(X) - split} validation seqs')
    train_ds = SimpleDataset(X[:split], Y[:split])
    valid_ds = SimpleDataset(X[split:], Y[split:])
    train_dl = DataLoader(train_ds,
                          batch_size=args.ft_bs,
                          pin_memory=True,
                          shuffle=True)
    valid_dl = DataLoader(valid_ds,
                          batch_size=args.ft_bs,
                          pin_memory=True,
                          shuffle=False)
    return train_dl, valid_dl


def calculate_logits(model, devset, batch_size):
    logits = []
    for i in range(len(devset) // batch_size):
        logits.append(
            model(devset[i * batch_size:(i + 1) *
                         batch_size].cuda())['logits'].cpu())
    logits = torch.concat(logits, dim=0)
    return logits
