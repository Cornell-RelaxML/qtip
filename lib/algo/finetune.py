"""
Utilities for fine tuning
"""
import copy
import math
from operator import attrgetter

import glog
import torch
from torch import nn, multiprocessing as mp
from transformers import AutoModelForCausalLM

from lib import codebook, utils
from lib.linear import QuantizedLinear
from . import ldlq


def finetune_decoder_layer(layer, name, device, train_dl, valid_dl, orig_dtype,
                           args):
    layer = layer.to(device)
    optim = torch.optim.Adam(layer.parameters(), lr=args.ft_lr)
    best_loss = utils.calculate_mse_loss(layer, valid_dl, device)
    best_sd = copy.deepcopy(layer.state_dict())
    glog.info(f'layer {name} initial loss {best_loss}')
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    worse_ct = 0

    position_ids = None

    for epoch in range(args.ft_epochs):
        for bidx, (source, targets) in enumerate(train_dl):

            if position_ids is None:
                position_ids = torch.arange(source.shape[1],
                                            device=device).unsqueeze(0)

            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda',
                                dtype=orig_dtype,
                                enabled=True):
                output = layer(source.to(device), position_ids=position_ids)[0]
                loss = nn.MSELoss()(output, targets.to(device))
            scaler.scale(loss).backward()
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                    train_dl) - 1:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            test_loss = utils.calculate_mse_loss(layer, valid_dl, device)
            if test_loss < best_loss:
                glog.info(
                    f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                )
                best_loss = test_loss
                best_sd = copy.deepcopy(layer.state_dict())
                worse_ct = 0
            else:
                glog.info(
                    f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                )
                worse_ct += 1
                if worse_ct >= args.ft_early_stop:
                    break

    del optim, train_dl, valid_dl

    layer.load_state_dict(best_sd)
    utils.clean()
    layer = layer.cpu()


def quantize_finetune_decoder_layer(mixed_layer, quant_order, idx, cb, args,
                                    device, pre_orig_emb, orig_emb):
    torch.manual_seed(idx)
    torch.set_num_threads(args.num_cpu_threads)
    torch.set_grad_enabled(False)

    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    orig_dtype = None
    for p in mixed_layer.parameters():
        orig_dtype = p.dtype
        break
    mixed_layer = mixed_layer.float()

    train_dl, valid_dl = utils.split_data(pre_orig_emb, orig_emb, args)

    has_kernel = utils.has_kernel(args.decode_mode, args.L, args.K, args.V,
                                  args.tlut_bits, args.td_x, args.td_y)

    for quant_i, (linear_attr, name, in_hess_name,
                  out_hess_name) in enumerate(quant_order):
        cb = cb.to(device).to(orig_dtype)
        orig_linear = attrgetter(linear_attr)(mixed_layer)
        W = orig_linear.weight.to(dtype_)
        (m, n) = W.shape
        SU = (torch.randn(n, device=device).sign() + 1e-5).sign().to(dtype_)
        SV = (torch.randn(m, device=device).sign() + 1e-5).sign().to(dtype_)
        Wr = utils.matmul_hadUt(utils.matmul_hadUt(W.T.to(device) * SV).T * SU)

        in_hess_path = f'{args.in_hess_path}/{idx}_{in_hess_name}.pt'
        H_data = torch.load(in_hess_path, map_location=torch.device('cpu'))
        HR = utils.flat_to_sym(H_data['flatH'], H_data['n'])
        if 'mu' in H_data:
            mu = H_data['mu']
            HR += mu[None, :] * mu[:, None]
            del mu
        del H_data

        HR = utils.regularize_H(HR, args.sigma_reg)
        HRr = utils.matmul_hadUt(utils.matmul_hadUt(HR.to(device) * SU).T * SU)

        Wscale = Wr.square().mean().sqrt() / (
            cb.lut.to(torch.float64).square().mean().sqrt().float() *
            args.scale_override)
        Wr /= Wscale
        LRr, _ = utils.block_LDL(HRr, args.td_y)
        diag = torch.arange(n, device=LRr.device)
        LRr[diag, diag] = 0

        hatWr, Qidxs = ldlq.LDLQ(Wr, LRr, cb, args, for_kernel=has_kernel)

        packed = cb.pack_trellis(
            Qidxs.reshape(m // args.td_x, args.td_x, n // args.td_y,
                          args.td_y // args.V).transpose(1, 2).reshape(
                              -1, args.td_x * args.td_y // args.V))

        if has_kernel:
            packed = packed.view(torch.uint8).view(-1, 2).flip((-1,)).reshape(
                m // 16 // 2, 2, n // 16 // 2, 2, 16 * 16 // 8, args.K).permute(
                    0, 2, 4, 3, 1, 5).flip((-1,)).contiguous().flatten().view(
                        torch.int16).reshape(packed.shape)

        Wr *= Wscale
        hatWr *= Wscale

        err = torch.trace(
            (Wr - hatWr) @ HRr @ (Wr - hatWr).T) / torch.trace(Wr @ HRr @ Wr.T)
        print(f'{idx}_{name} proxy err {err.item()}')

        save_path = f'{args.save_path}/{idx}_{name}.pt'
        # save as int16 since torch doesn't support serializing uint16
        torch.save(
            {
                'trellis':
                packed.cpu(),
                'SU':
                SU.to(orig_dtype).cpu(),
                'SV':
                SV.to(orig_dtype).cpu(),
                'Wscale':
                Wscale,
                'proxy_err':
                err.item(),
                'tlut':
                cb.tlut.data.to(orig_dtype).cpu()
                if hasattr(cb, 'tlut') else None
            }, save_path)

        q_linear = QuantizedLinear(
            n,
            m,
            args.td_x,
            args.td_y,
            args.L,
            args.K,
            args.V,
            args.tlut_bits,
            args.decode_mode,
            mode='train-recons' if args.ft_train_lut else 'train-fixW',
            dtype=orig_dtype,
            grad_ckpt=False)
        q_linear.trellis.copy_(packed)
        q_linear.SU.copy_(SU)
        q_linear.SV.copy_(SV)
        q_linear = q_linear.to(device).float()
        q_linear.SU = nn.Parameter(q_linear.SU, requires_grad=True)
        q_linear.SV = nn.Parameter(q_linear.SV * Wscale, requires_grad=True)
        if q_linear.tlut is not None:
            q_linear.tlut.copy_(cb.tlut.data)
            q_linear.tlut.requires_grad = args.ft_train_lut

        split_attr = linear_attr.split('.')
        setattr(
            attrgetter('.'.join(split_attr[:-1]))(mixed_layer), split_attr[-1],
            q_linear)

        with torch.enable_grad():
            finetune_decoder_layer(mixed_layer, f'{idx}_{name}', device,
                                   train_dl, valid_dl, orig_dtype, args)

        cb = cb.cpu()

    for quant_i, (linear_attr, name, in_hess_name,
                  out_hess_name) in enumerate(quant_order):
        quant_linear = attrgetter(linear_attr)(mixed_layer)
        save_path = f'{args.save_path}/{idx}_{name}.pt'
        data = torch.load(save_path)
        data['SU'] = quant_linear.SU.data.to(orig_dtype).cpu()
        data['SV'] = (
            quant_linear.SV.data /
            data['Wscale'].to(quant_linear.SV.device)).to(orig_dtype).cpu()
        if quant_linear.tlut is not None:
            data['tlut'] = quant_linear.tlut.data.to(orig_dtype).cpu()
        torch.save(data, save_path)

    mixed_layer = mixed_layer.to(orig_dtype).cpu()

    utils.clean()
    torch.set_grad_enabled(False)


def infer(args, end_dev, n_layers, in_q, out_q):
    with torch.no_grad():
        fake_dev_map = {
            'model.embed_tokens': 0,
            'model.rotary_emb': 0,
            'model.norm': end_dev - 1,
            'lm_head': end_dev - 1
        }
        per_dev = math.ceil(n_layers / end_dev)
        for i in range(n_layers):
            fake_dev_map[f'model.layers.{i}'] = (i + 1) // per_dev

        model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                     torch_dtype='auto',
                                                     device_map=fake_dev_map,
                                                     low_cpu_mem_usage=True)
        while True:
            data = in_q.get()
            if data is None:
                return
            out_q.put(
                model(data.to(0))['logits'][:, :-1].contiguous().softmax(
                    dim=-1).cpu())


def finetune_susv_e2e(quant_model, start_dev, devset, orig_dtype, args):

    in_q = mp.Queue()
    out_q = mp.Queue()
    p = mp.Process(target=infer,
                   args=(args, start_dev, len(quant_model.model.layers), in_q,
                         out_q))
    p.start()

    train_dl, valid_dl = utils.split_data(devset, devset, args)

    optim = torch.optim.Adam(quant_model.parameters(), lr=args.ft_lr)

    best_loss = utils.calculate_ce_loss_model(quant_model, valid_dl, start_dev,
                                              in_q, out_q)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_sd = copy.deepcopy(quant_model.state_dict())
    glog.info(f'initial loss {best_loss}')
    worse_ct = 0
    for epoch in range(args.ft_epochs):
        for bidx, (source, _) in enumerate(train_dl):
            in_q.put(source)
            with torch.autocast(device_type='cuda',
                                dtype=orig_dtype,
                                enabled=True):
                output = quant_model(
                    source.to(start_dev))['logits'][:, :-1].contiguous()
                target = out_q.get().to(output.device)
                target = target.view(-1, target.shape[-1])
                loss = nn.CrossEntropyLoss()(output.view(-1, output.shape[-1]),
                                             target)
            scaler.scale(loss).backward()
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                    train_dl) - 1:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            test_loss = utils.calculate_ce_loss_model(quant_model, valid_dl,
                                                      start_dev, in_q, out_q)
            if test_loss < best_loss:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                )
                best_loss = test_loss
                best_sd = copy.deepcopy(quant_model.state_dict())
                worse_ct = 0
            else:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                )
                worse_ct += 1
                if worse_ct >= args.ft_early_stop:
                    break

    in_q.put(None)
    p.join()
    with torch.no_grad():
        quant_model.load_state_dict(best_sd)
