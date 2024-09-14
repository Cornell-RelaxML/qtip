import argparse
import datetime
import os
import random
from copy import deepcopy

from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import numpy
import torch
import torch.multiprocessing as mp
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerFast)
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from lib import utils

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--large_batch_size', default=2048, type=int)
parser.add_argument('--devset_size', default=8192, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--base_model',
                    default='meta-llama/Llama-2-70b-hf',
                    type=str)
parser.add_argument('--save_path', default='hessians/llama2_70b', type=str)
parser.add_argument('--scratch_path', default=None, type=str)
parser.add_argument('--async_copy_speed', default=-1, type=int)
parser.add_argument('--act_save_rate', default=4, type=int)
parser.add_argument('--save_activations', action='store_true')
parser.add_argument('--sample_proc', default=4, type=int)



def forward_layer(layer, position_ids, attention_mask, bs, device, in_q,
                  out_q):
    torch.set_grad_enabled(False)
    layer = layer.to(device)
    position_ids = position_ids.to(device)
    attention_mask = attention_mask.to(device)
    done_qkv = utils.register_input_H_hook(layer.self_attn.q_proj, device)
    done_o = utils.register_input_H_hook(layer.self_attn.o_proj, device)
    done_up = utils.register_input_H_hook(layer.mlp.up_proj, device)
    done_down = utils.register_input_H_hook(layer.mlp.down_proj, device)

    while True:
        dev_emb = in_q.get()
        if dev_emb is None:
            layer = layer.cpu()
            position_ids = position_ids.cpu()
            attention_mask = attention_mask.cpu()
            out_q.put({
                'qkv': done_qkv(),
                'o': done_o(),
                'up': done_up(),
                'down': done_down()
            })
            return

        dev_emb[:] = layer(dev_emb.to(device),
                           position_ids=position_ids,
                           attention_mask=attention_mask,
                           use_cache=False,
                           output_attentions=False)[0].cpu()


def accumulate(in_q, move_q, ngpus, args, transformer_layer_index):
    Hs = {}
    mus = {}
    cts = {}

    for i in range(ngpus):
        out = in_q.get()
        if i == 0:
            for key in out:
                Hs[key] = torch.zeros(out[key][0].shape,
                                      dtype=out[key][0].dtype)
                mus[key] = torch.zeros(out[key][1].shape,
                                       dtype=out[key][1].dtype)
                cts[key] = 0
        for key in out:
            Hs[key].add_(out[key][0])
            mus[key].add_(out[key][1])
            cts[key] += out[key][2]

    keys = list(Hs.keys())

    for key in Hs:
        save_path = f"{args.save_path}/{transformer_layer_index}_{key}.pt"
        flatH = utils.sym_to_flat(Hs[key])
        if os.path.exists(save_path):
            data = torch.load(save_path, map_location=Hs[key].device)
            total_ct = data['ct'] + cts[key]
            Hkey = data['flatH'].to(torch.float64) * (data['ct'] / total_ct) + \
                flatH  / total_ct
            Hkey = Hkey.to(torch.float32)
        else:
            Hkey = (flatH / cts[key]).to(torch.float32)
            total_ct = cts[key]
            
        torch.save(
            {
                'flatH': Hkey,
                'n': Hs[key].shape[0],
                'ct': total_ct,
            }, save_path)

    del Hs, mus, cts, out


def main(args):
    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                 torch_dtype="auto",
                                                 low_cpu_mem_usage=True)
    print("loaded model!")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    print("loading dataset...")
    devset = utils.sample_rp1t_concat(tokenizer,
                                      args.devset_size,
                                      args.ctx_size,
                                      nproc=args.sample_proc)
    devset = torch.split(devset, args.large_batch_size)
    for lbi in range(len(devset)):
        print(f'processing split {lbi}')
        lbs = torch.split(devset[lbi], args.batch_size)
        dev_emb = [model.model.embed_tokens(chunk) for chunk in lbs]
        for i in range(len(dev_emb)):
            dev_emb[i].share_memory_()
        after_layer = -1
        print("loaded dataset!")

        position_ids = torch.arange(args.ctx_size, dtype=torch.int64)[None, :] + \
            torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int64)
        if hasattr(model.config, 'sliding_window'):
            attention_mask = _prepare_4d_causal_attention_mask(
                None, (args.batch_size, args.ctx_size),
                dev_emb[0],
                0,
                sliding_window=model.config.sliding_window)
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                None, (args.batch_size, args.ctx_size), dev_emb[0], 0)

        move_q = None

        for transformer_layer_index in range(len(model.model.layers)):
            print(transformer_layer_index)
            if (transformer_layer_index <= after_layer):
                print(
                    f"skipping layer {transformer_layer_index} because it is before cached activations at layer {after_layer}"
                )
                continue

            transformer_layer = model.model.layers[transformer_layer_index]
            ngpus = torch.cuda.device_count()

            manager = mp.get_context('spawn').Manager()
            in_q = manager.Queue()
            out_q = manager.Queue()

            accumulate_proc = mp.Process(target=accumulate,
                                         args=(out_q, move_q, ngpus, args,
                                               transformer_layer_index))
            accumulate_proc.start()

            forward_procs = []
            for i in range(ngpus):
                p = mp.Process(target=forward_layer,
                               args=(transformer_layer, position_ids,
                                     attention_mask, args.batch_size, i, in_q,
                                     out_q))
                p.start()
                forward_procs.append(p)

            for i in range(len(dev_emb)):
                in_q.put(dev_emb[i])

            for i in range(ngpus):
                in_q.put(None)

            for p in forward_procs:
                p.join()

            accumulate_proc.join()

            transformer_layer.cpu()
            utils.clean()

            print(f"done processing layer {transformer_layer_index}")
            
        del dev_emb, lbs
        utils.clean()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
