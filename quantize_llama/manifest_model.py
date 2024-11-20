import argparse
import json
import math
import os
import random

import datasets
import glog
import torch
from tqdm import tqdm

from lib.linear import QuantizedLinear
from lib.utils.unsafe_import import model_from_hf_path
from transformers import AutoModelForCausalLM
from operator import attrgetter

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--hf_path', type=str)
parser.add_argument('--output_path', type=str)


def main(args):
    model, model_str = model_from_hf_path(args.hf_path)
    orig_model = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype='auto')

    tok = torch.tensor([[1]]).cuda()
    model(tok)

    names = [n for n, _ in model.named_modules()]
    for name in names:
        try:
            module = attrgetter(name)(model)
        except:
            continue
        if isinstance(module, QuantizedLinear):
            module.codebook_class.cache_hatW(
                module.trellis,
                module.had_left,
                module.had_right,
                module.K_left,
                module.K_right,
                len(module.SV),
                len(module.SU),
                module.rcp,
                module.tp_rank,
            )
            w = module.codebook_class.hatW.float()
            w = ((w*module.SU).T * module.SV).T * module.codebook_class.scale            
            orig_linear = attrgetter(name)(orig_model)
            orig_linear.weight.copy_(w.to(orig_linear.weight.dtype))
            del module.codebook_class.hatW
            split_attr = name.split('.')
            setattr(
                attrgetter('.'.join(split_attr[:-1]))(model), split_attr[-1],
                orig_linear)
            print(name)

    model.config._name_or_path = model_str
    model.config._quantized_name_or_path = args.hf_path
    del model.config.quip_params
    model.save_pretrained(args.output_path, safe_serialization=True)
            
if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    main(args)
