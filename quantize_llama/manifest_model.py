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
    manifested_model = AutoModelForCausalLM.from_pretrained(args.hf_path)

    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            module.mode = 'train-fixW'
    
    tok = torch.tensor([[1]]).cuda()
    model(tok)

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            w = module.codebook_class.hatW.float()
            w = ((w*module.SU).T * module.SV).T * module.codebook_class.scale
            manifested_linear = attrgetter(name)(manifested_model)
            manifested_linear.weight.copy_(w)
    # set name or path to name of original model for tokenizer loading
    manifested_model.config._name_or_path = model_str
    manifested_model.save_pretrained(args.output_path, safe_serialization=True)
            
if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    main(args)
