# functions in this file cause circular imports so they cannot be loaded into __init__

import json
import os

import transformers

from model.llama import LlamaForCausalLM


def model_from_hf_path(path,
                       device_map='auto'):

    # AutoConfig fails to read name_or_path correctly
    bad_config = transformers.AutoConfig.from_pretrained(path)
    is_quantized = hasattr(bad_config, 'quip_params')
    model_type = bad_config.model_type
    if is_quantized:
        if model_type == 'llama':
            model_str = transformers.LlamaConfig.from_pretrained(
                path)._name_or_path
            model_cls = LlamaForCausalLM
        else:
            raise Exception
    else:
        model_cls = transformers.AutoModelForCausalLM
        model_str = path

    model = model_cls.from_pretrained(
        path,
        torch_dtype='auto',
        low_cpu_mem_usage=True,
        attn_implementation='sdpa',
        device_map=device_map)

    return model, model_str
