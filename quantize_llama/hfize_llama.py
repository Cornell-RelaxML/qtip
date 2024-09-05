import argparse
import os
import time

import glog
import torch
from transformers import AutoTokenizer

from lib import codebook, utils
from lib.utils.unsafe_import import model_from_hf_path
from model.llama import LlamaForCausalLM

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str)
parser.add_argument('--hf_output_path', type=str)


def main(args):
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'))
    model_config = saved_config['model_config']
    fused = model_config.quip_params.get('fused', True)

    tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path)

    model = LlamaForCausalLM.from_pretrained(model_config._name_or_path,
                                             torch_dtype='auto',
                                             low_cpu_mem_usage=True,
                                             config=model_config)
    cpu = torch.device('cpu')
    if os.path.exists(f'{args.quantized_path}/lmhead.pt'):
        lmhead_data = torch.load(f'{args.quantized_path}/lmhead.pt',
                                 map_location=cpu)
        model.lm_head.weight.copy_(lmhead_data['lm_head'].to(
            model.lm_head.weight.dtype))
        model.model.norm.weight.copy_(lmhead_data['norm'].to(
            model.model.norm.weight.dtype))

    for ii in range(len(model.model.layers)):
        layer = model.model.layers[ii]

        if os.path.exists(f'{args.quantized_path}/{ii}_layernorm.pt'):
            ln_data = torch.load(f'{args.quantized_path}/{ii}_layernorm.pt',
                                 map_location=cpu)
            layer.input_layernorm.weight.copy_(ln_data['input_layernorm'].to(
                layer.input_layernorm.weight.dtype))
            layer.post_attention_layernorm.weight.copy_(
                ln_data['post_attention_layernorm'].to(
                    layer.post_attention_layernorm.weight.dtype))

        saved_layer = torch.load(f'{args.quantized_path}/{ii}_q.pt',
                                 map_location=cpu)
        utils.unpack_quip(layer.self_attn.q_proj, saved_layer)

        saved_layer = torch.load(f'{args.quantized_path}/{ii}_k.pt',
                                 map_location=cpu)
        utils.unpack_quip(layer.self_attn.k_proj, saved_layer)

        saved_layer = torch.load(f'{args.quantized_path}/{ii}_v.pt',
                                 map_location=cpu)
        utils.unpack_quip(layer.self_attn.v_proj, saved_layer)

        saved_layer = torch.load(f'{args.quantized_path}/{ii}_up.pt',
                                 map_location=cpu)
        utils.unpack_quip(layer.mlp.up_proj, saved_layer)

        saved_layer = torch.load(f'{args.quantized_path}/{ii}_gate.pt',
                                 map_location=cpu)
        utils.unpack_quip(layer.mlp.gate_proj, saved_layer)

        saved_layer = torch.load(f'{args.quantized_path}/{ii}_o.pt',
                                 map_location=cpu)
        utils.unpack_quip(layer.self_attn.o_proj, saved_layer)

        saved_layer = torch.load(f'{args.quantized_path}/{ii}_down.pt',
                                 map_location=cpu)
        utils.unpack_quip(layer.mlp.down_proj, saved_layer)
        glog.info(f'loaded layer {ii}')

    glog.info(f'saving model...')
    model.save_pretrained(args.hf_output_path, safe_serialization=True)

    del model

    model, _ = model_from_hf_path(args.hf_output_path)

    glog.info('successfully loaded hfized model')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)
