# This script is based off of the generation script in https://github.com/chu-tianxiang/QuIP-for-all
import os
import time
from typing import Optional

import torch
from transformers import AutoTokenizer
from model.cache_utils import StaticCache

from lib.utils.unsafe_import import model_from_hf_path

torch.set_grad_enabled(False)


def multinomial_sample_one_no_sync(
        probs_sort
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1,
                        keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits,
                    temperature: float = 1.0,
                    top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

@torch.compile
def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


@torch.no_grad()
def decode_one_tokens(model, cur_token, past_kv, cache_position):
    logits = model(cur_token,
                   past_key_values=past_kv,
                   cache_position=cache_position)[0]
    new_token = sample(logits, temperature=0.6, top_k=5)[0]
    return new_token, logits


@torch.no_grad()
def generate(model, tokenizer, text, max_new_tokens, top_k, callback, past_kv):
    inputs = tokenizer(text, return_tensors="pt").to(0)
    batch_size, seq_length = inputs["input_ids"].shape
    cache_position = torch.arange(seq_length, device=0)
    generated_ids = torch.zeros(batch_size,
                                seq_length + max_new_tokens,
                                dtype=torch.int,
                                device=0)
    generated_ids[:, cache_position] = inputs["input_ids"].to(0).int()
    logits = model(**inputs,
                   past_key_values=past_kv,
                   cache_position=cache_position)[0]

    next_token, _ = sample(logits, top_k=top_k)

    generated_ids[:, seq_length] = next_token
    callback(next_token)

    cache_position = torch.tensor([seq_length + 1], device=0)
    decode_time = time.time()
    for _ in range(1, max_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=True,
                                            enable_mem_efficient=False,
                                            enable_math=True):
            next_token, logits = decode_one_tokens(model, next_token.clone(),
                                                   past_kv, cache_position)
        generated_ids[:, cache_position] = next_token.int()
        callback(next_token)
        cache_position += 1
    torch.cuda.synchronize()
    decode_time = time.time() - decode_time

    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_ids, text, max_new_tokens / decode_time


def llama_arg_fn(output, args, kwargs):
    return (output[0], *args[1:]), kwargs

def get_emb(args, kwargs):
    return args[0]


from lib.utils import shard_model as sm, clean

def main(hf_path, compile, interactive, max_tokens, top_k):
    device = "cuda"
    model, model_str = model_from_hf_path(hf_path)

    sharded = False
    
    if 'llama' in model_str.lower():
        n_shards = model.lm_head.weight.device.index + 1
        if n_shards > 1:
            sharded = True
            del model.model.layers
            clean()
            cpumodel, _ = model_from_hf_path(hf_path, device_map='cpu')
            nlayers = len(cpumodel.model.layers)
            shards = [torch.nn.ModuleList([]) for _ in range(n_shards)]
            layer_device_map = []
            for i in range(n_shards):
                for j in range(int(nlayers * i / n_shards),
                               int(nlayers * (i + 1) / n_shards)):
                    shards[i].append(cpumodel.model.layers[j])
                    layer_device_map.append(i)
                shards[i] = {'device': i, 'arg_fn': llama_arg_fn, 'shard': shards[i]}
            model.model.layers = [sm.ShardDecoderLayers(shards, model.lm_head.weight.dtype)]
                                                   
    
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    tokenizer.pad_token = tokenizer.eos_token
    if not sharded:
        past_kv = StaticCache(model.config,
                              1,
                              2*args.max_new_tokens,
                              device=0,
                              dtype=model.dtype)
    else:
        past_kv = StaticCache(model.config,
                              1,
                              2*args.max_new_tokens,
                              layer_device_map=layer_device_map,
                              dtype=model.dtype)

    text = "This is a test of this large language model"
    callback = lambda x: x
    ids, text, _ = generate(model, tokenizer, text,
                            8, top_k, callback, past_kv)

    if compile:
        print('Capturing CUDA graphs, may take some time. If you are running a model over multiple GPUs, the first generation will be very slow due to compiling the model.')
        if not sharded:
            global decode_one_tokens
            decode_one_tokens = torch.compile(decode_one_tokens,
                                              mode="max-autotune",
                                              fullgraph=True)
        else:
            for shard in model.model.layers[0].shards:
                shard.forward = torch.compile(shard.forward,
                                              mode='max-autotune',
                                              fullgraph=True)



    text = "This is a test of this large language model"
    ids, text, _ = generate(model, tokenizer, text,
                            16, top_k, callback, past_kv)

    while True:
        prompt = input("What is your prompt? ")
        if prompt == 'quit':
            exit()
        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages,
                                                 tokenize=False,
                                                 add_generation_prompt=True)
        else:
            text = prompt
        buffer = []
        period_id = tokenizer.encode('.')[-1]
        done_generating = False

        def callback(x):
            nonlocal done_generating
            if done_generating:
                return
            buffer.append(tokenizer.decode([period_id] + x[0].tolist())[1:])
            if x[0].item() == tokenizer.eos_token_id:
                done_generating = True
            if len(buffer) == 4 or done_generating:
                print(''.join(buffer), end='', flush=True)
                buffer.clear()

        if not interactive:
            callback = lambda x: x
        ids, text, decode_tps = generate(model, tokenizer, text,
                                         max_tokens, top_k, callback, past_kv)
        if not interactive:
            print(text)
            
        print(
            f"\nDecoding throughput: {decode_tps:.02f} tokens/sec. Includes tokens generated after the EOS token.\n\n"
        )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--hf_path', type=str, help="Path to checkpoint")
    parser.add_argument('--streaming',
                        action='store_true',
                        help='Whether to launch in stream mode')
    parser.add_argument('--max_new_tokens',
                        type=int,
                        default=512,
                        help='Maximum number of new tokens.')
    parser.add_argument('--top_k',
                        type=int,
                        default=32,
                        help='Top-k for sampling.')
    parser.add_argument('--no_compile',
                        action='store_true',
                        help='Whether to compile the model.')
    parser.add_argument('--disable_tf32',
                        action='store_true',
                        help='Whether to disable TF32 for FP32 matmuls.')

    args = parser.parse_args()

    if not args.disable_tf32:
        torch.set_float32_matmul_precision('high')

    main(args.hf_path, not args.no_compile, args.streaming,
         args.max_new_tokens, args.top_k)
