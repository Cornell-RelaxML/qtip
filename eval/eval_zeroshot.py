import argparse
import json
import os
import random

import datasets
import glog
import torch
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer

#from lib.utils import LMEvalAdaptor
from lib.utils.unsafe_import import model_from_hf_path
from lib.linear import QuantizedLinear

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', default='hfized/quantized_hada_70b', type=str)
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--tasks", type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument('--num_fewshot', type=int, default=0)
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--apply_chat_template', action='store_true')
parser.add_argument('--fewshot_as_multiturn', action='store_true')
#parser.add_argument('--max_model_len', type=int)

def main(args):
    model, model_str = model_from_hf_path(args.hf_path)

    # manifest for faster inference
    # use for codebooks without kernel support
    '''
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            module.mode = 'train-fixW'
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    glog.info('loaded model!')
    tokenizer.pad_token = tokenizer.eos_token

    task_names = args.tasks.split(",")

    lm_eval_model = HFLM(
        model,
        tokenizer=tokenizer)
        
    #lm_eval_model = LMEvalAdaptor(model_str, model, tokenizer, args.batch_size)
    
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        batch_size=args.batch_size,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn
    )

    print(results['versions'])
    print(results['transformers_version'])
    print(results['n-shot'])
    print(results['results'])
    #print(evaluator.make_table(results))

    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        # otherwise cannot save
        results["config"]["model"] = args.hf_path
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
