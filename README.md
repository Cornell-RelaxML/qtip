# [QTIP: Quantization with Trellises and Incoherence Processing](https://arxiv.org/abs/2406.11235), NeurIPS 2024 Spotlight

<img src="assets/qtip_overview.png" width="800">

This repository contains code for QTIP, a weight-only large language model (LLM) quantization method that achieves a state-of-the-art combination of quantization quality and speed.
QTIP uses incoherence processing to make LLM weight matrices approximately i.i.d Gaussian, and then uses trellis coded quantization (TCQ) to quantize these weights with near-optimal distortion.
QTIP solves naive TCQ's inherent slowness by introducing a series of novel compute-based codes for use with the "bitshift trellis."
For more details, please see the [paper](https://arxiv.org/abs/2406.11235).

## How to use this codebase

This codebase is based off of the [QuIP#](https://github.com/Cornell-RelaxML/quip-sharp) codebase, with modifications made to support trellis quantization.
The main QTIP code is in `lib/codebook/bitshift.py`, and the QuIP# algorithm files have been merged into `lib/algo/finetune.py`.
Example scripts can be found in `examples/`

The main QTIP-related arguments in `quantize_llama/quantize_finetune_llama.py` are:
- `L`, `K`, `V`: same as in the paper.
- `tlut_bits`: the number of tunable lookup table bits. This is Q for the HYB code. Set this to 0 if using 3INST or 1MAD or L if using a pure LUT.
- `decode_mode`: `quantlut_sym` (HYB), `3inst` (3INST), `1mad` (1MAD), or 'lut' (pure LUT).
- `td_x` and `td_y`: dimensions of trellis tile in LDLQ ($T_x$ and $T_y$ in the paper). `td_x` goes along the output dimension and `td_y` the input (channel) dimension.

You will need to install the packages in `requirements.txt` to use this codebase with `pip install -r requirements.txt`. If you have issues installing `fast-hadamard-transform`, try building from [source](https://github.com/Dao-AILab/fast-hadamard-transform). 

## Fast inference

QTIP achieves the same inference throughput as QuIP# despite achieving higher quality quantization.
The numbers below measure bs=1 inference speed on a RTX6000 Ada with matrix fusion (q, k, and v, and up and gate together) for QuIP# and QTIP.

|    Method   |    2-7B    | 2-70B |
|:-----------:|:----------:|:-----:|
|     FP16    | 55.9 tok/s |  OOM  |
|  AQLM 2 Bit |    81.5    |  8.78 |
| QuIP# 2 Bit |     186    |  22.2 |
|  QTIP 2 Bit |     188    |  23.5 |

This codebase contains 2, 3, and 4 bit matrix-vector multiplication kernels for the HYB code with L=16, Q=9, V=2, and $T_x = T_y = 16$.
These kernels are located in `qtip_kernels` and have been integrated into the `BitshiftLinear` class in `lib/codebook/bitshift.py`.
`eval/interactive_gen.py` contains a simple generation script that is compatible with those kernels and CUDA graphs (through `torch.compile`).

For example, if you want to generate up to 256 tokens of text from a 3 bit QTIP Llama 2 13B Chat model in "streaming mode" (slower than not streaming), run

`python -m eval.interactive_gen --hf_path relaxml/Llama-2-13b-QTIP-3Bit --max_new_tokens 256 --streaming`

**Note:**
This script does not fuse matrices (q/k/v and up/gate) so you will not get the speeds in the table above if you run it.
The publicly available models should get 80-90\% of matrix fusion speeds.
If you wish to quantize a model with matrix fusion, the QuIP# codebase has plumbing to do so and should mostly translate over to this one.


### Compiling the kernels

```
cd qtip-kernels
python setup.py install
```

## Prequantized Models

Prequantized QTIP models with the HYB code, L=16, and V=2 can be found [here](https://huggingface.co/collections/relaxml/qtip-quantized-models-66fa253ad3186746f4b62803). These models can be used by passing in the HF Hub path (e.g. `relaxml/Llama-2-7b-QTIP-2Bit`) to the `--hf-path` flag in the eval scripts. The Llama 3.1 405B models ending in `TP8` were quantized with 8-way tensor parallelism support. Here, the RHT is performed per-GPU and instead of across GPUs. This results in slightly worse quality but enables inference with TP. We have not actually tested TP inference speeds, but feel free to use these models with your own TP inference codebase.

**You must have access to the original gated Llama tokenizers to be able to run these models with our eval scripts.** 

## Other

If you found this work useful, please consider citing
```
@inproceedings{
      tseng2024qtip,
      title={{QTIP}: Quantization with Trellises and Incoherence Processing},
      author={Albert Tseng and Qingyao Sun and David Hou and Christopher De Sa},
      booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
      year={2024},
      url={https://openreview.net/forum?id=7sdkLVuYCU}
}
```

Use of Llama models is governed by the Llama Community License. Use of this code is governed by the GNU GPL v3 license.
