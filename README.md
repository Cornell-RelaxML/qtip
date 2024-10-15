# QTIP: Quantization with Trellises and Incoherence Processing [[arXiv]](https://arxiv.org/abs/2406.11235)

<img src="assets/qtip_overview.PNG" width="800">

This repository contains code for QTIP, a weight-only large language model (LLM) quantization method that achieves a state-of-the-art combination of quantization quality and speed.
QTIP uses incoherence processing to make LLM weight matrices approximately i.i.d Gaussian, and then uses trellis coded quantization (TCQ) to quantize these weights with near-optimal distortion.
QTIP solves naive TCQ's inherent slowness by introducing a series of novel compute-based codes for use with the "bitshift trellis."
For more details, please see the [paper](https://arxiv.org/abs/2406.11235).

## 👉 QTIP will appear at NeurIPS 2024 as a Spotlight! Feel free to reach out if you'll be there too!

## How to use this codebase

This codebase is based off of the [QuIP#](https://github.com/Cornell-RelaxML/quip-sharp) codebase, with modifications made to support trellis quantization.
The main QTIP code is in `lib/codebook/bitshift.py`, and the QuIP# algorithm files have been merged into `lib/algo/finetune.py`.
Example scripts can be found in `examples/`

The main QTIP-related arguments in `quantize_llama/quantize_finetune_llama.py` are:
- `L`, `K`, `V`: same as in the paper.
- `tlut_bits`: the number of tunable lookup table bits. This is Q for the HYB code. Set this to 0 if using 3INST or 1MAD or L if using a pure LUT.
- `decode_mode`: `quantlut_sym` (HYB), `3inst` (3INST), `1mad` (1MAD), or 'lut' (pure LUT).
- `td_x` and `td_y`: dimensions of trellis tile in LDLQ ($T_x$ and $T_y$ in the paper). `td_x` goes along the output dimension and `td_y` the input (channel) dimension.

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

This script **does not fuse matrices** so you will not get get the speeds in the table above if you run it.
If you wish to quantize a model with matrix fusion, the QuIP# codebase has plumbing to do so and should mostly translate over to this one.
This script also **does not support CUDA graphs if the model spans multiple GPUs**, so expect very slow inference if your model spans multiple GPUs. 


### Compiling the kernels

```
cd qtip-kernels
python setup.py install
```

## Prequantized Models

Prequantized QTIP models with the HYB code, L=16, and V=2 can be found [here](https://huggingface.co/collections/relaxml/qtip-quantized-models-66fa253ad3186746f4b62803). These models can be used by passing in the HF Hub path (e.g. `relaxml/Llama-2-7b-QTIP-2Bit`) to the `--hf-path` flag in the eval scripts.

## Other

If you found this work useful, please consider citing
```
@misc{tseng2024qtipquantizationtrellisesincoherence,
      title={QTIP: Quantization with Trellises and Incoherence Processing}, 
      author={Albert Tseng and Qingyao Sun and David Hou and Christopher De Sa},
      year={2024},
      eprint={2406.11235},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.11235}, 
}
```
