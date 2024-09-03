# This script quantizes Llama 2 7B using QTIP with the hybrid code.

CKPT=ckpt
HF=hfized
LOG=logs
HESS=YOUR/HESSIAN/PATH/HERE

mkdir $CKPT
mkdir $LOG
mkdir $HF

# Quantize to 2 bits (K = 2) with the hybrid code, a 16x16 tile size (td_x = 16, td_y = 16), L = 16, 2 weights at once (V = 2), and 5 epochs of fine tuning.
# Changing K changes the bitrate. --decode_mode controls the code (eg 3INST or 1MAD). To use 3INST and 1MAD, additionally set --tlut_bits to 0.
python -m quantize_llama.quantize_finetune_llama \
       --save_path $CKPT/2_7b_2bit \
       --codebook bitshift \
       --base_model meta-llama/Llama-2-7b-hf \
       --in_hess_path $HESS/hessians/llama2_7b_6144/ \
       --scale_override 0.9 \
       --ft_epochs 5 \
       --td_x 16 \
       --td_y 16 \
       --L 16 \
       --K 2 \
       --V 2 \
       --decode_mode quantlut_sym \
       --tlut_bits 9 \
       >> $LOG/2_7b_2bit 2>&1

CUDA_VISIBLE_DEVICES=0 python -m quantize_llama.hfize_llama --quantized_path $CKPT/2_7b_2bit --hf_output_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1 

# (Optional) Do end to end fine-tuning. This script is poorly optimized and is essentially the same E2E fine tuning script as QuIP#.
python -m quantize_llama.finetune_e2e_llama --base_model meta-llama/Llama-2-7b-hf --hf_path $HF/2_7b_2bit --devset_size 640 --ft_valid_size 128 --ft_epochs 2 --ft_update_freq 4 --ft_bs 2 --ctx_size 4096 --ft_train_lut --hf_output_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1

# Evaluate perplexity and zeroshot performance.
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_ppl  --hf_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1 
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 16  --hf_path $HF/2_7b_2bit # >> $LOG/2_7b_2bit 2>&1
