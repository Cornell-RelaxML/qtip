# Example script to quantize Llama 2 7B to 2 bits

# Fill these in with your own paths
'''
CKPT=
HF=
LOG=
HESS=
'''

mkdir $CKPT
mkdir $LOG
mkdir $HF

# main quantization script
python -m quantize_llama.quantize_finetune_llama \
       --save_path $CKPT/2_7b_2bit \
       --codebook bitshift \
       --base_model meta-llama/Llama-2-7b-hf \
       --in_hess_path $HESS \
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

# convert the quantized model to a hf model
python -m quantize_llama.hfize_llama --quantized_path $CKPT/2_7b_2bit --hf_output_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1 

# do end to end finetuning
python -m quantize_llama.finetune_e2e_llama --base_model meta-llama/Llama-2-7b-hf --hf_path $HF/2_7b_2bit --devset_size 640 --ft_valid_size 128 --ft_epochs 4 --ft_update_freq 4 --ft_bs 2 --ctx_size 4096 --ft_train_lut --hf_output_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1

# evaluate perplexity and zeroshot results
python -m eval.eval_ppl  --hf_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1
python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 16  --hf_path $HF/2_7b_2bit >> $LOG/2_7b_2bit 2>&1
