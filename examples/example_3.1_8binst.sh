# This script reproduces the example models.

#HF_HOME=/data/alberttseng/hf_cache

CKPT=/scratch/qtip_examples/ckpt
HF=/scratch/qtip_examples/hfized
LOG=/scratch/qtip_examples/logs
HESS=/scratch/3.1hessians8binstruct

mkdir $CKPT
mkdir $LOG
mkdir $HF
'''
python -m quantize_llama.quantize_finetune_llama \
       --save_path $CKPT/3.1_8binst_2bit \
       --codebook bitshift \
       --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
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
       --ctx_size 8192 \
       --devset_size 512 \
       >> $LOG/3.1_8binst_2bit 2>&1


python -m quantize_llama.quantize_finetune_llama \
       --save_path $CKPT/3.1_8binst_3bit \
       --codebook bitshift \
       --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
       --in_hess_path $HESS \
       --scale_override 0.9 \
       --ft_epochs 5 \
       --td_x 16 \
       --td_y 16 \
       --L 16 \
       --K 3 \
       --V 2 \
       --decode_mode quantlut_sym \
       --tlut_bits 9 \
       --ctx_size 8192 \
       --devset_size 512 \
       >> $LOG/3.1_8binst_2bit 2>&1

python -m quantize_llama.quantize_finetune_llama \
       --save_path $CKPT/3.1_8binst_4bit \
       --codebook bitshift \
       --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
       --in_hess_path $HESS \
       --scale_override 0.9 \
       --ft_epochs 5 \
       --td_x 16 \
       --td_y 16 \
       --L 16 \
       --K 4 \
       --V 2 \
       --decode_mode quantlut_sym \
       --tlut_bits 9 \
       --ctx_size 8192 \
       --devset_size 512 \
       >> $LOG/3.1_8binst_2bit 2>&1


CUDA_VISIBLE_DEVICES=0 python -m quantize_llama.hfize_llama --quantized_path $CKPT/3.1_8binst_2bit --hf_output_path $HF/3.1_8binst_2bit >> $LOG/3.1_8binst_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m quantize_llama.hfize_llama --quantized_path $CKPT/3.1_8binst_3bit --hf_output_path $HF/3.1_8binst_3bit >> $LOG/3.1_8binst_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m quantize_llama.hfize_llama --quantized_path $CKPT/3.1_8binst_4bit --hf_output_path $HF/3.1_8binst_4bit >> $LOG/3.1_8binst_4bit 2>&1 &

wait



python -m quantize_llama.finetune_e2e_llama --base_model meta-llama/Meta-Llama-3.1-8B-Instruct --hf_path $HF/3.1_8binst_2bit --devset_size 1024 --ft_valid_size 128 --ft_epochs 4 --ft_update_freq 8 --ft_bs 1 --ctx_size 8192 --ft_train_lut --hf_output_path $HF/3.1_8binst_2bit >> $LOG/3.1_8binst_2bit 2>&1

python -m quantize_llama.finetune_e2e_llama --base_model meta-llama/Meta-Llama-3.1-8B-Instruct --hf_path $HF/3.1_8binst_3bit --devset_size 1024 --ft_valid_size 128 --ft_epochs 4 --ft_update_freq 8 --ft_bs 1 --ctx_size 8192 --ft_train_lut --hf_output_path $HF/3.1_8binst_3bit >> $LOG/3.1_8binst_3bit 2>&1

python -m quantize_llama.finetune_e2e_llama --base_model meta-llama/Meta-Llama-3.1-8B-Instruct --hf_path $HF/3.1_8binst_4bit --devset_size 1024 --ft_valid_size 128 --ft_epochs 4 --ft_update_freq 8 --ft_bs 1 --ctx_size 8192 --ft_train_lut --hf_output_path $HF/3.1_8binst_4bit >> $LOG/3.1_8binst_4bit 2>&1
'''

CUDA_VISIBLE_DEVICES=0 python -m eval.eval_ppl --seqlen 8192 --hf_path meta-llama/Meta-Llama-3.1-8B-Instruct >> $LOG/3.1_8binst_bf16 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m eval.eval_ppl --seqlen 8192 --hf_path $HF/3.1_8binst_2bit >> $LOG/3.1_8binst_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m eval.eval_ppl --seqlen 8192 --hf_path $HF/3.1_8binst_3bit >> $LOG/3.1_8binst_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -m eval.eval_ppl --seqlen 8192 --hf_path $HF/3.1_8binst_4bit >> $LOG/3.1_8binst_4bit 2>&1 &
CUDA_VISIBLE_DEVICES=4 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 16  --hf_path meta-llama/Meta-Llama-3.1-8B-Instruct >> $LOG/3.1_8binst_bf16 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 16  --hf_path $HF/3.1_8binst_2bit >> $LOG/3.1_8binst_2bit 2>&1 &
CUDA_VISIBLE_DEVICES=6 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 16  --hf_path $HF/3.1_8binst_3bit >> $LOG/3.1_8binst_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=7 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 16  --hf_path $HF/3.1_8binst_4bit >> $LOG/3.1_8binst_4bit 2>&1 &

wait
