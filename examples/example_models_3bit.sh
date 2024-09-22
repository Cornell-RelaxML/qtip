# This script reproduces the example models.

CKPT=/data_persistent3/alberttseng/qtip_examples/ckpt
HF=/data_persistent3/alberttseng/qtip_examples/hfized
LOG=/data_persistent3/alberttseng/qtip_examples/logs
HESS=/data_persistent3/restoration/mk-1-20240603/hessians

mkdir $CKPT
mkdir $LOG
mkdir $HF

python -m quantize_llama.quantize_finetune_llama \
       --save_path $CKPT/2_7b_3bit \
       --codebook bitshift \
       --base_model meta-llama/Llama-2-7b-hf \
       --in_hess_path $HESS/llama2_7b_6144/ \
       --scale_override 0.9 \
       --ft_epochs 5 \
       --td_x 16 \
       --td_y 16 \
       --L 16 \
       --K 3 \
       --V 2 \
       --decode_mode quantlut_sym \
       --tlut_bits 9 \
       >> $LOG/2_7b_3bit 2>&1


python -m quantize_llama.quantize_finetune_llama \
       --save_path $CKPT/2_7b_chat_3bit \
       --codebook bitshift \
       --base_model meta-llama/Llama-2-7b-chat-hf \
       --in_hess_path $HESS/llama2_7b_chat_6144/ \
       --scale_override 0.9 \
       --ft_epochs 5 \
       --td_x 16 \
       --td_y 16 \
       --L 16 \
       --K 3 \
       --V 2 \
       --decode_mode quantlut_sym \
       --tlut_bits 9 \
       >> $LOG/2_7b_chat_3bit 2>&1


python -m quantize_llama.quantize_finetune_llama \
       --save_path $CKPT/2_13b_3bit \
       --codebook bitshift \
       --base_model meta-llama/Llama-2-13b-hf \
       --in_hess_path $HESS/llama2_13b_6144/ \
       --scale_override 0.9 \
       --ft_epochs 5 \
       --td_x 16 \
       --td_y 16 \
       --L 16 \
       --K 3 \
       --V 2 \
       --decode_mode quantlut_sym \
       --tlut_bits 9 \
       >> $LOG/2_13b_3bit 2>&1


python -m quantize_llama.quantize_finetune_llama \
       --save_path $CKPT/2_13b_chat_3bit \
       --codebook bitshift \
       --base_model meta-llama/Llama-2-13b-chat-hf \
       --in_hess_path $HESS/llama2_13b_chat_6144/ \
       --scale_override 0.9 \
       --ft_epochs 5 \
       --td_x 16 \
       --td_y 16 \
       --L 16 \
       --K 3 \
       --V 2 \
       --decode_mode quantlut_sym \
       --tlut_bits 9 \
       >> $LOG/2_13b_chat_3bit 2>&1

python -m quantize_llama.quantize_finetune_llama \
       --save_path $CKPT/2_70b_3bit \
       --codebook bitshift \
       --base_model meta-llama/Llama-2-70b-hf \
       --in_hess_path $HESS/llama2_70b_6144/ \
       --scale_override 0.9 \
       --ft_epochs 5 \
       --td_x 16 \
       --td_y 16 \
       --L 16 \
       --K 3 \
       --V 2 \
       --decode_mode quantlut_sym \
       --tlut_bits 9 \
       >> $LOG/2_70b_3bit 2>&1


python -m quantize_llama.quantize_finetune_llama \
       --save_path $CKPT/2_70b_chat_3bit \
       --codebook bitshift \
       --base_model meta-llama/Llama-2-70b-chat-hf \
       --in_hess_path $HESS/llama2_70b_chat_6144/ \
       --scale_override 0.9 \
       --ft_epochs 5 \
       --td_x 16 \
       --td_y 16 \
       --L 16 \
       --K 3 \
       --V 2 \
       --decode_mode quantlut_sym \
       --tlut_bits 9 \
       >> $LOG/2_70b_chat_3bit 2>&1


CUDA_VISIBLE_DEVICES=0 python -m quantize_llama.hfize_llama --quantized_path $CKPT/2_7b_3bit --hf_output_path $HF/2_7b_3bit >> $LOG/2_7b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m quantize_llama.hfize_llama --quantized_path $CKPT/2_7b_chat_3bit --hf_output_path $HF/2_7b_chat_3bit >> $LOG/2_7b_chat_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m quantize_llama.hfize_llama --quantized_path $CKPT/2_13b_3bit --hf_output_path $HF/2_13b_3bit >> $LOG/2_13b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -m quantize_llama.hfize_llama --quantized_path $CKPT/2_13b_chat_3bit --hf_output_path $HF/2_13b_chat_3bit >> $LOG/2_13b_chat_3bit 2>&1 & 
CUDA_VISIBLE_DEVICES=4 python -m quantize_llama.hfize_llama --quantized_path $CKPT/2_70b_3bit --hf_output_path $HF/2_70b_3bit >> $LOG/2_70b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -m quantize_llama.hfize_llama --quantized_path $CKPT/2_70b_chat_3bit --hf_output_path $HF/2_70b_chat_3bit >> $LOG/2_70b_chat_3bit 2>&1 &

wait

python -m quantize_llama.finetune_e2e_llama --base_model meta-llama/Llama-2-7b-hf --hf_path $HF/2_7b_3bit --devset_size 640 --ft_valid_size 128 --ft_epochs 4 --ft_update_freq 4 --ft_bs 2 --ctx_size 4096 --ft_train_lut --hf_output_path $HF/2_7b_3bit >> $LOG/2_7b_3bit 2>&1
python -m quantize_llama.finetune_e2e_llama --base_model meta-llama/Llama-2-7b-chat-hf --hf_path $HF/2_7b_chat_3bit --devset_size 640 --ft_valid_size 128 --ft_epochs 4 --ft_update_freq 4 --ft_bs 2 --ctx_size 4096 --ft_train_lut --hf_output_path $HF/2_7b_chat_3bit >> $LOG/2_7b_chat_3bit 2>&1

python -m quantize_llama.finetune_e2e_llama --base_model meta-llama/Llama-2-13b-hf --hf_path $HF/2_13b_3bit --devset_size 640 --ft_valid_size 128 --ft_epochs 4 --ft_update_freq 4 --ft_bs 2 --ctx_size 4096 --ft_train_lut --hf_output_path $HF/2_13b_3bit >> $LOG/2_13b_3bit 2>&1
python -m quantize_llama.finetune_e2e_llama --base_model meta-llama/Llama-2-13b-chat-hf --hf_path $HF/2_13b_chat_3bit --devset_size 640 --ft_valid_size 128 --ft_epochs 4 --ft_update_freq 4 --ft_bs 2 --ctx_size 4096 --ft_train_lut --hf_output_path $HF/2_13b_chat_3bit >> $LOG/2_13b_chat_3bit 2>&1


python -m quantize_llama.finetune_e2e_llama --base_model meta-llama/Llama-2-70b-hf --hf_path $HF/2_70b_3bit --devset_size 640 --ft_valid_size 128 --ft_epochs 4 --ft_update_freq 8 --ft_bs 1 --ctx_size 4096 --ft_train_lut --hf_output_path $HF/2_70b_3bit --ft_grad_ckpt >> $LOG/2_70b_3bit 2>&1
python -m quantize_llama.finetune_e2e_llama --base_model meta-llama/Llama-2-70b-chat-hf --hf_path $HF/2_70b_chat_3bit --devset_size 640 --ft_valid_size 128 --ft_epochs 4 --ft_update_freq 8 --ft_bs 1 --ctx_size 4096 --ft_train_lut --hf_output_path $HF/2_70b_chat_3bit --ft_grad_ckpt >> $LOG/2_70b_chat_3bit 2>&1


CUDA_VISIBLE_DEVICES=0 python -m eval.eval_ppl  --hf_path $HF/2_7b_3bit >> $LOG/2_7b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m eval.eval_ppl  --hf_path $HF/2_7b_chat_3bit >> $LOG/2_7b_chat_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m eval.eval_ppl  --hf_path $HF/2_13b_3bit >> $LOG/2_13b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -m eval.eval_ppl  --hf_path $HF/2_13b_chat_3bit >> $LOG/2_13b_chat_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=4 python -m eval.eval_ppl  --hf_path $HF/2_70b_3bit >> $LOG/2_70b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -m eval.eval_ppl  --hf_path $HF/2_70b_chat_3bit >> $LOG/2_70b_chat_3bit 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 16  --hf_path $HF/2_7b_3bit >> $LOG/2_7b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 16  --hf_path $HF/2_7b_chat_3bit >> $LOG/2_7b_chat_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 16  --hf_path $HF/2_13b_3bit >> $LOG/2_13b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 16  --hf_path $HF/2_13b_chat_3bit >> $LOG/2_13b_chat_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=4 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 16  --hf_path $HF/2_70b_3bit >> $LOG/2_70b_3bit 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -m eval.eval_zeroshot --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size 16  --hf_path $HF/2_70b_chat_3bit >> $LOG/2_70b_chat_3bit 2>&1 &

wait
