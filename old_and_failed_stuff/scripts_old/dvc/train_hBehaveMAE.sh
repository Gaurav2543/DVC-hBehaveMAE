

#!/bin/bash

# Number of GPUs to use (passed from sbatch)
# see README.md for explainations on the arguments

GPUS=$1
common_args="--dataset dvc \
    --path_to_data_dir dvc-data/Imputed_single_cage\
    --batch_size 512 \
    --model hbehavemae \
    --input_size 1440 1 12 \
    --stages 2 3 4 \
    --q_strides 15,1,1;6,1,1 \
    --mask_unit_attn True False False \
    --patch_kernel 2 1 12 \
    --init_embed_dim 96 \
    --init_num_heads 2 \
    --out_embed_dims 64 96 128 \
    --epochs 200 \
    --num_frames 1440 \
    --decoding_strategy single \
    --decoder_embed_dim 128 \
    --decoder_depth 1 \
    --decoder_num_heads 1 \
    --pin_mem \
    --num_workers 8 \
    --sliding_window 7 \
    --blr 1.6e-4 \
    --warmup_epochs 40 \
    --masking_strategy random \
    --mask_ratio 0.70 \
    --clip_grad 0.02 \
    --checkpoint_period 20 \
    --norm_loss False \
    --seed 0 \
    --output_dir dvc-data/outputs/experiment6 \
    --log_dir dvc-data/logs/experiment6 \
    --distributed "


if [[ $GPUS -gt 1 ]]; then
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --node_rank 0 --master_addr=127.0.0.1 --master_port=2999 \
        run_pretrain.py --distributed $common_args
elif [[ $GPUS -eq 1 ]]; then
    OMP_NUM_THREADS=1 python run_pretrain.py $common_args
else
    OMP_NUM_THREADS=1 python run_pretrain.py --device cpu $common_args
fi
