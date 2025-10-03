#!/bin/bash

GPUS=$1
common_args="--dataset dvc \
    --path_to_data_dir /scratch/bhole/dvc_data_normal/cage_activations \
    --batch_size 16384 \
    --model hbehavemae \
    --input_size 1440 1 12 \
    --stages 2 3 4 \
    --q_strides 15,1,1;6,1,1 \
    --mask_unit_attn True False False \
    --patch_kernel 2 1 12 \
    --init_embed_dim 96 \
    --init_num_heads 2 \
    --out_embed_dims 64 96 128 \
    --epochs 100 \
    --num_frames 1440 \
    --decoding_strategy single \
    --decoder_embed_dim 128 \
    --decoder_depth 1 \
    --decoder_num_heads 1 \
    --pin_mem \
    --num_workers 8 \
    --sliding_window 7 \
    --blr 1e-4 \
    --warmup_epochs 25 \
    --masking_strategy random \
    --mask_ratio 0.75 \
    --clip_grad 0.02 \
    --checkpoint_period 5 \
    --norm_loss False \
    --seed 0 \
    --output_dir /scratch/bhole/dvc_data_normal/models_3days \
    --log_dir logs \
    --normalization_method none" 
    # --precomputed_stats_path /scratch/bhole/z_score_stats/dvc_z_score.npz" 

torchrun --master_addr=127.0.0.1 --master_port=2998 train.py $common_args