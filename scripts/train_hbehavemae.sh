#!/bin/bash
common_args="--dataset dvc \
    --path_to_data_dir /scratch/bhole/dvc_data/smoothed/cage_activations_full_data_final \
    --batch_size 16 \
    --model hbehavemae \
    --input_size 40320 1 12 \
    --stages 2 3 4 4 5 5 6 \
    --q_strides 6,1,1;4,1,1;3,1,1;4,1,1;2,1,1;2,1,1 \
    --mask_unit_attn True False False False False False False \
    --patch_kernel 5 1 12 \
    --init_embed_dim 96 \
    --init_num_heads 2 \
    --out_embed_dims 128 160 192 192 224 224 256 \
    --epochs 10000 \
    --num_frames 40320 \
    --decoding_strategy single \
    --decoder_embed_dim 128 \
    --decoder_depth 1 \
    --decoder_num_heads 1 \
    --pin_mem \
    --num_workers 8 \
    --sliding_window 15 \
    --blr 1e-4 \
    --warmup_epochs 50 \
    --masking_strategy random \
    --mask_ratio 0.8 \
    --clip_grad 0.02 \
    --checkpoint_period 50 \
    --norm_loss False \
    --output_dir /scratch/bhole/dvc_data/smoothed/models_4weeks \
    --log_dir logs_4weeks \
    --normalization_method none \
    --summary_csv /scratch/bhole/dvc_data/smoothed/40320/summary_metadata_40320.csv"

torchrun --master_addr=127.0.0.1 --master_port=2998 train.py $common_args