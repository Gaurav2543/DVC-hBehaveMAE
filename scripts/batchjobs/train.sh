#!/bin/bash
#SBATCH --job-name=4w_train
#SBATCH --output=logs_4weeks/train_hBehaveMAE_%j.out  # Save output logs
#SBATCH --error=logs_4weeks/train_hBehaveMAE_%j.err   # Save error logs
#SBATCH --gres=gpu:1  # Number of GPUs requested
#SBATCH --cpus-per-task=8  # Number of CPU threads
#SBATCH --mem=256G  # Memory requested
#SBATCH --time=72:00:00  # Time limit (adjust as needed)
#SBATCH --partition=h100  # Change this based on your cluster’s GPU partition
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --nodes=1  # Run on a single node

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

GPUS=1  

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# bash scripts/train_hbehavemae.sh $GPUS

echo "Extracting embeddings..."
# python3 downstream_tasks/extract_embeddings.py --dvc_root /scratch/bhole/dvc_data/smoothed/cage_activations_full_data_final \
python3 downstream_tasks/extract_embeddings_multi.py --dvc_root /scratch/bhole/dvc_data/smoothed/cage_activations_full_data_final \
    --summary_csv /scratch/bhole/dvc_data/smoothed/40320/summary_metadata_40320.csv \
    --ckpt_dir /scratch/bhole/dvc_data/smoothed/models_4weeks --ckpt_name checkpoint-09999.pth \
    --output_dir /scratch/bhole/dvc_data/smoothed/models_4weeks/embeddings_multi --time_aggregations 40320 \
    --aggregation_names 1day --batch_size 256 \
    --model hbehavemae \
    --input_size 40320 1 12 \
    --stages 2 3 4 4 5 5 6 \
    --q_strides "6,1,1;4,1,1;3,1,1;4,1,1;2,1,1;2,1,1" \
    --mask_unit_attn True False False False False False False \
    --patch_kernel 5 1 12 \
    --init_embed_dim 96 \
    --init_num_heads 2 \
    --out_embed_dims 128 160 192 192 224 224 256 \
    --decoding_strategy single \
    --decoder_embed_dim 128 \
    --decoder_depth 1 \
    --decoder_num_heads 1 \
    --num_workers 8 
echo "Embedding extraction completed"

