#!/bin/bash
#SBATCH --job-name=opt_4w_e_embeds
#SBATCH --output=../generated_outputs/opt_4w_embeds.out
#SBATCH --error=../generated_outputs/opt_4w_embeds.err
#SBATCH --cpus-per-task=32
#SBATCH --partition=h100
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --exclusive

# Clear any existing CUDA settings
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=8
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# Print initial GPU status
echo "=== Initial GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo ""

# Print Python's view of GPUs
python3 -c "import torch; print(f'PyTorch sees {torch.cuda.device_count()} GPUs'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

echo "Starting embedding extraction..."

python3 downstream_tasks/opt_extract_embeddings.py \
    --dvc_root /scratch/bhole/dvc_data/smoothed/cage_activations_full_data_final \
    --summary_csv /scratch/bhole/dvc_data/smoothed/40320/summary_metadata_40320.csv \
    --multi_gpu \
    --gpu_ids 0 1 2 3 \
    --ckpt_dir /scratch/bhole/dvc_data/smoothed/models_4weeks \
    --ckpt_name checkpoint-09999.pth \
    --output_dir /scratch/bhole/dvc_data/smoothed/models_4weeks/embeddings_full_new \
    --time_aggregations 40320 \
    --aggregation_names 4weeks \
    --batch_size 32 \
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
    --num_workers 4

echo "=== Final GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

echo "Embedding extraction completed"