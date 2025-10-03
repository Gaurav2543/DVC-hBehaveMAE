#!/bin/bash
#SBATCH --job-name=4w_ddp_embeds
#SBATCH --output=../generated_outputs/4w_embeds_ddp.out
#SBATCH --error=../generated_outputs/4w_embeds_ddp.err
#SBATCH --cpus-per-task=32
#SBATCH --partition=h100
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --mem=256G
#SBATCH --nodes=4
#SBATCH --exclusive

# Environment setup for DDP
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO  # Enable NCCL debugging (can remove after testing)

# Print GPU information
echo "=== GPU Information ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Check PyTorch DDP setup
python3 -c "
import torch
import torch.distributed as dist
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'NCCL available: {torch.distributed.is_nccl_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""
echo "Starting DDP embedding extraction..."

# Run with DDP - the script will spawn processes internally
python3 downstream_tasks/extract_embeddings_chunked.py \
    --dvc_root /scratch/bhole/dvc_data/smoothed/cage_activations_full_data_final \
    --summary_csv /scratch/bhole/dvc_data/smoothed/40320/summary_metadata_40320.csv \
    --ckpt_dir /scratch/bhole/dvc_data/smoothed/models_4weeks \
    --ckpt_name checkpoint-09999.pth \
    --output_dir /scratch/bhole/dvc_data/smoothed/models_4weeks/embeddings_subsampled \
    --time_aggregations 40320 \
    --aggregation_names 4weeks \
    --batch_size 1 \
    --max_sequence_length 40320 \
    --world_size 4 \
    --master_port 12355 \
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

echo ""
echo "=== Final GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

echo "DDP embedding extraction completed"