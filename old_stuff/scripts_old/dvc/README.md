# What do arguments in training script mean?

The `train_hBehaveMAE.sh` script provides training configurations for the h/BehaveMAE models. Below are the available arguments and their descriptions.

## Data Configuration
- `--dataset`: Dataset type (e.g.: `dvc`)
- `--path_to_data_dir`: Path to training data directory
- `--batch_size`: Number of samples per training iteration
- `--num_frames`: Length of input sample sequence (e.g. full day: `1440`)
- `--sliding_window`: Sliding window size for creating sequence samples (e.g.: `7`)
- `--input_size`: Size of sample [length x individuals x features] (e.g.: `1440 1 12`)

## Model Architecture
- `--model`: Model architecture to use (default: `hbehavemae`)
- `--stages`: Number of transformer layers per hierarchical block (default: `2 3 4`)
- `--patch_kernel`: Size of token in first level [t x n x f] (e.g.: `2 1 12`)
  - Alternatives:
    - `1 4 12`: Every frame has its own token
    - `4 1 6`: Split into upper and lower half of cage
- `--q_strides`: Strides to fuse tokens between hierarchical levels (e.g.: `15,1,1;4,1,1`)
  - Format: semicolon-separated values for each block transition
  - Example: `2,1,1` transforms `4 1 12 -> 8 1 12`

### Attention Configuration
- `--mask_unit_attn`: Local/global attention for each block (default: `True False False`)
- `--init_embed_dim`: Initial learned representation size (e.g.: `96`)
  - Each subsequent block has 2x this size
- `--init_num_heads`: Initial number of attention heads (e.g.: `2`)
  - Each subsequent block has 2x this size
- `--out_embed_dims`: Dimensions after linear compression per block (default: `64 96 128`)

### Decoder Settings
- `--decoding_strategy`: Training decoding strategy (default: `single`)
  - `single`: Use only last block
  - `multi`: Use all blocks
- `--decoder_embed_dim`: Decoder embedding size (e.g.: `128`)
- `--decoder_depth`: Number of decoder layers (default: `1`)
- `--decoder_num_heads`: Number of decoder attention heads (default: `1`)

## Training Parameters
- `--epochs`: Number of training epochs (default: `200`)
- `--blr`: Base learning rate (default: `1.6e-4`)
- `--warmup_epochs`: Learning rate warmup epochs (default: `40`)
- `--masking_strategy`: Token masking strategy (default: `random`)
- `--mask_ratio`: Ratio of tokens to mask during training (default: `0.70`)
- `--clip_grad`: Gradient clipping value (default: `0.02`)
- `--checkpoint_period`: Checkpoint save frequency in epochs (default: `20`)
- `--norm_loss`: Whether to normalize loss (default: `False`)
- `--seed`: Random seed (default: `0`)

## System Settings
- `--num_workers`: Number of data loading workers (default: `8`)
- `--pin_mem`: Enable memory pinning
- `--output_dir`: Directory for model outputs
- `--log_dir`: Directory for training logs

## Usage

For single GPU training:
```bash
OMP_NUM_THREADS=1 python run_pretrain.py $common_args
```

For multi-GPU training:
```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --node_rank 0 --master_addr=127.0.0.1 --master_port=2999 \
    run_pretrain.py --distributed $common_args
```