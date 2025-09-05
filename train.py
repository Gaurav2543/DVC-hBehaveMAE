import argparse
import datetime
import json
import os
import time
from pathlib import Path # Added

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from iopath.common.file_io import g_pathmgr as pathmgr
from torch.utils.tensorboard import SummaryWriter

# Corrected import paths
from data_pipeline.dvc_dataset import DVCDataset # Specific DVC dataset
from engine.engine_pretrain import train_one_epoch   # New engine path
from models import models_defs                       # Core models
from util import misc as misc                        # Core utils
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import parse_tuples, str2bool

def get_args_parser():
    parser = argparse.ArgumentParser("DVC hBehaveMAE pre-training", add_help=False) # Renamed for clarity
    parser.add_argument(
        "--batch_size",
        default=4, # You had 4096 in your DVC script, adjust as needed
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # --- DVC Specific Dataset Args ---
    parser.add_argument(
        "--dataset", # Kept for consistency, but will be dvc
        default="dvc",
        type=str,
        help="Type of dataset [dvc]",
    )
    parser.add_argument(
        "--normalization_method",
        default="none",                 
        type=str,
        help=(
            "Normalisation to apply: "
            "[none | percentage | global_z_score | local_z_score]. "
            "'global_z_score' is synonymous with the old 'z_score'."
        ),
    )
    parser.add_argument(
        "--precomputed_stats_path",
        default=None, # e.g., "dvc_data_pipeline/dvc_zscore_stats.npy"
        type=str,
        help="Path to load/save Z-score mean/std.",
    )
    # --- End DVC Specific ---


    parser.add_argument("--sliding_window", default=7, type=int) # From your DVC script
    # fill_holes, data_augment, centeralign, include_test_data are less relevant for DVC as imputed
    parser.add_argument("--fill_holes", default=False, type=str2bool, help="DVC data is pre-imputed")
    parser.add_argument("--data_augment", default=False, type=str2bool, help="Augmentation not typical for DVC sensor data")


    # Model parameters
    parser.add_argument(
        "--model",
        default="hbehavemae",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--non_hierarchical", default=False, type=str2bool)

    parser.add_argument("--mask_ratio", default=0.70, type=float, help="Masking ratio.") # From DVC script
    parser.add_argument("--masking_strategy", default="random", type=str)
    parser.add_argument("--decoding_strategy", default="single", type=str, help="[multi, single]") # From DVC script
    parser.add_argument("--decoder_embed_dim", default=128, type=int)
    parser.add_argument("--decoder_depth", default=1, type=int)
    parser.add_argument("--decoder_num_heads", default=1, type=int)
    parser.add_argument("--num_frames", default=1440, type=int) # From DVC script
    parser.add_argument("--checkpoint_period", default=20, type=int)
    parser.add_argument("--sampling_rate", default=1, type=int) # DVC default frame rate is 4
    parser.add_argument("--distributed", action="store_true") # Keep this for multi-GPU

    # hBehaveMAE specific parameters (defaults from your DVC script)
    parser.add_argument("--input_size", default=(1440, 1, 12), nargs="+", type=int)
    parser.add_argument("--stages", default=(2, 3, 4), nargs="+", type=int)
    parser.add_argument("--q_strides", default=[(15,1,1),(6,1,1)], type=parse_tuples) # Adjusted for DVC
    parser.add_argument("--mask_unit_attn", default=[True, False, False], nargs="+", type=str2bool)
    parser.add_argument("--patch_kernel", default=(2, 1, 12), nargs="+", type=int)
    parser.add_argument("--init_embed_dim", default=96, type=int)
    parser.add_argument("--init_num_heads", default=2, type=int)
    parser.add_argument("--out_embed_dims", default=(64, 96, 128), nargs="+", type=int)
    parser.add_argument("--norm_loss", default=False, type=str2bool) # From DVC script

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=1.6e-4, metavar="LR") # From DVC script
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=40, metavar="N") # From DVC script
    
    parser.add_argument("--path_to_data_dir", default="/scratch/bhole/train_files", help="path where to load data from") # From DVC script
    parser.add_argument("--summary_csv", default="/scratch/bhole/dvc_data/smoothed/1440/final_summary_metadata_1440.csv", help="path to the summary CSV file") # From DVC script
    parser.add_argument("--output_dir", default="./outputs", help="path where to save") # From DVC script
    parser.add_argument("--log_dir", default="./logs", help="path where to tensorboard log") # From DVC script
    
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory.")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--no_env", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    parser.add_argument("--clip_grad", type=float, default=0.02) # From DVC script
    parser.add_argument("--no_qkv_bias", action="store_true")
    parser.add_argument("--bias_wd", action="store_true")
    parser.add_argument("--num_checkpoint_del", default=20, type=int)
    parser.add_argument("--sep_pos_embed", action="store_true")
    parser.set_defaults(sep_pos_embed=True)
    parser.add_argument("--trunc_init", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.set_defaults(fp32=True)
    parser.add_argument("--beta", default=None, type=float, nargs="+")
    parser.add_argument("--print_freq", default=20, type=int, help="Frequency of printing training logs")


    return parser
        
def main(args):
    misc.init_distributed_mode(args)
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # --- Dataset Setup for DVC ---
    if args.dataset.lower() != "dvc":
        raise ValueError("This script is configured for the DVC dataset only.")
    
    # --- MODIFICATION FOR STATS FILE HANDLING ---
    zscore_stats_file_path = args.precomputed_stats_path
    
    if args.normalization_method in {'z_score', 'global_z_score'}:
        if misc.is_main_process(): # Only rank 0 calculates and saves
            print(f"Rank {misc.get_rank()}: Checking/Creating Z-score stats file: {zscore_stats_file_path}")
            # Temporarily create a DVCDataset instance JUST to calculate/save stats
            # if the file doesn't exist. This instance won't be used for actual training.
            if not (zscore_stats_file_path and os.path.exists(zscore_stats_file_path)):
                print(f"Rank {misc.get_rank()}: Stats file not found or path not given. Calculating on rank 0...")
                temp_stats_dataset = DVCDataset(
                    mode="pretrain", # To trigger calculation logic
                    path_to_data_dir=args.path_to_data_dir,
                    summary_csv=args.summary_csv, # Path to summary CSV
                    # Provide minimal args needed for data loading and stat calculation
                    sampling_rate=args.sampling_rate, 
                    normalization_method='z_score', # Force this for calculation
                    precomputed_stats_path=zscore_stats_file_path, # Path to save
                    # These might not be strictly needed for just stat calculation
                    # but add them if DVCDataset __init__ or load_data depends on them
                    num_frames=args.num_frames, 
                    sliding_window=args.sliding_window,
                )
                # _prepare_zscore_stats would have been called and saved the file.
                del temp_stats_dataset # Clean up
                print(f"Rank {misc.get_rank()}: Stats file should now be saved at {zscore_stats_file_path}")
            else:
                print(f"Rank {misc.get_rank()}: Using existing Z-score stats file: {zscore_stats_file_path}")
        
        if args.distributed:
            torch.distributed.barrier() # All processes wait here until rank 0 is done
                                       # This ensures the file is created before others try to load it.
        # --- MODIFICATION: RETRY LOOP FOR FILE CHECK ---
        file_found = False
        max_retries = 5
        retry_delay = 0.5 # seconds
        for attempt in range(max_retries):
            if zscore_stats_file_path and os.path.exists(zscore_stats_file_path):
                file_found = True
                print(f"Rank {misc.get_rank()}: Confirmed Z-score stats file {zscore_stats_file_path} exists (attempt {attempt+1}).")
                break
            else:
                print(f"Rank {misc.get_rank()}: Z-score stats file {zscore_stats_file_path} not yet visible (attempt {attempt+1}/{max_retries}). Waiting {retry_delay}s...")
                time.sleep(retry_delay)
        
        if not file_found:
            raise FileNotFoundError(f"Rank {misc.get_rank()}: Z-score stats file {zscore_stats_file_path} still not found after {max_retries} retries and barrier. This should not happen.")
        # --- END MODIFICATION ---
        
        # Now, all processes should be able to find the file
        if not (zscore_stats_file_path and os.path.exists(zscore_stats_file_path)):
            raise FileNotFoundError(f"Rank {misc.get_rank()}: Z-score stats file {zscore_stats_file_path} still not found after barrier. This should not happen.")
    # --- END MODIFICATION ---

    dataset_train = DVCDataset(
        mode="pretrain",
        path_to_data_dir=args.path_to_data_dir,
        num_frames=args.num_frames,
        sliding_window=args.sliding_window,
        sampling_rate=args.sampling_rate, # Make sure this aligns with DVC data characteristics
        summary_csv=args.summary_csv, # Path to summary CSV
        normalization_method=args.normalization_method,
        precomputed_stats_path=args.precomputed_stats_path,
    )
    # --- End Dataset Setup ---

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        try:
            pathmgr.mkdirs(args.log_dir)
        except Exception as _:
            pass
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    model = models_defs.__dict__[args.model](**vars(args))
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        fup = True if args.decoding_strategy == "single" else False
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()], find_unused_parameters=fup
        )
        model_without_ddp = model.module

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay, bias_wd=args.bias_wd)
    beta = (0.9, 0.95) if args.beta is None else args.beta
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=beta)
    loss_scaler = NativeScaler(fp32=args.fp32)

    misc.load_model(
        args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler
    )

    checkpoint_path = "" # Initialize checkpoint_path
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer, # data_loader_val was here, now passed directly
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
            fp32=args.fp32
        )
        
        if args.output_dir and (epoch % args.checkpoint_period == 0 or epoch + 1 == args.epochs):
            checkpoint_path = misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
            )

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        # if 'val_loss' in train_stats: # Add val_loss to log_stats if it exists
        #     log_stats['val_loss'] = train_stats['val_loss']
            
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with pathmgr.open(f"{args.output_dir}/log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    if torch.cuda.is_available(): # Check if CUDA is available before printing memory
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
    return [checkpoint_path]


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)