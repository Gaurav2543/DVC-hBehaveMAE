# dvc_hbehavemae_project/extract_dvc_embeddings_simplified.py
from __future__ import annotations

import argparse
import datetime
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Assuming these are in the new structure relative to this script's execution path
# or dvc_hbehavemae_project is in PYTHONPATH
from data_pipeline.load_dvc import load_dvc_data # Corrected import path
from models import models_defs
from util import misc as misc
from util.misc import parse_tuples, str2bool
from util.pos_embed import interpolate_pos_embed


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("DVC GenHiera Simplified Embedding Extractor", add_help=False)
    # I/O
    parser.add_argument("--dvc_data_root_dir", required=True, type=str)
    parser.add_argument("--dvc_summary_table_filename", default="summary_table_imputed_with_sets_sub_20_CompleteAge_Strains.csv")
    parser.add_argument("--model_checkpoint_dir", required=True, type=str)
    parser.add_argument("--model_checkpoint_filename", default="checkpoint-best.pth")
    parser.add_argument("--embeddings_output_dir", required=True, type=str)
    # Normalization
    parser.add_argument("--normalization_method", choices=["percentage", "z_score"], default="percentage")
    parser.add_argument("--precomputed_stats_path", type=str, default=None, help="For z-score")
    # Model Config (must match training of the loaded checkpoint)
    parser.add_argument("--model_name", default="gen_hiera", help="Should be 'gen_hiera' for this script")
    parser.add_argument("--input_size", default=(1440, 1, 12), nargs="+", type=int)
    parser.add_argument("--patch_kernel", default=(2, 1, 12), nargs="+", type=int)
    parser.add_argument("--stages", default=(2, 3, 4), nargs="+", type=int)
    parser.add_argument("--q_strides", default=[(15, 1, 1), (6, 1, 1)], type=parse_tuples)
    parser.add_argument("--init_embed_dim", default=96, type=int)
    parser.add_argument("--init_num_heads", default=2, type=int)
    parser.add_argument("--out_embed_dims", default=(64, 96, 128), nargs="+", type=int) # Output dim per stage
    parser.add_argument("--mask_unit_attn", default=[True, False, False], nargs="+", type=str2bool)
    parser.add_argument("--sep_pos_embed", default=True, type=str2bool)
    parser.add_argument("--no_qkv_bias", action="store_true")
    parser.add_argument("--non_hierarchical", default=False, type=str2bool) # For gen_hiera, usually False
    # Extraction Behavior
    parser.add_argument("--num_frames_per_window", default=1440, type=int, help="Length of sequence fed to model each time (window length)")
    parser.add_argument("--sliding_window_step", default=1, type=int, help="Step for sliding window over input sequences")
    # Fast inference is assumed (central token selection)
    parser.add_argument("--combine_stages", default=True, type=str2bool, help="Concatenate embeddings from different stages")
    parser.add_argument("--pca_target_dim", default=64, type=int, help="Target dim for PCA if combined embeddings are too large")
    # System
    parser.add_argument("--batch_size_inf", default=8, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--fp16", default=True, type=str2bool)
    parser.add_argument("--skip_missing_electrodes", action="store_true")
    return parser


def load_dvc_sequences_for_test(data_root: str, summary_csv: str, skip_missing: bool) -> List[Tuple[str, np.ndarray]]:
    """Loads DVC data for sequences marked with sets==1."""
    df = pd.read_csv(os.path.join(data_root, summary_csv))
    testing_dict: Dict[str, List[str]] = {}
    ordered_sequence_keys: List[str] = []

    for _, row in df.iterrows():
        if row.get("sets", np.nan) == 1: # 1 indicates test set
            cage_id, day_str = str(row["cage"]), str(row["day"])
            testing_dict.setdefault(cage_id, []).append(day_str)
            ordered_sequence_keys.append(f"{cage_id}_{day_str}")

    # loaded_data_frames_by_cage: Dict[str, pd.DataFrame]
    loaded_data_frames_by_cage = load_dvc_data(data_root, testing_dict)

    final_sequences: List[Tuple[str, np.ndarray]] = []
    
    # Define expected columns once (V1 to V12)
    # Assuming load_dvc_data has already selected appropriate columns,
    # but if not, this is where you'd ensure you get V0-V11 or similar.
    # For simplicity, assuming load_dvc_data returns DataFrames where
    # relevant electrode data is in the first 12 columns (or accessible via iloc[:, :12])
    # The original version had a complex way to map V_i, Vi etc.
    # Simplified: load_dvc_data should ideally return cleaned data (first 12 cols are electrodes)

    for cage_id, days_df in loaded_data_frames_by_cage.items():
        # If days_df contains multiple days, split by day
        # Assuming 'Timestamp' or a date column exists for splitting.
        # The original load_datav2 in dvc_dataset.py implies it returns data per CAGE,
        # and then that data is split by day if multiple test days for a cage.
        # Let's assume `days_df` is for ONE cage but potentially MULTIPLE days.
        # We need to ensure each day becomes a separate sequence.
        days_df["__day_str__"] = pd.to_datetime(days_df.iloc[:, 13]).dt.strftime('%Y-%m-%d') # Assuming 14th col is timestamp

        for day_str, day_group_df in days_df.groupby("__day_str__"):
            seq_key = f"{cage_id}_{day_str}"
            if seq_key in ordered_sequence_keys: # Process only if it was in our initial test set list
                # Assuming electrode data is in the first 12 columns
                # Your previous code used iloc[:, 0:-2] on values, check if this is still right
                electrode_data = day_group_df.iloc[:, 0:12].values.astype(np.float32)
                if electrode_data.shape[1] != 12 and skip_missing:
                    print(f"Warning: Sequence {seq_key} has {electrode_data.shape[1]} features, expected 12. Skipping.")
                    continue
                elif electrode_data.shape[1] != 12:
                     raise ValueError(f"Sequence {seq_key} has {electrode_data.shape[1]} features, expected 12.")
                final_sequences.append((seq_key, electrode_data))

    # Ensure the order matches ordered_sequence_keys
    final_sequences.sort(key=lambda item: ordered_sequence_keys.index(item[0]))
    return final_sequences


def load_pretrained_encoder(args, device: torch.device) -> nn.Module:
    if args.model_name != "gen_hiera":
        print(f"Warning: This script is optimized for 'gen_hiera'. Using '{args.model_name}' may require adjustments.")
    
    model = models_defs.__dict__[args.model_name](**vars(args)) # Build a GenHiera instance

    ckpt_path = os.path.join(args.model_checkpoint_dir, args.model_checkpoint_filename)
    if args.model_checkpoint_filename == "checkpoint-best.pth" or not os.path.exists(ckpt_path):
        best_path = os.path.join(args.model_checkpoint_dir, "checkpoint-best.pth")
        if os.path.exists(best_path): ckpt_path = best_path
        else:
            last_ckpt = misc.get_last_checkpoint(args.model_checkpoint_dir)
            if last_ckpt: ckpt_path = last_ckpt
            else: raise FileNotFoundError(f"No checkpoint found in {args.model_checkpoint_dir}")
    
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    checkpoint_model = checkpoint.get("model", checkpoint.get("model_state", checkpoint))

    interpolate_pos_embed(model, checkpoint_model)
    model_state_dict = misc.convert_checkpoint(checkpoint_model)
    
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    print(f"[Model Load] Missing Keys: {missing_keys if missing_keys else 'None'}")
    print(f"[Model Load] Unexpected Keys: {unexpected_keys if unexpected_keys else 'None'}")
    
    model.to(device).eval()
    return model


def main(args):
    t_start = time.time()
    device = torch.device(args.device)
    Path(args.embeddings_output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load DVC test sequences
    print("Loading DVC test data...")
    # List of (sequence_id_str, numpy_array[Time, 12_Features])
    all_test_sequences = load_dvc_sequences_for_test(
        args.dvc_data_root_dir, args.dvc_summary_table_filename, args.skip_missing_electrodes
    )
    if not all_test_sequences:
        print("No test sequences loaded. Exiting.")
        return

    # 2. Prepare Normalization
    norm_mean, norm_std = None, None
    if args.normalization_method == "z_score":
        if not args.precomputed_stats_path or not os.path.exists(args.precomputed_stats_path):
            raise FileNotFoundError(f"Z-score stats file not found: {args.precomputed_stats_path}")
        stats = np.load(args.precomputed_stats_path, allow_pickle=True).item()
        norm_mean, norm_std = stats["mean"], stats["std"]
        if np.any(norm_std == 0): norm_std[norm_std == 0] = 1e-6 # Avoid division by zero
        print(f"Using Z-score normalization with Mean: {norm_mean}, Std: {norm_std}")

    # 3. Load Model
    print("Loading pre-trained encoder model...")
    encoder_model = load_pretrained_encoder(args, device)

    # --- Store embeddings for each stage and combined ---
    # Key: stage_index (0, 1, 2) or "combined"
    # Value: List of numpy arrays, each array is (n_frames_for_seq, embedding_dim_for_stage)
    collected_embeddings_per_stage: Dict[str | int, List[np.ndarray]] = {
        stage_idx: [] for stage_idx in range(len(args.stages))
    }
    if args.combine_stages:
        collected_embeddings_per_stage["combined"] = []
    
    frame_number_map: Dict[str, Tuple[int, int]] = {}
    current_global_frame_index = 0

    # 4. Process each sequence
    for seq_id, seq_data_raw in tqdm(all_test_sequences, desc="Processing sequences"):
        # Normalize
        if args.normalization_method == "percentage":
            seq_data_normalized = seq_data_raw / 100.0
        elif args.normalization_method == "z_score":
            seq_data_normalized = (seq_data_raw - norm_mean) / norm_std
        else: # Should not happen
            seq_data_normalized = seq_data_raw
        
        seq_data_normalized = seq_data_normalized.astype(np.float32)
        current_seq_full_len = seq_data_normalized.shape[0]

        # Prepare windows for this sequence
        # Pad for sliding window to ensure central alignment for fast inference
        # Pad amount ensures that the *center* of the num_frames_per_window aligns with original frames
        pad_amount = (args.num_frames_per_window - args.sliding_window_step) // 2
        padded_seq = np.pad(seq_data_normalized, ((pad_amount, pad_amount), (0, 0)), mode="edge")

        # Create sliding windows: shape (num_windows, window_T, num_features=12)
        seq_windows_np = sliding_window_view(
            padded_seq,
            window_shape=(args.num_frames_per_window, seq_data_normalized.shape[1]), # (window_T, 12)
            axis=(0,1) # This means each view element IS the window
        )[::args.sliding_window_step, 0] # Step and remove redundant middle dimension from view
        
        # Reshape for model: (num_windows, window_T, 1_Individual_H, 12_Electrodes_W)
        seq_windows_np_reshaped = np.ascontiguousarray(
            seq_windows_np.reshape(-1, args.num_frames_per_window, 1, 12)
        )
        
        seq_windows_torch = torch.from_numpy(seq_windows_np_reshaped)
        if args.fp16:
            seq_windows_torch = seq_windows_torch.half()
        
        window_dataset = TensorDataset(seq_windows_torch)
        window_loader = DataLoader(
            window_dataset, batch_size=args.batch_size_inf, shuffle=False, num_workers=args.num_workers
        )

        # Store embeddings from all batches for this sequence, per stage
        batch_embeddings_this_seq_per_stage: Dict[int, List[torch.Tensor]] = {
             stage_idx: [] for stage_idx in range(len(args.stages))
        }

        with torch.no_grad():
            for (batch_window_data,) in window_loader: # DataLoader wraps it in a tuple
                # Prepare for model: (B, C_in=1, T_window, H_Ind=1, W_Elec=12)
                model_input_batch = batch_window_data.unsqueeze(1).to(device)
                
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    if args.model_name == "hbehavemae":
                        # GenHiera.forward(x, mask=None)
                        _, intermediate_stage_outputs = encoder_model(model_input_batch, mask=None)
                    else:
                        # GenHiera.forward(x, mask=None, return_intermediates=True)
                        _, intermediate_stage_outputs = encoder_model(
                            model_input_batch, mask=None, return_intermediates=True
                        )
                # intermediate_stage_outputs is a list of tensors from projection layers
                # Each tensor: (B_batch, T_out_stage, H_out_stage, W_out_stage, C_stage)
                for stage_idx, stage_output_batch in enumerate(intermediate_stage_outputs):
                    # Fast inference: select central temporal token from T_out_stage
                    # Then average over H_out_stage, W_out_stage if they are not 1
                    central_t_idx = stage_output_batch.shape[1] // 2
                    # features_batch shape: (B_batch, H_out_stage, W_out_stage, C_stage)
                    features_batch = stage_output_batch[:, central_t_idx, ...] 
                    
                    if features_batch.ndim == 4: # B, H, W, C
                        final_features_batch = torch.mean(features_batch, dim=[1, 2]) # Avg H, W -> (B, C)
                    elif features_batch.ndim == 3: # B, H_or_W, C
                        final_features_batch = torch.mean(features_batch, dim=1)       # Avg H_or_W -> (B, C)
                    elif features_batch.ndim == 2: # B, C
                        final_features_batch = features_batch
                    else:
                        raise ValueError("Unexpected feature tensor ndim after time selection.")
                    batch_embeddings_this_seq_per_stage[stage_idx].append(final_features_batch.cpu())
                torch.cuda.empty_cache()
        
        # Concatenate batch results for this sequence and store
        seq_final_embeddings_per_stage: Dict[int, np.ndarray] = {}
        for stage_idx in range(len(args.stages)):
            stage_embs_np = torch.cat(batch_embeddings_this_seq_per_stage[stage_idx], dim=0).numpy()
            # Ensure length matches current_seq_full_len by repeating/truncating if step > 1
            if args.sliding_window_step > 1 and stage_embs_np.shape[0] < current_seq_full_len :
                stage_embs_np = np.repeat(stage_embs_np, args.sliding_window_step, axis=0)[:current_seq_full_len]
            elif stage_embs_np.shape[0] > current_seq_full_len: # Should not happen if padding/step is right
                stage_embs_np = stage_embs_np[:current_seq_full_len]
            
            seq_final_embeddings_per_stage[stage_idx] = stage_embs_np
            collected_embeddings_per_stage[stage_idx].append(stage_embs_np)

        if args.combine_stages:
            combined_for_seq = np.concatenate(
                [seq_final_embeddings_per_stage[stage_idx] for stage_idx in range(len(args.stages))],
                axis=1 # Concatenate features
            )
            collected_embeddings_per_stage["combined"].append(combined_for_seq)

        frame_number_map[seq_id] = (current_global_frame_index, current_global_frame_index + current_seq_full_len)
        current_global_frame_index += current_seq_full_len

    # 5. Finalize and Save Embeddings
    print("Finalizing and saving embeddings...")
    for stage_key, list_of_seq_arrays in collected_embeddings_per_stage.items():
        if not list_of_seq_arrays:
            print(f"No embeddings collected for stage {stage_key}. Skipping.")
            continue
        
        # Concatenate embeddings from all sequences for this stage
        stage_all_frames_embeddings = np.concatenate(list_of_seq_arrays, axis=0).astype(np.float16)
        print(f"Stage {stage_key}: Final aggregated shape {stage_all_frames_embeddings.shape}")

        # PCA if needed
        if stage_all_frames_embeddings.shape[1] > args.pca_target_dim:
            print(f"Applying PCA to stage {stage_key}: {stage_all_frames_embeddings.shape[1]} -> {args.pca_target_dim}")
            # Using IncrementalPCA for potentially large arrays
            ipca = IncrementalPCA(n_components=args.pca_target_dim, batch_size=min(4096, stage_all_frames_embeddings.shape[0]))
            # PCA requires float32
            pca_transformed = ipca.fit_transform(stage_all_frames_embeddings.astype(np.float32))
            print(f"  PCA Explained Variance: {ipca.explained_variance_ratio_.sum():.4f}")
            stage_all_frames_embeddings = pca_transformed.astype(np.float16)
        
        output_content = {
            "frame_number_map": frame_number_map,
            "embeddings": stage_all_frames_embeddings
        }
        save_filename = f"test_submission_stage_{stage_key}.npy"
        save_path = os.path.join(args.embeddings_output_dir, save_filename)
        np.save(save_path, output_content)
        print(f"  Saved to {save_path}")

    t_end = time.time()
    print(f"Total embedding extraction runtime: {str(datetime.timedelta(seconds=int(t_end - t_start)))}")


if __name__ == "__main__":
    cli_args = get_args_parser().parse_args()
    main(cli_args)