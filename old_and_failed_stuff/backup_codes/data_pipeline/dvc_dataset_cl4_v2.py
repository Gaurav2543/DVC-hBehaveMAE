from pathlib import Path
import os
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_pipeline.load_dvc import load_dvc_data
from data_pipeline.pose_traj_dataset import BasePoseTrajDataset

class DVCDataset(BasePoseTrajDataset):
    """
    Corrected DVC dataset for hBehaveMAE.
    This version correctly handles discrete time-series chunks as defined in the summary CSV
    and maintains the 4x3 spatial grid structure.
    """
    DEFAULT_FRAME_RATE = 1/60  # 1 sample per minute
    NUM_KEYPOINTS = 12
    KPTS_DIMENSIONS = 1
    NUM_INDIVIDUALS = 1
    KEYFRAME_SHAPE = (NUM_INDIVIDUALS, 4, 3, KPTS_DIMENSIONS)  # 4x3 grid
    GRID_SHAPE = (4, 3)

    def __init__(
        self,
        mode: str,
        path_to_data_dir: Path,
        summary_csv: str,
        scale: bool = True,
        sampling_rate: int = 1,
        num_frames: int = 400,  # This is the "page" size for the model
        sliding_window: int = 1,
        normalization_method: str = 'percentage',
        precomputed_stats_path: str = None,
        max_samples: int = None,  # For testing with limited data
        **kwargs
    ):
        # Determine the length of each continuous data chunk from the filename
        match = re.search(r'_(\d+)\.csv', summary_csv)
        if match:
            self.SAMPLE_LEN = int(match.group(1))
            print(f"Data chunks are expected to be {self.SAMPLE_LEN} minutes long.")
        else:
            raise ValueError(f"Could not determine sample length from summary file: {summary_csv}")
        
        # Validate num_frames against SAMPLE_LEN
        if num_frames > self.SAMPLE_LEN:
            print(f"Warning: num_frames ({num_frames}) > SAMPLE_LEN ({self.SAMPLE_LEN})")
            print(f"Setting num_frames to SAMPLE_LEN ({self.SAMPLE_LEN})")
            num_frames = self.SAMPLE_LEN
        
        super().__init__(
            path_to_data_dir, scale, sampling_rate, num_frames, sliding_window, **kwargs
        )

        self.sample_frequency = self.DEFAULT_FRAME_RATE
        self.mode = mode
        self.normalization_method = normalization_method
        self.precomputed_stats_path = precomputed_stats_path
        self.max_samples = max_samples
        self.mean_val, self.std_val = None, None

        self.load_data(summary_csv)

        if self.normalization_method == 'z_score':
            self._prepare_zscore_stats()

        self.preprocess()

    def _prepare_zscore_stats(self):
        """Prepare statistics for z-score normalization"""
        if self.mode == 'pretrain':
            if self.precomputed_stats_path and os.path.exists(self.precomputed_stats_path):
                stats = np.load(self.precomputed_stats_path, allow_pickle=True).item()
                self.mean_val, self.std_val = stats['mean'], stats['std']
                print(f"Loaded Z-score stats: Mean={self.mean_val:.4f}, Std={self.std_val:.4f}")
            elif hasattr(self, 'raw_data') and self.raw_data:
                all_train_data = np.concatenate([seq.flatten() for seq in self.raw_data])
                self.mean_val = np.mean(all_train_data)
                self.std_val = np.std(all_train_data)
                if self.std_val == 0: 
                    self.std_val = 1e-6
                print(f"Calculated Z-score stats: Mean={self.mean_val:.4f}, Std={self.std_val:.4f}")
                
                if self.precomputed_stats_path:
                    os.makedirs(os.path.dirname(self.precomputed_stats_path), exist_ok=True)
                    np.save(self.precomputed_stats_path, {'mean': self.mean_val, 'std': self.std_val})
                    print(f"Saved Z-score stats to {self.precomputed_stats_path}")

    def load_data(self, summary_csv):
        """
        Load data by treating each row in the summary CSV as a distinct sample,
        ensuring only data specified in summary file is used.
        """
        print(f"Reading summary file from: {summary_csv}")
        summary_df = pd.read_csv(summary_csv, low_memory=False)
        
        # Limit samples for testing if specified
        if self.max_samples is not None:
            summary_df = summary_df.head(self.max_samples)
            print(f"Limited to {self.max_samples} samples for testing")
        
        # Validate required columns
        required_cols = ['cage_id', 'from_tpt', 'to_tpt']
        missing_cols = [col for col in required_cols if col not in summary_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in summary CSV: {missing_cols}")
        
        summary_df['from_tpt'] = pd.to_datetime(summary_df['from_tpt'])
        summary_df['to_tpt'] = pd.to_datetime(summary_df['to_tpt'])

        self.raw_data = []
        self.chunk_info = []  # Store metadata for validation
        loaded_chunks = 0
        skipped_chunks = 0
        
        # Group by cage to load per-minute data efficiently
        for cage_id, group in tqdm(summary_df.groupby('cage_id'), desc="Loading cage data"):
            try:
                # Load the raw per-minute data for the entire cage
                cage_file_path = os.path.join(self.path, f"{cage_id}.csv")
                if not os.path.exists(cage_file_path):
                    print(f"Warning: Raw data file for cage {cage_id} not found at {cage_file_path}. Skipping.")
                    skipped_chunks += len(group)
                    continue
                
                raw_cage_df = pd.read_csv(cage_file_path, parse_dates=['timestamp'])
                raw_cage_df = raw_cage_df.set_index('timestamp')
                
                # Validate electrode columns exist
                electrode_cols = [f"v_{i}" for i in range(1, 13)]
                missing_electrodes = [col for col in electrode_cols if col not in raw_cage_df.columns]
                if missing_electrodes:
                    print(f"Warning: Missing electrode columns in {cage_id}: {missing_electrodes}. Skipping cage.")
                    skipped_chunks += len(group)
                    continue
                
            except Exception as e:
                print(f"Error loading cage data for {cage_id}: {e}. Skipping.")
                skipped_chunks += len(group)
                continue

            # Process each chunk defined in the summary file for this cage
            for _, row in group.iterrows():
                start_time, end_time = row['from_tpt'], row['to_tpt']
                
                try:
                    # Slice the raw data to get the exact time chunk
                    chunk_df = raw_cage_df[start_time:end_time]
                    
                    # Strict validation of chunk length
                    if len(chunk_df) != self.SAMPLE_LEN:
                        print(f"Warning: Chunk for {cage_id} from {start_time} has length {len(chunk_df)}, "
                              f"expected {self.SAMPLE_LEN}. Skipping.")
                        skipped_chunks += 1
                        continue
                    
                    # Extract electrode data and validate
                    chunk_data = chunk_df[electrode_cols].values.astype(np.float32)
                    
                    # Check for missing data
                    if np.isnan(chunk_data).any():
                        nan_count = np.isnan(chunk_data).sum()
                        print(f"Warning: Chunk for {cage_id} from {start_time} contains {nan_count} NaN values. Skipping.")
                        skipped_chunks += 1
                        continue
                    
                    # Additional validation: check data shape
                    if chunk_data.shape != (self.SAMPLE_LEN, 12):
                        print(f"Warning: Unexpected data shape {chunk_data.shape} for cage {cage_id}. Skipping.")
                        skipped_chunks += 1
                        continue
                    
                    # Store the chunk data and metadata
                    self.raw_data.append(chunk_data)
                    self.chunk_info.append({
                        'cage_id': cage_id,
                        'start_time': start_time,
                        'end_time': end_time,
                        'length': len(chunk_data)
                    })
                    loaded_chunks += 1
                    
                except Exception as e:
                    print(f"Error processing chunk for {cage_id} from {start_time}: {e}. Skipping.")
                    skipped_chunks += 1
                    continue

        print(f"Data loading complete:")
        print(f"  Successfully loaded: {loaded_chunks} chunks")
        print(f"  Skipped: {skipped_chunks} chunks")
        print(f"  Total chunks in summary: {len(summary_df)}")
        
        if loaded_chunks == 0:
            raise ValueError("No valid data chunks were loaded. Check your data files and summary CSV.")

    def preprocess(self):
        """
        Preprocess the loaded chunks into the format expected by hBehaveMAE.
        Maintains the 4x3 spatial grid structure.
        """
        if not hasattr(self, 'raw_data') or not self.raw_data:
            print("No raw data to preprocess.")
            self.seq_keypoints, self.keypoints_ids, self.items = [], [], []
            self.n_frames = 0
            return
        
        print("Preprocessing data chunks...")
        
        # Convert each chunk to the proper spatial grid format
        sequences_gridded = []
        for i, data_chunk in enumerate(self.raw_data):
            try:
                # Reshape from (T, 12) to (T, 1, 4, 3, 1) to preserve 4x3 grid
                gridded_chunk = data_chunk.reshape(
                    self.SAMPLE_LEN, 
                    self.NUM_INDIVIDUALS, 
                    self.GRID_SHAPE[0], 
                    self.GRID_SHAPE[1], 
                    self.KPTS_DIMENSIONS
                )
                sequences_gridded.append(gridded_chunk)
            except Exception as e:
                print(f"Error reshaping chunk {i}: {e}")
                continue

        seq_keypoints_list = []
        keypoints_ids_list = []
        sub_seq_length = self.max_keypoints_len  # This comes from num_frames
        sliding_w = self.sliding_window

        for seq_ix, vec_seq_chunk in enumerate(sequences_gridded):
            # Apply padding to handle subsequences at chunk edges
            if sub_seq_length <= self.SAMPLE_LEN:
                # Normal case: subsequences fit within chunk
                pad_length = min(sub_seq_length // 2, 60)  # Limit padding to 1 hour
            else:
                # Edge case: subsequence larger than chunk
                pad_length = 0
                print(f"Warning: sub_seq_length ({sub_seq_length}) > SAMPLE_LEN ({self.SAMPLE_LEN})")
            
            if pad_length > 0:
                pad_width = ((pad_length, pad_length), (0, 0), (0, 0), (0, 0), (0, 0))
                pad_vec = np.pad(vec_seq_chunk, pad_width, mode="edge")
            else:
                pad_vec = vec_seq_chunk
            
            seq_keypoints_list.append(pad_vec.astype(np.float32))

            # Generate sliding window indices
            max_start = len(vec_seq_chunk) - sub_seq_length + 1
            if max_start > 0:
                for i in range(0, max_start, sliding_w):
                    keypoints_ids_list.append((seq_ix, i + pad_length))  # Adjust for padding
            else:
                # If chunk is smaller than subsequence length, use the whole chunk
                keypoints_ids_list.append((seq_ix, pad_length))
        
        self.seq_keypoints = seq_keypoints_list
        self.keypoints_ids = keypoints_ids_list
        self.n_frames = len(self.keypoints_ids)
        self.items = list(range(self.n_frames))
        
        print(f"Preprocessing complete:")
        print(f"  Sequences: {len(self.seq_keypoints)}")
        print(f"  Subsequences: {self.n_frames}")
        print(f"  Subsequence length: {sub_seq_length}")
        print(f"  Sliding window: {sliding_w}")
        
        # Clean up raw data to save memory
        del self.raw_data

    def featurise_keypoints(self, keypoints):
        """Apply normalization and convert to tensor, preserving spatial structure"""
        keypoints_normalized = self.normalize(keypoints.astype(np.float32))
        keypoints_tensor = torch.tensor(keypoints_normalized, dtype=torch.float32)
        return keypoints_tensor

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data according to specified method"""
        method = self.normalization_method
        
        if method in {"none", "raw", None}: 
            return data.astype(np.float32)
        
        if method == "percentage": 
            return data.astype(np.float32) / 100.0
        
        if method in {"z_score", "global_z_score"}:
            if self.mean_val is None or self.std_val is None: 
                raise ValueError("Global μ/σ not initialized.")
            return (data - self.mean_val) / self.std_val
        
        if method == "local_z_score":
            mean, std = float(data.mean()), float(data.std())
            if std == 0: std = 1e-6
            return (data - mean) / std
        
        raise ValueError(f"Unknown normalization method: {method}")

    def unnormalize(self, data: np.ndarray) -> np.ndarray:
        """Reverse normalization"""
        method = self.normalization_method

        if method in {"none", "raw", None}:
            return data

        if method == "percentage":
            return data * 100.0

        if method in {"z_score", "global_z_score"}:
            if self.mean_val is None or self.std_val is None:
                raise ValueError("Global μ/σ not initialized.")
            return (data * self.std_val) + self.mean_val

        if method == "local_z_score":
            # Note: Cannot perfectly reverse without original stats
            return data

        raise ValueError(f"Unknown normalization method: {method}")

    def prepare_subsequence_sample(self, sequence: np.ndarray) -> torch.Tensor:
        """
        Prepare sequence for model input.
        Input: (T, 1, 4, 3, 1) -> Output: (T, 1, 12)
        """
        # Apply normalization and convert to tensor
        feats_tensor = self.featurise_keypoints(sequence)
        
        # Reshape from (T, 1, 4, 3, 1) to (T, 1, 12) for model input
        feats_reshaped = feats_tensor.reshape(self.max_keypoints_len, self.NUM_INDIVIDUALS, -1)
        
        return feats_reshaped

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list]:
        """Get a training sample"""
        if not self.keypoints_ids: 
            raise IndexError("keypoints_ids is empty.")
        
        seq_idx, frame_start_in_padded = self.keypoints_ids[idx]
        padded_sequence = self.seq_keypoints[seq_idx]
        
        # Extract subsequence
        subsequence = padded_sequence[
            frame_start_in_padded : frame_start_in_padded + self.max_keypoints_len
        ]
        
        # Validate subsequence shape
        expected_shape = (self.max_keypoints_len, self.NUM_INDIVIDUALS, 
                         self.GRID_SHAPE[0], self.GRID_SHAPE[1], self.KPTS_DIMENSIONS)
        if subsequence.shape != expected_shape:
            raise ValueError(f"Subsequence shape error at index {idx}: "
                           f"got {subsequence.shape}, expected {expected_shape}")

        inputs = self.prepare_subsequence_sample(subsequence)
        return inputs, []  # Empty labels for self-supervised learning

    def get_chunk_info(self):
        """Return information about loaded chunks for debugging"""
        return getattr(self, 'chunk_info', [])
    
    @staticmethod
    def fill_holes():
        """No hole filling needed for DVC data"""
        pass