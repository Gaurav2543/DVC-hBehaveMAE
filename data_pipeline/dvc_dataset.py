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
    Corrected DVC dataset for hBehaveMAE with proper spatial grid mapping.
    """
    DEFAULT_FRAME_RATE = 1/60  # 1 sample per minute
    NUM_KEYPOINTS = 12
    KPTS_DIMENSIONS = 1
    NUM_INDIVIDUALS = 1
    
    # CRITICAL: Define how electrodes V1-V12 map to 4x3 grid positions
    # ADJUST THIS BASED ON YOUR ACTUAL ELECTRODE LAYOUT!
    ELECTRODE_TO_GRID_MAPPING = np.array([
        [0, 1, 2],     # Row 1: V1, V2, V3 
        [3, 4, 5],     # Row 2: V4, V5, V6
        [6, 7, 8],     # Row 3: V7, V8, V9
        [9, 10, 11],   # Row 4: V10, V11, V12
    ])
    
    KEYFRAME_SHAPE = (NUM_INDIVIDUALS, 4, 3, KPTS_DIMENSIONS)  # 4x3 grid

    def __init__(
        self,
        mode: str,
        path_to_data_dir: Path,
        summary_csv: str,
        scale: bool = True,
        sampling_rate: int = 1,
        num_frames: int = 400,
        sliding_window: int = 1,
        normalization_method: str = 'z_score',  # Changed default to z_score
        precomputed_stats_path: str = None,
        strict_summary_mode: bool = True,  # NEW: Ensure only summary entries are used
        **kwargs
    ):
        # Extract SAMPLE_LEN from filename
        match = re.search(r'_(\d+)\.csv', summary_csv)
        if match:
            self.SAMPLE_LEN = int(match.group(1))
            print(f"Data chunks are expected to be {self.SAMPLE_LEN} minutes long.")
        else:
            raise ValueError(f"Could not determine sample length from summary file: {summary_csv}")
        
        super().__init__(
            path_to_data_dir, scale, sampling_rate, num_frames, sliding_window, **kwargs
        )

        self.sample_frequency = self.DEFAULT_FRAME_RATE
        self.mode = mode
        self.normalization_method = normalization_method
        self.precomputed_stats_path = precomputed_stats_path
        self.strict_summary_mode = strict_summary_mode
        self.mean_val, self.std_val = None, None
        
        # Statistics for validation
        self.summary_entries_total = 0
        self.summary_entries_loaded = 0
        self.summary_entries_skipped = 0

        self.load_data(summary_csv)

        if self.normalization_method == 'z_score':
            self._prepare_zscore_stats()

        self.preprocess()
        
        # Print loading statistics
        self._print_loading_stats()

    def _prepare_zscore_stats(self):
        if self.mode == 'pretrain':
            if self.precomputed_stats_path and os.path.exists(self.precomputed_stats_path):
                print(f"Loading Z-score stats from {self.precomputed_stats_path}")
                stats = np.load(self.precomputed_stats_path, allow_pickle=True).item()
                self.mean_val, self.std_val = stats['mean'], stats['std']
                print(f"Loaded Z-score stats: Mean={self.mean_val:.4f}, Std={self.std_val:.4f}")
            elif hasattr(self, 'raw_data') and self.raw_data:
                print("Calculating Z-score stats from training data...")
                all_train_data = np.concatenate([seq.flatten() for seq in self.raw_data])
                self.mean_val = np.mean(all_train_data)
                self.std_val = np.std(all_train_data)
                if self.std_val == 0: 
                    self.std_val = 1e-6
                print(f"Calculated Z-score stats: Mean={self.mean_val:.4f}, Std={self.std_val:.4f}")
                
                if self.precomputed_stats_path:
                    dirname = os.path.dirname(self.precomputed_stats_path)
                    if dirname and not os.path.exists(dirname):
                        os.makedirs(dirname, exist_ok=True)
                    np.save(self.precomputed_stats_path, {'mean': self.mean_val, 'std': self.std_val})
                    print(f"Saved Z-score stats to {self.precomputed_stats_path}")
            else:
                raise ValueError("Cannot calculate Z-score stats: no raw data available")
        
        if self.mean_val is None or self.std_val is None:
            raise ValueError("Mean and Std for Z-score normalization are not set.")

    def electrode_linear_to_grid(self, linear_data):
        """
        Convert linear electrode data (V1-V12) to proper 4x3 spatial grid.
        
        Args:
            linear_data: (timesteps, 12) - electrodes in V1, V2, ..., V12 order
        Returns:
            grid_data: (timesteps, 1, 4, 3, 1) - spatially arranged data
        """
        timesteps = linear_data.shape[0]
        
        # Create 4x3 grid
        grid_data = np.zeros((timesteps, 1, 4, 3, 1), dtype=np.float32)
        
        # Map each electrode to its correct spatial position
        for row in range(4):
            for col in range(3):
                electrode_idx = self.ELECTRODE_TO_GRID_MAPPING[row, col]
                grid_data[:, 0, row, col, 0] = linear_data[:, electrode_idx]
        
        return grid_data

    def load_data(self, summary_csv):
        """
        Load data ensuring ONLY entries from summary CSV are used.
        """
        print(f"Reading summary file from: {summary_csv}")
        summary_df = pd.read_csv(summary_csv, low_memory=False)
        
        # For testing, limit to first 1000 rows
        original_len = len(summary_df)
        # summary_df = summary_df.head(100)
        if len(summary_df) < original_len:
            print(f"Limited to first {len(summary_df)} entries for testing (original: {original_len})")
        
        self.summary_entries_total = len(summary_df)
        
        # Parse timestamps
        summary_df['from_tpt'] = pd.to_datetime(summary_df['from_tpt'])
        summary_df['to_tpt'] = pd.to_datetime(summary_df['to_tpt'])

        self.raw_data = []
        processed_entries = []  # Track what we actually loaded
        
        print("Loading data chunks based on summary CSV entries...")
        
        # Process each entry in summary CSV individually
        for idx, row in tqdm(summary_df.iterrows(), total=len(summary_df), desc="Processing summary entries"):
            cage_id = row['cage_id']
            start_time = row['from_tpt']
            end_time = row['to_tpt']
            
            # Expected duration check
            expected_duration_minutes = (end_time - start_time).total_seconds() / 60
            if abs(expected_duration_minutes - self.SAMPLE_LEN) > 1:  # Allow 1 minute tolerance
                print(f"Warning: Entry {idx} duration ({expected_duration_minutes:.1f} min) doesn't match expected {self.SAMPLE_LEN} min")
                self.summary_entries_skipped += 1
                continue
            
            try:
                # Load raw cage data file
                cage_file = os.path.join(self.path, f"{cage_id}.csv")
                if not os.path.exists(cage_file):
                    print(f"Warning: Cage file {cage_file} not found. Skipping entry {idx}.")
                    self.summary_entries_skipped += 1
                    continue
                
                # Load and filter to exact time range
                raw_cage_df = pd.read_csv(cage_file, parse_dates=['timestamp'])
                raw_cage_df = raw_cage_df.set_index('timestamp')
                
                # Extract exact time chunk specified in summary
                chunk_df = raw_cage_df[start_time:end_time]
                
                # Strict validation of chunk length
                if len(chunk_df) != self.SAMPLE_LEN:
                    print(f"Warning: Chunk {idx} has {len(chunk_df)} timesteps, expected {self.SAMPLE_LEN}. Skipping.")
                    self.summary_entries_skipped += 1
                    continue
                
                # Extract electrode data
                electrode_cols = [f"v_{i}" for i in range(1, 13)]  # v_1, v_2, ..., v_12
                
                # Validate all electrode columns exist
                missing_cols = [col for col in electrode_cols if col not in chunk_df.columns]
                if missing_cols:
                    print(f"Warning: Missing electrode columns {missing_cols} in cage {cage_id}. Skipping.")
                    self.summary_entries_skipped += 1
                    continue
                
                chunk_data = chunk_df[electrode_cols].values.astype(np.float32)
                
                # Validate data quality
                if np.any(np.isnan(chunk_data)):
                    print(f"Warning: NaN values found in chunk {idx}. Skipping.")
                    self.summary_entries_skipped += 1
                    continue
                
                # Convert to spatial grid format
                gridded_data = self.electrode_linear_to_grid(chunk_data)
                self.raw_data.append(gridded_data)
                
                # Track successful loading
                processed_entries.append({
                    'summary_idx': idx,
                    'cage_id': cage_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'data_shape': gridded_data.shape
                })
                self.summary_entries_loaded += 1
                
            except Exception as e:
                print(f"Error processing entry {idx} for cage {cage_id}: {e}")
                self.summary_entries_skipped += 1
                continue

        print(f"Data loading complete:")
        print(f"  - Summary entries total: {self.summary_entries_total}")
        print(f"  - Successfully loaded: {self.summary_entries_loaded}")
        print(f"  - Skipped: {self.summary_entries_skipped}")
        
        if not self.raw_data:
            raise ValueError("No valid data chunks were loaded!")
        
        # Additional validation: ensure all chunks have consistent shape
        expected_shape = (self.SAMPLE_LEN, 1, 4, 3, 1)
        for i, chunk in enumerate(self.raw_data):
            if chunk.shape != expected_shape:
                raise ValueError(f"Chunk {i} has shape {chunk.shape}, expected {expected_shape}")

    def _print_loading_stats(self):
        """Print detailed loading statistics."""
        print("\n" + "="*50)
        print("DVC DATASET LOADING SUMMARY")
        print("="*50)
        print(f"Summary file entries: {self.summary_entries_total}")
        print(f"Successfully loaded: {self.summary_entries_loaded}")
        print(f"Skipped entries: {self.summary_entries_skipped}")
        print(f"Loading success rate: {self.summary_entries_loaded/self.summary_entries_total*100:.1f}%")
        # print(f"Final training chunks: {len(self.raw_data) if hasattr(self, 'raw_data') else 0}")
        print(f"Chunk length: {self.SAMPLE_LEN} minutes")
        print(f"Training window size: {self.max_keypoints_len} minutes")
        print(f"Spatial grid: 4×3 electrodes")
        print(f"Total training samples: {self.n_frames if hasattr(self, 'n_frames') else 0}")
        print("="*50 + "\n")

    def preprocess(self):
        """
        Preprocess the loaded data chunks for training.
        """
        if not hasattr(self, 'raw_data') or not self.raw_data:
            print("No raw data to preprocess.")
            self.seq_keypoints, self.keypoints_ids, self.items = [], [], []
            self.n_frames = 0
            return

        seq_keypoints_list = []
        keypoints_ids_list = []
        sub_seq_length = self.max_keypoints_len
        sliding_w = self.sliding_window

        print(f"Preprocessing {len(self.raw_data)} chunks with sub-sequence length {sub_seq_length}")

        for seq_ix, gridded_chunk in enumerate(self.raw_data):
            # gridded_chunk shape: (SAMPLE_LEN, 1, 4, 3, 1)
            
            # Apply padding for edge cases
            if sub_seq_length < 120:
                pad_length = sub_seq_length // 2
            else:
                pad_length = 60  # Don't over-pad for long sequences
            
            pad_width = ((pad_length, pad_length), (0, 0), (0, 0), (0, 0), (0, 0))
            pad_vec = np.pad(gridded_chunk, pad_width, mode="edge")
            seq_keypoints_list.append(pad_vec.astype(np.float32))

            # Generate sliding window indices
            num_possible_starts = len(gridded_chunk)  # Use original length
            if num_possible_starts < sub_seq_length:
                print(f"Warning: Chunk {seq_ix} length {num_possible_starts} < required {sub_seq_length}")
                continue
                
            for i in range(0, num_possible_starts - sub_seq_length + 1, sliding_w):
                keypoints_ids_list.append((seq_ix, i))

        self.seq_keypoints = seq_keypoints_list
        self.keypoints_ids = keypoints_ids_list
        self.n_frames = len(self.keypoints_ids)
        self.items = list(range(self.n_frames))
        
        print(f"Preprocessing complete: {len(self.seq_keypoints)} sequences, {self.n_frames} training samples")
        
        # Clean up memory
        del self.raw_data

    def featurise_keypoints(self, keypoints):
        """Apply normalization and convert to tensor."""
        keypoints_normalized = self.normalize(keypoints.astype(np.float32))
        keypoints_tensor = torch.tensor(keypoints_normalized, dtype=torch.float32)
        return keypoints_tensor

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize electrode data."""
        method = self.normalization_method
        
        if method in {"none", "raw", None}: 
            return data.astype(np.float32)
        elif method == "percentage": 
            return data.astype(np.float32) / 100.0
        elif method in {"z_score", "global_z_score"}:
            if self.mean_val is None or self.std_val is None: 
                raise ValueError("Global μ/σ not initialised.")
            return (data - self.mean_val) / self.std_val
        elif method == "local_z_score":
            mean, std = float(data.mean()), float(data.std())
            if std == 0: std = 1e-6
            return (data - mean) / std
        else:
            raise ValueError(f"Unknown normalisation method: {method}")

    def unnormalize(self, data: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        method = self.normalization_method

        if method in {"none", "raw", None}:
            return data
        elif method == "percentage":
            return data * 100.0
        elif method in {"z_score", "global_z_score"}:
            if self.mean_val is None or self.std_val is None:
                raise ValueError("Global μ/σ not initialised.")
            return (data * self.std_val) + self.mean_val
        elif method == "local_z_score":
            mean = float(data.mean())
            std = float(data.std())
            if std == 0: std = 1e-6
            return (data * std) + mean
        else:
            raise ValueError(f"Unknown normalisation method: {method}")

    def prepare_subsequence_sample(self, sequence: np.ndarray) -> torch.Tensor:
        """Prepare training sample from sequence."""
        # sequence shape: (max_keypoints_len, 1, 4, 3, 1)
        feats_tensor = self.featurise_keypoints(sequence)
        # Reshape to (max_keypoints_len, 1, 4*3*1) = (max_keypoints_len, 1, 12)
        feats_reshaped = feats_tensor.reshape(self.max_keypoints_len, self.NUM_INDIVIDUALS, -1)
        return feats_reshaped

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list]:
        """Get training sample."""
        if not self.keypoints_ids: 
            raise IndexError("keypoints_ids is empty.")
        
        seq_idx, frame_start_in_original = self.keypoints_ids[idx]
        padded_sequence = self.seq_keypoints[seq_idx]
        
        # Adjust for padding
        pad_length = (padded_sequence.shape[0] - self.SAMPLE_LEN) // 2
        frame_start_in_padded = frame_start_in_original + pad_length
        
        subsequence = padded_sequence[frame_start_in_padded:frame_start_in_padded + self.max_keypoints_len]
        
        if subsequence.shape[0] != self.max_keypoints_len:
            raise ValueError(f"Subsequence shape error at index {idx}: got {subsequence.shape[0]}, expected {self.max_keypoints_len}")

        inputs = self.prepare_subsequence_sample(subsequence)
        return inputs, []

    @staticmethod
    def fill_holes():
        """No hole filling needed for DVC data."""
        pass