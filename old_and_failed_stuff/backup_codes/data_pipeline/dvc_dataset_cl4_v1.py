from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch

from data_pipeline.load_dvc import load_dvc_data
from data_pipeline.pose_traj_dataset import BasePoseTrajDataset

# class SpatialAwareDVCDataset(BasePoseTrajDataset):
class DVCDataset(BasePoseTrajDataset):
    # FIXED: DVC always samples at 1 data point per minute
    DEFAULT_FRAME_RATE = 1.0 / 60.0  # 1/60 Hz = 0.0167 Hz (1 sample per minute)
    
    # CONFIRMED: 4 rows × 3 columns electrode grid layout
    ELECTRODE_GRID_SHAPE = (4, 3)  # 4 rows, 3 columns
    NUM_KEYPOINTS = 12  # Total electrodes (4×3=12)
    KPTS_DIMENSIONS = 1  # Each electrode has 1 capacitance value
    NUM_INDIVIDUALS = 1  # One cage unit (collective mouse activity)
    
    # Spatial grid dimensions
    SPATIAL_HEIGHT, SPATIAL_WIDTH = ELECTRODE_GRID_SHAPE  # 4, 3
    KEYFRAME_SHAPE = (NUM_INDIVIDUALS, SPATIAL_HEIGHT, SPATIAL_WIDTH)  # (1, 4, 3)
    
    STR_BODY_PARTS = [
        "V1", "V2", "V3", "V4", "V5", "V6",
        "V7", "V8", "V9", "V10", "V11", "V12",
    ]
    BODY_PART_2_INDEX = {w: i for i, w in enumerate(STR_BODY_PARTS)}
    
    # FIXED: 4×3 grid mapping for electrodes V1-V12
    # Adjust this if your actual electrode-to-position mapping differs!
    ELECTRODE_TO_GRID_MAPPING = np.array([
        [0, 1, 2],     # Row 1: V1, V2, V3
        [3, 4, 5],     # Row 2: V4, V5, V6
        [6, 7, 8],     # Row 3: V7, V8, V9
        [9, 10, 11],   # Row 4: V10, V11, V12
    ])

    def __init__(
        self,
        mode: str,
        path_to_data_dir: Path,
        scale: bool = True,
        sampling_rate: int = 1,
        num_frames: int = 80,
        sliding_window: int = 1,
        summary_csv: str = None,
        normalization_method: str = 'z_score',
        precomputed_stats_path: str = None,
        use_spatial_structure: bool = True,  # Enable spatial grid awareness
        **kwargs
    ):
        super().__init__(
            path_to_data_dir, scale, sampling_rate, num_frames, sliding_window, **kwargs
        )

        self.sample_frequency = self.DEFAULT_FRAME_RATE
        self.mode = mode
        self.normalization_method = normalization_method
        self.precomputed_stats_path = precomputed_stats_path
        self.use_spatial_structure = use_spatial_structure
        self.mean_val = None
        self.std_val = None
        
        # Extract aggregation info from summary CSV filename
        self.aggregation_window = self._extract_aggregation_from_filename(summary_csv)
        print(f"Detected aggregation window: {self.aggregation_window} minutes")

        self.load_data(summary_csv)

        if self.normalization_method == 'z_score':
            self._prepare_zscore_stats()

        self.preprocess()

    def _extract_aggregation_from_filename(self, summary_csv):
        """
        Extract aggregation window from summary CSV filename
        e.g., final_summary_metadata_1440.csv → 1440 minutes
        e.g., final_summary_metadata_10080.csv → 10080 minutes
        """
        if summary_csv is None:
            raise ValueError("summary_csv is required to determine aggregation window")
        
        filename = Path(summary_csv).name
        # Extract number from filename (assuming format: *_NUMBER.csv)
        import re
        numbers = re.findall(r'_(\d+)', filename)
        if not numbers:
            raise ValueError(f"Could not extract aggregation window from filename: {filename}")
        
        aggregation = int(numbers[-1])  # Take the last number found
        
        # Validate common aggregation windows
        valid_windows = {
            1440: "1 day",
            2880: "2 days", 
            10080: "1 week (7 days)",
            20160: "2 weeks (14 days)",
            40320: "4 weeks (28 days)",
            43200: "1 month (30 days)",
        }
        
        if aggregation in valid_windows:
            print(f"Recognized aggregation: {valid_windows[aggregation]} ({aggregation} minutes)")
        else:
            print(f"Custom aggregation: {aggregation} minutes ({aggregation/1440:.1f} days)")
        
        return aggregation

    def _prepare_zscore_stats(self):
        if self.mode == 'pretrain':
            if self.precomputed_stats_path and os.path.exists(self.precomputed_stats_path):
                print(f"Loading Z-score stats from {self.precomputed_stats_path}")
                stats = np.load(self.precomputed_stats_path, allow_pickle=True).item()
                self.mean_val = stats['mean']
                self.std_val = stats['std']
            elif hasattr(self, 'raw_data') and self.raw_data:
                print("Calculating Z-score stats from training data...")
                all_train_data = np.concatenate([seq.flatten() for seq in self.raw_data])
                self.mean_val = np.mean(all_train_data)
                self.std_val = np.std(all_train_data)
                if self.std_val == 0:
                    self.std_val = 1e-6
                print(f"Calculated Mean: {self.mean_val:.4f}, Std: {self.std_val:.4f}")
                if self.precomputed_stats_path:
                    dirname = os.path.dirname(self.precomputed_stats_path)
                    if dirname and not os.path.exists(dirname):
                        os.makedirs(dirname, exist_ok=True)
                    try:
                        np.save(self.precomputed_stats_path, {'mean': self.mean_val, 'std': self.std_val})
                        print(f"Successfully saved stats to {self.precomputed_stats_path}")
                    except Exception as e:
                        print(f"ERROR saving stats: {e}")
                        raise
            else:
                raise ValueError("Z-score stats cannot be computed: raw_data not loaded.")
        
        if self.mean_val is None or self.std_val is None:
             raise ValueError("Mean and Std for Z-score normalization are not set.")

    def electrode_data_to_spatial_grid(self, electrode_data):
        """
        Convert linear electrode data (V1-V12) to 4×3 spatial grid
        Args:
            electrode_data: (timesteps, 12) - raw electrode readings V1-V12
        Returns:
            grid_data: (timesteps, 1, 4, 3) - spatially arranged data
        """
        timesteps = electrode_data.shape[0]
        
        if self.use_spatial_structure:
            # Create 4×3 spatial grid
            grid_data = np.zeros((timesteps, 1, 4, 3), dtype=np.float32)
            
            # Map each electrode to its grid position
            for row in range(4):
                for col in range(3):
                    electrode_idx = self.ELECTRODE_TO_GRID_MAPPING[row, col]
                    grid_data[:, 0, row, col] = electrode_data[:, electrode_idx]
            
            print(f"Converted to spatial grid: {electrode_data.shape} → {grid_data.shape}")
            return grid_data
        else:
            # Fallback to linear arrangement: (timesteps, 1, 12, 1)
            linear_data = electrode_data.reshape(timesteps, 1, 12, 1)
            print(f"Using linear arrangement: {electrode_data.shape} → {linear_data.shape}")
            return linear_data

    def load_data(self, summary_csv):
        csv_path = summary_csv
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        print(f"Reading CSV from: {csv_path}")
        
        df = pd.read_csv(csv_path, low_memory=False)
        df = df.head(1000)  # For testing, load first 1000 rows
        print(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns.")

        # Build training list based on aggregation periods
        training_list = {}
        for _, row in df.iterrows():
            cage = row['cage_id']
            # The aggregation period is already handled by your data pipeline
            # We just need to group by the time periods in the CSV
            time_key = row['from_tpt']  # Use the full timestamp as key
            
            if cage not in training_list:
                training_list[cage] = []
            training_list[cage].append(time_key)
        del df

        if self.mode == "pretrain":
            loaded_data_dict = load_dvc_data(self.path, training_list)
            self.raw_data = []
            
            for cage_id, df in loaded_data_dict.items():
                # Get electrode data: (timesteps, 12_electrodes)
                electrode_columns = [col for col in df.columns if col.startswith('v_')]
                if len(electrode_columns) != 12:
                    raise ValueError(f"Expected 12 electrode columns (V1-V12), got {len(electrode_columns)}")
                
                electrode_data = df[electrode_columns].values.astype(np.float32)
                
                # Validate sequence length matches aggregation window
                expected_len = self.aggregation_window
                if electrode_data.shape[0] != expected_len:
                    print(f"Warning: Expected {expected_len} timesteps, got {electrode_data.shape[0]} for cage {cage_id}")
                    if electrode_data.shape[0] < expected_len:
                        # Pad if too short
                        pad_length = expected_len - electrode_data.shape[0]
                        electrode_data = np.pad(electrode_data, ((0, pad_length), (0, 0)), mode='edge')
                        print(f"Padded sequence to {electrode_data.shape[0]} timesteps")
                    else:
                        # Truncate if too long
                        electrode_data = electrode_data[:expected_len]
                        print(f"Truncated sequence to {electrode_data.shape[0]} timesteps")
                
                # Convert to spatial grid format
                spatial_data = self.electrode_data_to_spatial_grid(electrode_data)
                self.raw_data.append(spatial_data)
            
            print(f"Loaded {len(self.raw_data)} sequences with {self.aggregation_window} timesteps each")
            if self.raw_data:
                print(f"Data shape per sequence: {self.raw_data[0].shape}")
                print(f"Using spatial structure: {self.use_spatial_structure}")
            else:
                print("Warning: No training data loaded.")
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def featurise_keypoints(self, keypoints):
        """Apply normalization and convert to tensor"""
        keypoints_normalized = self.normalize(keypoints.astype(np.float32))
        keypoints_tensor = torch.tensor(keypoints_normalized, dtype=torch.float32)
        return keypoints_tensor

    def preprocess(self):
        """Preprocess sequences for training with sliding windows"""
        if not hasattr(self, 'raw_data') or not self.raw_data:
            print(f"No raw data to preprocess for mode: {self.mode}.")
            self.seq_keypoints = []
            self.keypoints_ids = []
            self.n_frames = 0
            self.items = []
            return

        seq_keypoints_list = []
        keypoints_ids_list = []
        sub_seq_length = self.max_keypoints_len
        sliding_w = self.sliding_window

        print(f"Preprocessing with sub-sequence length: {sub_seq_length}, sliding window: {sliding_w}")

        for seq_ix, data_sequence in enumerate(self.raw_data):
            # data_sequence shape: (timesteps, 1, 4, 3) or (timesteps, 1, 12, 1)
            
            # Flatten spatial dimensions for padding
            original_shape_spatial = data_sequence.shape[1:]  # (1, 4, 3) or (1, 12, 1)
            seq_flat_spatial = data_sequence.reshape(data_sequence.shape[0], -1)

            # Adaptive padding based on sequence length
            if sub_seq_length < 120:
                pad_length = sub_seq_length
            else:
                pad_length = min(120, sub_seq_length // 2)  # Don't over-pad

            pad_width_time = (pad_length // 2, pad_length - 1 - (pad_length // 2))
            
            # Pad only along time axis
            pad_seq_flat = np.pad(
                seq_flat_spatial,
                (pad_width_time, (0, 0)),
                mode="edge",
            )
            
            # Reshape back to spatial format
            pad_seq_reshaped = pad_seq_flat.reshape(pad_seq_flat.shape[0], *original_shape_spatial)
            seq_keypoints_list.append(pad_seq_reshaped.astype(np.float32))

            # Generate sliding window indices
            num_possible_starts = len(pad_seq_reshaped) - sub_seq_length + 1
            if num_possible_starts <= 0:
                print(f"Warning: Sequence {seq_ix} too short for sub_seq_length {sub_seq_length}")
                continue
                
            for i in range(0, num_possible_starts, sliding_w):
                keypoints_ids_list.append((seq_ix, i))

        self.seq_keypoints = seq_keypoints_list
        self.keypoints_ids = keypoints_ids_list
        self.n_frames = len(keypoints_ids_list)
        self.items = list(range(self.n_frames))
        
        print(f"Preprocessing complete:")
        print(f"  - Sequences: {len(self.seq_keypoints)}")
        print(f"  - Sub-sequences: {self.n_frames}")
        print(f"  - Aggregation window: {self.aggregation_window} minutes")
        print(f"  - Sub-sequence length: {sub_seq_length} minutes")
        
        # Clean up raw data
        if hasattr(self, 'raw_data'):
            del self.raw_data

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize electrode capacitance values"""
        method = self.normalization_method

        if method in {"none", "raw", None}:
            return data.astype(np.float32)
        elif method == "percentage":
            return data.astype(np.float32) / 100.0
        elif method in {"z_score", "global_z_score"}:
            if (self.mean_val is None or self.std_val is None) \
                and self.precomputed_stats_path \
                and os.path.exists(self.precomputed_stats_path):
                stats = np.load(self.precomputed_stats_path, allow_pickle=True).item()
                self.mean_val = stats["mean"]
                self.std_val = stats["std"]
            if self.mean_val is None or self.std_val is None:
                raise ValueError("Global μ/σ not initialised.")
            return (data - self.mean_val) / self.std_val
        elif method == "local_z_score":
            mean = float(data.mean())
            std = float(data.std())
            if std == 0:
                std = 1e-6
            return (data - mean) / std
        else:
            raise ValueError(f"Unknown normalisation method: {method}")

    def prepare_subsequence_sample(self, sequence: np.ndarray) -> torch.Tensor:
        """Prepare a subsequence sample for training"""
        feats_tensor = self.featurise_keypoints(sequence)
        
        if self.use_spatial_structure:
            # Spatial structure: (num_frames, 1, 4*3) = (num_frames, 1, 12)
            feats_reshaped = feats_tensor.reshape(self.max_keypoints_len, 1, -1)
        else:
            # Linear structure: (num_frames, 1, 12)
            feats_reshaped = feats_tensor.reshape(self.max_keypoints_len, 1, -1)
        
        return feats_reshaped

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list]:
        """Get a training sample"""
        if not self.keypoints_ids:
            raise IndexError("No keypoint IDs available.")

        seq_idx, frame_start = self.keypoints_ids[idx]
        padded_sequence = self.seq_keypoints[seq_idx]
        subsequence = padded_sequence[frame_start:frame_start + self.max_keypoints_len]
        
        if subsequence.shape[0] != self.max_keypoints_len:
            raise ValueError(
                f"Subsequence length mismatch: got {subsequence.shape[0]}, expected {self.max_keypoints_len}"
            )

        inputs = self.prepare_subsequence_sample(subsequence)
        return inputs, []  # Empty labels for self-supervised learning

    @staticmethod
    def fill_holes():
        """DVC data is pre-processed, no hole filling needed"""
        pass