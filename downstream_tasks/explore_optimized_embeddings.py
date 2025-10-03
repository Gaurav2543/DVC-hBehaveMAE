"""
Simple script to explore and understand the structure of extracted embeddings
"""
import numpy as np
import pandas as pd
import zarr
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

class EmbeddingExplorer:
    def __init__(self, embeddings_dir):
        self.embeddings_dir = Path(embeddings_dir)
        self.core_dir = self.embeddings_dir / 'core_embeddings'
        self.spatial_dir = self.embeddings_dir / 'spatial_embeddings'
        
    def list_available_sequences(self):
        """List all available sequences"""
        print("AVAILABLE SEQUENCES")
        print("=" * 40)
        
        core_files = list(self.core_dir.glob("*.zarr")) if self.core_dir.exists() else []
        spatial_files = list(self.spatial_dir.glob("*.zarr")) if self.spatial_dir.exists() else []
        
        print(f"Core embeddings: {len(core_files)} sequences")
        print(f"Spatial embeddings: {len(spatial_files)} sequences")
        
        if core_files:
            print("\nSequences available:")
            for f in sorted(core_files)[:5]:
                seq_name = f.stem
                print(f"  - {seq_name}")
            if len(core_files) > 5:
                print(f"  ... and {len(core_files) - 5} more")
        
        return [f.stem for f in core_files]
    
    def inspect_sequence(self, sequence_name, detailed=True):
        """Inspect a specific sequence in detail"""
        print(f"\nINSPECTING SEQUENCE: {sequence_name}")
        print("=" * 50)
        
        # Load core embeddings
        core_path = self.core_dir / f"{sequence_name}.zarr"
        if not core_path.exists():
            print(f"Core embeddings not found: {core_path}")
            return
        
        core_store = zarr.open(str(core_path), mode='r')
        
        print("CORE EMBEDDINGS STRUCTURE:")
        print("-" * 30)
        
        embedding_keys = []
        for key in core_store.keys():
            array = core_store[key]
            if key in ['timestamps', 'frame_indices']:
                print(f"  {key}: {array.shape} ({array.dtype})")
                if key == 'timestamps' and detailed:
                    print(f"    Sample timestamps: {array[:3]} ... {array[-2:]}")
            else:
                embedding_keys.append(key)
                print(f"  {key}: {array.shape} ({array.dtype})")
                if detailed:
                    sample_data = array[:3] if len(array.shape) == 2 else array[:2].flatten()[:10]
                    print(f"    Sample values: {sample_data}")
                    print(f"    Mean: {np.mean(array[:]):.6f}, Std: {np.std(array[:]):.6f}")
        
        # Show metadata
        print(f"\nMETADATA:")
        for attr_name in core_store.attrs:
            print(f"  {attr_name}: {core_store.attrs[attr_name]}")
        
        # Analyze temporal structure
        if 'timestamps' in core_store:
            self._analyze_temporal_structure(core_store['timestamps'][:], sequence_name)
        
        # Analyze embedding dimensions
        if detailed:
            self._analyze_embedding_dimensions(core_store, embedding_keys)
        
        # Check spatial embeddings if available
        spatial_path = self.spatial_dir / f"{sequence_name}.zarr"
        if spatial_path.exists():
            print(f"\nSPATIAL EMBEDDINGS AVAILABLE:")
            print("-" * 30)
            spatial_store = zarr.open(str(spatial_path), mode='r')
            for key in spatial_store.keys():
                if key not in ['timestamps', 'frame_indices']:
                    array = spatial_store[key]
                    print(f"  {key}: {array.shape} ({array.dtype})")
    
    def _analyze_temporal_structure(self, timestamps, sequence_name):
        """Analyze the temporal structure of embeddings"""
        print(f"\nTEMPORAL ANALYSIS:")
        print("-" * 20)
        
        # Parse timestamps
        parsed_times = [datetime.strptime(ts, '%Y-%m-%d_%H:%M') for ts in timestamps]
        
        start_time = parsed_times[0]
        end_time = parsed_times[-1]
        duration = end_time - start_time
        
        print(f"  Time range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Duration: {duration.total_seconds() / 3600:.1f} hours")
        print(f"  Total frames: {len(timestamps)}")
        print(f"  Resolution: ~{duration.total_seconds() / len(timestamps):.1f} seconds per frame")
        
        # Check for gaps or irregularities
        time_diffs = [(parsed_times[i+1] - parsed_times[i]).total_seconds() 
                      for i in range(len(parsed_times)-1)]
        avg_diff = np.mean(time_diffs)
        std_diff = np.std(time_diffs)
        print(f"  Average time between frames: {avg_diff:.1f} ± {std_diff:.1f} seconds")
        
        if std_diff > 1:
            print("  ⚠️  Warning: Irregular time intervals detected")
        
        # Show time distribution by hour
        hours = [t.hour for t in parsed_times]
        unique_hours = sorted(set(hours))
        print(f"  Hours covered: {min(unique_hours):02d}:00 to {max(unique_hours):02d}:00")
    
    def _analyze_embedding_dimensions(self, store, embedding_keys):
        """Analyze the dimensionality of embeddings"""
        print(f"\nEMBEDDING DIMENSIONALITY ANALYSIS:")
        print("-" * 35)
        
        total_dims = 0
        for key in sorted(embedding_keys):
            array = store[key]
            if len(array.shape) == 2:  # (time, features)
                dims = array.shape[1]
                total_dims += dims
                print(f"  {key}: {dims} dimensions")
                
                # Basic statistics
                mean_activation = np.mean(np.abs(array[:]))
                sparsity = np.mean(array[:] == 0) * 100
                print(f"    Mean |activation|: {mean_activation:.6f}")
                print(f"    Sparsity: {sparsity:.1f}% zeros")
        
        print(f"\nTotal embedding dimensions: {total_dims}")
        
        # Check for 'combined' embedding
        if 'combined' in store:
            combined_dims = store['combined'].shape[1]
            print(f"Combined embedding: {combined_dims} dimensions")
            if combined_dims == total_dims - store['combined'].shape[1]:  # Avoid double counting
                print("✓ Combined embedding matches sum of individual levels")
    
    def demonstrate_temporal_access(self, sequence_name):
        """Demonstrate how to access embeddings by timestamp"""
        print(f"\nTEMPORAL ACCESS DEMONSTRATION")
        print("=" * 35)
        
        core_path = self.core_dir / f"{sequence_name}.zarr"
        if not core_path.exists():
            print(f"Sequence not found: {sequence_name}")
            return
        
        store = zarr.open(str(core_path), mode='r')
        
        timestamps = store['timestamps'][:]
        level_1_pooled = store['level_1_pooled'][:]
        
        print(f"Loaded {len(timestamps)} timestamps and embeddings")
        
        # Example 1: Find embedding for a specific time
        target_times = ['06:30', '12:00', '18:00', '23:30']
        
        for target_time in target_times:
            # Find indices matching this time pattern
            matching_indices = [i for i, ts in enumerate(timestamps) 
                              if ts.endswith(f'_{target_time}')]
            
            if matching_indices:
                idx = matching_indices[0]  # Take first match
                embedding = level_1_pooled[idx]
                print(f"\nTime {target_time}: Index {idx}")
                print(f"  Timestamp: {timestamps[idx]}")
                print(f"  Embedding shape: {embedding.shape}")
                print(f"  First 5 values: {embedding[:5]}")
            else:
                print(f"\nTime {target_time}: Not found")
        
        # Example 2: Get embeddings for a time range
        print(f"\nTIME RANGE EXAMPLE:")
        morning_indices = [i for i, ts in enumerate(timestamps) 
                          if any(ts.endswith(f'_{h:02d}:{m:02d}') 
                                for h in range(6, 9) for m in range(0, 60, 30))]
        
        if morning_indices:
            morning_embeddings = level_1_pooled[morning_indices]
            print(f"Morning (6-8 AM) embeddings: {morning_embeddings.shape}")
            print(f"Time range: {timestamps[morning_indices[0]]} to {timestamps[morning_indices[-1]]}")
    
    def compare_sequences(self, sequence_names):
        """Compare embeddings across multiple sequences"""
        print(f"\nCOMPARING SEQUENCES")
        print("=" * 25)
        
        sequences_data = {}
        
        for seq_name in sequence_names:
            core_path = self.core_dir / f"{seq_name}.zarr"
            if core_path.exists():
                store = zarr.open(str(core_path), mode='r')
                sequences_data[seq_name] = {
                    'level_1_pooled': store['level_1_pooled'][:],
                    'timestamps': store['timestamps'][:],
                    'metadata': dict(store.attrs)
                }
        
        if len(sequences_data) < 2:
            print("Need at least 2 sequences to compare")
            return
        
        print(f"Comparing {len(sequences_data)} sequences:")
        
        for seq_name, data in sequences_data.items():
            embeddings = data['level_1_pooled']
            print(f"\n{seq_name}:")
            print(f"  Shape: {embeddings.shape}")
            print(f"  Mean: {np.mean(embeddings):.6f}")
            print(f"  Std: {np.std(embeddings):.6f}")
            print(f"  Date: {data['metadata'].get('date', 'Unknown')}")
        
        # Cross-sequence correlation (sample-based to avoid memory issues)
        print(f"\nCROSS-SEQUENCE SIMILARITY:")
        seq_names = list(sequences_data.keys())
        sample_size = min(1000, min(len(data['level_1_pooled']) for data in sequences_data.values()))
        
        for i, seq1 in enumerate(seq_names):
            for seq2 in seq_names[i+1:]:
                # Sample embeddings from both sequences
                emb1 = sequences_data[seq1]['level_1_pooled'][:sample_size].flatten()
                emb2 = sequences_data[seq2]['level_1_pooled'][:sample_size].flatten()
                
                correlation = np.corrcoef(emb1, emb2)[0, 1]
                print(f"  {seq1.split('_')[-1]} vs {seq2.split('_')[-1]}: r={correlation:.3f}")
    
    def create_simple_visualization(self, sequence_name, save_dir=None):
        """Create simple visualizations of embeddings"""
        print(f"\nCREATING VISUALIZATIONS FOR: {sequence_name}")
        print("=" * 40)
        
        core_path = self.core_dir / f"{sequence_name}.zarr"
        if not core_path.exists():
            print(f"Sequence not found: {sequence_name}")
            return
        
        store = zarr.open(str(core_path), mode='r')
        timestamps = store['timestamps'][:]
        level_1_pooled = store['level_1_pooled'][:]
        
        # Parse timestamps for plotting
        parsed_times = [datetime.strptime(ts, '%Y-%m-%d_%H:%M') for ts in timestamps]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Embedding Analysis: {sequence_name}', fontsize=14)
        
        # 1. Temporal evolution of first few dimensions
        axes[0, 0].plot(parsed_times[::60], level_1_pooled[::60, :5])  # Sample every hour
        axes[0, 0].set_title('First 5 Dimensions Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Activation')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Distribution of activations
        axes[0, 1].hist(level_1_pooled.flatten(), bins=50, alpha=0.7)
        axes[0, 1].set_title('Distribution of All Activations')
        axes[0, 1].set_xlabel('Activation Value')
        axes[0, 1].set_ylabel('Count')
        
        # 3. Heatmap of embeddings over time (sample)
        sample_indices = np.linspace(0, len(level_1_pooled)-1, 100, dtype=int)
        sample_embeddings = level_1_pooled[sample_indices, :20]  # First 20 dims
        
        im = axes[1, 0].imshow(sample_embeddings.T, aspect='auto', cmap='RdBu_r')
        axes[1, 0].set_title('Embedding Heatmap (First 20 Dims)')
        axes[1, 0].set_xlabel('Time (sampled)')
        axes[1, 0].set_ylabel('Dimension')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Daily pattern (if data spans multiple days)
        hours = [t.hour for t in parsed_times]
        hourly_means = []
        for h in range(24):
            hour_indices = [i for i, hour in enumerate(hours) if hour == h]
            if hour_indices:
                hourly_means.append(np.mean(level_1_pooled[hour_indices]))
            else:
                hourly_means.append(np.nan)
        
        axes[1, 1].plot(range(24), hourly_means, 'o-')
        axes[1, 1].set_title('Daily Pattern (Mean Activation by Hour)')
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Mean Activation')
        axes[1, 1].set_xticks(range(0, 24, 4))
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / f'{sequence_name}_analysis.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")
        else:
            plt.show()
        
        plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python embedding_explorer.py <embeddings_directory>")
        print("Example: python embedding_explorer.py /path/to/output/3days")
        sys.exit(1)
    
    embeddings_dir = sys.argv[1]
    explorer = EmbeddingExplorer(embeddings_dir)
    
    # 1. List available sequences
    sequences = explorer.list_available_sequences()
    
    if not sequences:
        print("No sequences found!")
        return
    
    # 2. Inspect first sequence in detail
    print("\n" + "="*60)
    print("DETAILED INSPECTION OF FIRST SEQUENCE")
    print("="*60)
    explorer.inspect_sequence(sequences[0], detailed=True)
    
    # 3. Demonstrate temporal access
    explorer.demonstrate_temporal_access(sequences[0])
    
    # 4. Compare sequences if multiple available
    if len(sequences) > 1:
        compare_sequences = sequences[:3]  # Compare first 3
        explorer.compare_sequences(compare_sequences)
    
    # 5. Create visualization
    try:
        explorer.create_simple_visualization(sequences[0])
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    print("\n" + "="*60)
    print("EXPLORATION COMPLETE")
    print("="*60)
    print("Key takeaways:")
    print("1. Embeddings are stored as (time, features) matrices")
    print("2. Timestamps provide exact temporal mapping")
    print("3. Each hierarchical level has different dimensionality")
    print("4. Access embeddings by timestamp or time range")
    print("5. Compare patterns across sequences/days")

if __name__ == "__main__":
    main()

