import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import zarr

warnings.filterwarnings("ignore", category=UserWarning)

class EmbeddingExplorer:
    def __init__(self, embedding_path, labels_path, embedding_level='combined'):
        """
        Initialize the embedding explorer.
        
        Args:
            embedding_path: Path to the embedding directory (contains aggregation subdirs)
            labels_path: Path to the CSV labels file
            embedding_level: Which level to analyze ('level_1_pooled', 'combined', etc.)
        """
        self.embedding_path = Path(embedding_path)
        self.labels_path = labels_path
        self.embedding_level = embedding_level
        self.labels_df = None
        self.load_data()
    
    def load_data(self):
        """Load labels from specified path"""
        print(f"[INFO] Loading labels from: {self.labels_path}")
        self.labels_df = pd.read_csv(self.labels_path, low_memory=False)
        print(f"[INFO] Labels loaded. Shape: {self.labels_df.shape}")
    
    def get_unique_strains(self):
        """Get all unique strains from the labels dataframe"""
        strain_series = self.labels_df['strain'].dropna()
        unique_strains = sorted(strain_series.unique())
        
        nan_count = self.labels_df['strain'].isna().sum()
        if nan_count > 0:
            print(f"[WARN] Found {nan_count} rows with missing strain values (NaN). These will be excluded from analysis.")
        
        print(f"[INFO] Found {len(unique_strains)} unique strains: {unique_strains}")
        return unique_strains
    
    def get_samples_for_cage(self, strain_name, cage_id, aggregation_name='1day'):
        """
        Extract all data samples for a specific cage of a specific strain.
        
        NEW: Loads from individual Zarr files instead of concatenated array
        """
        cage_mask = (
            (self.labels_df['strain'] == strain_name) & 
            (self.labels_df['cage_id'] == cage_id) &
            (self.labels_df['strain'].notna())
        )
        cage_labels_df = self.labels_df[cage_mask]
        
        if cage_labels_df.empty:
            return []
        
        cage_samples = []
        
        # Path to aggregation directory
        agg_dir = self.embedding_path / aggregation_name / 'core_embeddings'
        
        for _, row in tqdm(cage_labels_df.iterrows(), desc=f"Processing cage {cage_id}", 
                          total=len(cage_labels_df), leave=False):
            day = str(row["from_tpt"]).split()[0]
            key = f"{row['cage_id']}_{day}"
            
            # NEW: Load from individual Zarr file
            zarr_path = agg_dir / f"{key}.zarr"
            
            if zarr_path.exists():
                try:
                    store = zarr.open(str(zarr_path), mode='r')
                    sample_embeddings = store[self.embedding_level][:]
                    
                    cage_samples.append({
                        'key': key,
                        'embeddings': sample_embeddings,
                        'labels': row.to_dict(),
                        'timestamps': store['timestamps'][:] if 'timestamps' in store else None
                    })
                except Exception as e:
                    print(f"[ERROR] Failed to load {zarr_path}: {e}")
        
        return cage_samples

    def analyze_embedding_statistics(self, embeddings):
        """Analyze basic statistics of a single embedding matrix"""
        stats = {
            'mean_per_dimension': np.mean(embeddings, axis=0, dtype=np.float64).tolist(),
            'std_per_dimension': np.std(embeddings, axis=0, dtype=np.float64).tolist(),
            'min_per_dimension': np.min(embeddings, axis=0).tolist(),
            'max_per_dimension': np.max(embeddings, axis=0).tolist(),
            'global_mean': float(np.mean(embeddings, dtype=np.float64)),
            'global_std': float(np.std(embeddings, dtype=np.float64)),
            'sparsity_fraction': float(np.mean(np.abs(embeddings) < 1e-4)),
        }
        return stats
    
    def save_matrix_as_txt(self, matrix, save_path):
        """Saves a numpy matrix to a human-readable text file"""
        np.savetxt(save_path, matrix, fmt='%.8f', delimiter=',')
        print(f"  -> Saved raw matrix to: {save_path}")

    def visualize_raw_embedding_matrix(self, embeddings, title, save_path, timestamps=None):
        """Visualizes a raw embedding matrix as a heatmap"""
        plt.figure(figsize=(12, 8))
        num_frames = embeddings.shape[0]
        ytick_freq = max(10, num_frames // 10)
        
        ax = sns.heatmap(embeddings, cmap='viridis', cbar=True, 
                        xticklabels=10, yticklabels=ytick_freq)
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Embedding Dimensions', fontsize=12)
        
        # NEW: Use timestamps for y-axis if available
        if timestamps is not None:
            ax.set_ylabel('Time', fontsize=12)
            # Sample timestamps for y-tick labels
            ytick_indices = list(range(0, num_frames, ytick_freq))
            ax.set_yticks(ytick_indices)
            ax.set_yticklabels([timestamps[i] for i in ytick_indices], rotation=0)
        else:
            ax.set_ylabel('Time (Frames / Minutes)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  -> Saved heatmap to: {save_path}")
        plt.close()

    def analyze_single_strain(self, strain_name, output_dir, aggregation_name='1day'):
        """Analyze all cages for a single strain"""
        strain_output_dir = Path(output_dir) / strain_name.replace('/', '_')
        strain_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[INFO] Starting analysis for strain: '{strain_name}'")
        
        strain_mask = (self.labels_df['strain'] == strain_name) & (self.labels_df['strain'].notna())
        strain_cages = sorted(self.labels_df[strain_mask]['cage_id'].unique())
        
        if not strain_cages:
            print(f"[ERROR] No cages found for strain '{strain_name}'.")
            return False

        print(f"[INFO] Found {len(strain_cages)} cages for this strain: {strain_cages}")

        for cage_id in tqdm(strain_cages, desc=f"Processing cages for {strain_name}", leave=False):
            print(f"\n{'='*25} Processing Cage: {cage_id} {'='*25}")
            cage_output_dir = strain_output_dir / cage_id
            cage_output_dir.mkdir(exist_ok=True)
            
            cage_samples = self.get_samples_for_cage(strain_name, cage_id, aggregation_name)
            
            if not cage_samples:
                print(f"[WARN] No embedding samples found for cage '{cage_id}'. Skipping.")
                continue
            
            all_embeddings_for_cage = []
            
            # Individual sample analysis
            print(f"[INFO] Analyzing {len(cage_samples)} individual samples for this cage...")
            for i, sample in enumerate(cage_samples):
                sample_key = sample['key']
                print(f"--- Analyzing sample {i+1}/{len(cage_samples)}: {sample_key} ---")
                
                self.visualize_raw_embedding_matrix(
                    sample['embeddings'],
                    title=f'Raw Embedding Matrix for: {sample_key}',
                    save_path=cage_output_dir / f"{sample_key}_heatmap.png",
                    timestamps=sample.get('timestamps')
                )
                self.save_matrix_as_txt(
                    sample['embeddings'],
                    save_path=cage_output_dir / f"{sample_key}_matrix.txt"
                )
                stats = self.analyze_embedding_statistics(sample['embeddings'])
                with open(cage_output_dir / f"{sample_key}_stats.json", 'w') as f:
                    json.dump({'sample_info': sample['labels'], 'embedding_stats': stats}, f, indent=2)
                print(f"  -> Saved statistics to: {cage_output_dir / f'{sample_key}_stats.json'}")
                
                all_embeddings_for_cage.append(sample['embeddings'])
            
            # Aggregated cage analysis
            if all_embeddings_for_cage:
                print(f"\n--- Analyzing AGGREGATED view for cage: {cage_id} ---")
                
                aggregated_output_dir = cage_output_dir / "aggregated_view"
                aggregated_output_dir.mkdir(exist_ok=True)
                
                aggregated_matrix = np.vstack(all_embeddings_for_cage)
                print(f"[INFO] Created aggregated matrix with shape: {aggregated_matrix.shape}")
                
                self.visualize_raw_embedding_matrix(
                    aggregated_matrix,
                    title=f'Aggregated Embedding Matrix for Cage: {cage_id}',
                    save_path=aggregated_output_dir / f"{cage_id}_aggregated_heatmap.png"
                )
                self.save_matrix_as_txt(
                    aggregated_matrix,
                    save_path=aggregated_output_dir / f"{cage_id}_aggregated_matrix.txt"
                )
                stats = self.analyze_embedding_statistics(aggregated_matrix)
                with open(aggregated_output_dir / f"{cage_id}_aggregated_stats.json", 'w') as f:
                    json.dump({'cage_id': cage_id, 'embedding_stats': stats}, f, indent=2)
                print(f"  -> Saved aggregated statistics to: {aggregated_output_dir / f'{cage_id}_aggregated_stats.json'}")

        print(f"[SUCCESS] Analysis complete for strain '{strain_name}'. Files saved in: {strain_output_dir}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Export and visualize raw embeddings for all strains or a specific strain.")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings directory (contains aggregation subdirs)")
    parser.add_argument("--labels", required=True, help="Path to labels .csv file")
    parser.add_argument("--strain", help="Specific strain to analyze (optional)")
    parser.add_argument("--output-dir", default="./embedding_analysis", help="Main output directory")
    parser.add_argument("--aggregation", default="1day", help="Aggregation level to analyze (e.g., '1day', '1.5h')")
    parser.add_argument("--embedding-level", default="combined", 
                       help="Which embedding level to analyze ('level_1_pooled', 'combined', etc.)")
    
    args = parser.parse_args()
    
    explorer = EmbeddingExplorer(args.embeddings, args.labels, args.embedding_level)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.strain:
        success = explorer.analyze_single_strain(args.strain, args.output_dir, args.aggregation)
        if not success:
            print(f"[ERROR] Failed to analyze strain '{args.strain}'")
    else:
        unique_strains = explorer.get_unique_strains()
        
        if not unique_strains:
            print("[ERROR] No strains found in the dataset.")
            return
        
        print(f"\n[INFO] Starting analysis for ALL {len(unique_strains)} strains...")
        
        successful_strains = []
        failed_strains = []
        
        for strain in tqdm(unique_strains, desc="Processing all strains"):
            try:
                success = explorer.analyze_single_strain(strain, args.output_dir, args.aggregation)
                if success:
                    successful_strains.append(strain)
                else:
                    failed_strains.append(strain)
            except Exception as e:
                print(f"[ERROR] Failed to analyze strain '{strain}': {str(e)}")
                failed_strains.append(strain)
        
        # Summary report
        print(f"\n{'='*50}")
        print(f"ANALYSIS SUMMARY")
        print(f"{'='*50}")
        print(f"Total strains processed: {len(unique_strains)}")
        print(f"Successful: {len(successful_strains)}")
        print(f"Failed: {len(failed_strains)}")
        
        if successful_strains:
            print(f"\nSuccessfully analyzed strains:")
            for strain in successful_strains:
                print(f"  ✓ {strain}")
        
        if failed_strains:
            print(f"\nFailed to analyze strains:")
            for strain in failed_strains:
                print(f"  ✗ {strain}")
        
        summary = {
            'total_strains': len(unique_strains),
            'successful_strains': successful_strains,
            'failed_strains': failed_strains,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[SUCCESS] Complete analysis finished. Summary saved to: {output_dir / 'analysis_summary.json'}")


if __name__ == "__main__":
    main()