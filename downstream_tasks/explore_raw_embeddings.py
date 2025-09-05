import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

# Suppress UserWarning from matplotlib/seaborn
warnings.filterwarnings("ignore", category=UserWarning)

class EmbeddingExplorer:
    def __init__(self, embedding_path, labels_path):
        """
        Initialize the embedding explorer.
        
        Args:
            embedding_path: Path to the .npy embedding file.
            labels_path: Path to the CSV labels file.
        """
        self.embedding_path = embedding_path
        self.labels_path = labels_path
        self.embeddings_data = None
        self.labels_df = None
        self.frame_map = None
        self.load_data()
    
    def load_data(self):
        """Load embeddings and labels from specified paths."""
        print(f"[INFO] Loading embeddings from: {self.embedding_path}")
        emb_data_loaded = np.load(self.embedding_path, allow_pickle=True)
        
        if emb_data_loaded.shape == () and isinstance(emb_data_loaded.item(), dict):
            emb_data = emb_data_loaded.item()
        else:
            raise ValueError("Embedding file is not in the expected dictionary format.")

        self.embeddings_data = emb_data
        self.frame_map = emb_data.get('frame_number_map', {})
        
        print(f"[INFO] Embedding matrix shape: {self.embeddings_data['embeddings'].shape}")
        
        print(f"[INFO] Loading labels from: {self.labels_path}")
        self.labels_df = pd.read_csv(self.labels_path, low_memory=False)
        print(f"[INFO] Labels loaded. Shape: {self.labels_df.shape}")
    
    def get_unique_strains(self):
        """Get all unique strains from the labels dataframe."""
        # Filter out NaN values before sorting
        strain_series = self.labels_df['strain'].dropna()
        unique_strains = sorted(strain_series.unique())
        
        # Check if there were any NaN values and report
        nan_count = self.labels_df['strain'].isna().sum()
        if nan_count > 0:
            print(f"[WARN] Found {nan_count} rows with missing strain values (NaN). These will be excluded from analysis.")
        
        print(f"[INFO] Found {len(unique_strains)} unique strains: {unique_strains}")
        return unique_strains
    
    def get_samples_for_cage(self, strain_name, cage_id):
        """
        Extract all data samples (e.g., daily recordings) for a specific cage of a specific strain.
        """
        # Add additional filter to exclude NaN strain values
        cage_mask = (
            (self.labels_df['strain'] == strain_name) & 
            (self.labels_df['cage_id'] == cage_id) &
            (self.labels_df['strain'].notna())  # Explicitly exclude NaN strains
        )
        cage_labels_df = self.labels_df[cage_mask]
        
        if cage_labels_df.empty:
            return []
        
        cage_samples = []
        for _, row in tqdm(cage_labels_df.iterrows(), desc=f"Processing cage {cage_id}", total=len(cage_labels_df), leave=False):
            day = str(row["from_tpt"]).split()[0]
            key = f"{row['cage_id']}_{day}"
            
            if key in self.frame_map:
                start_idx, end_idx = self.frame_map[key]
                sample_embeddings = self.embeddings_data['embeddings'][start_idx:end_idx]
                
                cage_samples.append({
                    'key': key,
                    'embeddings': sample_embeddings,
                    'labels': row.to_dict(),
                })
        
        return cage_samples

    def analyze_embedding_statistics(self, embeddings):
        """
        Analyze basic statistics of a single embedding matrix.
        --- FIX for RuntimeWarning ---
        Calculations are done in float64 to prevent numerical overflow.
        """
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
        """Saves a numpy matrix to a human-readable text file."""
        np.savetxt(save_path, matrix, fmt='%.8f', delimiter=',')
        print(f"  -> Saved raw matrix to: {save_path}")

    def visualize_raw_embedding_matrix(self, embeddings, title, save_path):
        """Visualizes a raw embedding matrix as a heatmap."""
        plt.figure(figsize=(12, 8))
        # Use yticklabels that are sensible for the number of frames
        num_frames = embeddings.shape[0]
        ytick_freq = max(10, num_frames // 10) # Show ~10 ticks on y-axis
        
        ax = sns.heatmap(embeddings, cmap='viridis', cbar=True, xticklabels=10, yticklabels=ytick_freq)
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Embedding Dimensions', fontsize=12)
        ax.set_ylabel('Time (Frames / Minutes)', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  -> Saved heatmap to: {save_path}")
        plt.close() # Close the figure to free up memory

    def analyze_single_strain(self, strain_name, output_dir):
        """Analyze all cages for a single strain."""
        strain_output_dir = Path(output_dir) / strain_name.replace('/', '_')  # Handle special characters in strain names
        strain_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[INFO] Starting analysis for strain: '{strain_name}'")
        
        # Filter out NaN strain values when getting cages for this strain
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
            
            cage_samples = self.get_samples_for_cage(strain_name, cage_id)
            
            if not cage_samples:
                print(f"[WARN] No embedding samples found for cage '{cage_id}'. Skipping.")
                continue
                
            # --- NEW: Aggregation Logic ---
            all_embeddings_for_cage = []
            
            # --- 1. INDIVIDUAL SAMPLE ANALYSIS ---
            print(f"[INFO] Analyzing {len(cage_samples)} individual samples for this cage...")
            for i, sample in enumerate(cage_samples):
                sample_key = sample['key']
                print(f"--- Analyzing sample {i+1}/{len(cage_samples)}: {sample_key} ---")
                
                self.visualize_raw_embedding_matrix(
                    sample['embeddings'],
                    title=f'Raw Embedding Matrix for: {sample_key}',
                    save_path=cage_output_dir / f"{sample_key}_heatmap.png"
                )
                self.save_matrix_as_txt(
                    sample['embeddings'],
                    save_path=cage_output_dir / f"{sample_key}_matrix.txt"
                )
                stats = self.analyze_embedding_statistics(sample['embeddings'])
                with open(cage_output_dir / f"{sample_key}_stats.json", 'w') as f:
                    json.dump({'sample_info': sample['labels'], 'embedding_stats': stats}, f, indent=2)
                print(f"  -> Saved statistics to: {cage_output_dir / f'{sample_key}_stats.json'}")
                
                # Collect embeddings for aggregation
                all_embeddings_for_cage.append(sample['embeddings'])
                
            # --- 2. AGGREGATED CAGE ANALYSIS ---
            if all_embeddings_for_cage:
                print(f"\n--- Analyzing AGGREGATED view for cage: {cage_id} ---")
                
                aggregated_output_dir = cage_output_dir / "aggregated_view"
                aggregated_output_dir.mkdir(exist_ok=True)
                
                # Combine all embeddings for the cage into one large matrix
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
    parser.add_argument("--embeddings", required=True, help="Path to embeddings .npy file")
    parser.add_argument("--labels", required=True, help="Path to labels .csv file")
    parser.add_argument("--strain", help="Specific strain to analyze (optional). If not provided, all strains will be analyzed.")
    parser.add_argument("--output-dir", default="./embedding_analysis", 
                       help="Main output directory for all results")
    
    args = parser.parse_args()
    
    explorer = EmbeddingExplorer(args.embeddings, args.labels)
    
    # Create main output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.strain:
        # Analyze single strain (original behavior)
        success = explorer.analyze_single_strain(args.strain, args.output_dir)
        if not success:
            print(f"[ERROR] Failed to analyze strain '{args.strain}'")
    else:
        # Analyze all strains (new behavior)
        unique_strains = explorer.get_unique_strains()
        
        if not unique_strains:
            print("[ERROR] No strains found in the dataset.")
            return
        
        print(f"\n[INFO] Starting analysis for ALL {len(unique_strains)} strains...")
        
        successful_strains = []
        failed_strains = []
        
        for strain in tqdm(unique_strains, desc="Processing all strains"):
            try:
                success = explorer.analyze_single_strain(strain, args.output_dir)
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
        
        # Save summary to file
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
    
    
# Usage examples:
# 
# Analyze all strains:
# python3 explore_raw.py --embeddings /scratch/bhole/dvc_data/smoothed/models_3days/embeddings/test_3days_comb.npy --labels /scratch/bhole/dvc_data/smoothed/1440/final_summary_metadata_1440.csv --output-dir ./embedding_analysis
#
# Analyze specific strain (original behavior):
# python3 explore_raw.py --embeddings /scratch/bhole/dvc_data/smoothed/models_3days/embeddings/test_3days_comb.npy --labels /scratch/bhole/dvc_data/smoothed/1440/final_summary_metadata_1440.csv --strain "C57BL/6J" --output-dir ./embedding_analysis