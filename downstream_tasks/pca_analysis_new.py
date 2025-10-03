#!/usr/bin/env python3
"""
Multi-Level Temporal UMAP Analysis by Strain
Creates UMAP visualizations showing all temporal frames across embedding levels,
with age progression and cage-specific markers.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import umap
import zarr
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

plt.style.use('dark_background')

class TemporalStrainUMAPAnalyzer:
    """Analyzer for temporal UMAP visualizations by strain"""
    
    def __init__(self, embeddings_dir, metadata_path, aggregation, output_dir="temporal_strain_umaps"):
        self.embeddings_dir = Path(embeddings_dir)
        self.metadata_path = metadata_path
        self.aggregation = aggregation
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Embedding levels to analyze
        self.embedding_levels = [
            'level_1_pooled', 'level_2_pooled', 'level_3_pooled', 'level_4_pooled',
            'level_5_pooled', 'level_6_pooled', 'level_7_pooled', 'combined'
        ]
        
        # Markers for different cages
        self.markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>']
        
        # Load metadata
        self.load_metadata()
        
    def load_metadata(self):
        """Load and prepare metadata"""
        print("Loading metadata...")
        self.metadata = pd.read_csv(self.metadata_path)
        #  please replcae / by _ in the strain column
        self.metadata['strain'] = self.metadata['strain'].str.replace('/', '_')
        self.metadata['from_tpt'] = pd.to_datetime(self.metadata['from_tpt'])
        self.metadata['key'] = self.metadata.apply(
            lambda row: f"{row['cage_id']}_{row['from_tpt'].strftime('%Y-%m-%d')}", axis=1
        )
        
        # Filter for valid entries
        self.metadata = self.metadata[
            self.metadata['strain'].notna() & 
            self.metadata['avg_age_days_chunk_start'].notna()
        ].copy()
        
        print(f"Loaded metadata for {len(self.metadata)} entries")
        print(f"Strains: {self.metadata['strain'].nunique()}")
    
    def load_temporal_data_for_level(self, level, keys_to_load):
        """Load temporal embeddings for a specific level and set of keys"""
        core_emb_dir = self.embeddings_dir / self.aggregation / 'core_embeddings'
        
        temporal_data = {}
        
        for key in tqdm(keys_to_load, desc=f"Loading {level}", leave=False):
            zarr_path = core_emb_dir / f"{key}.zarr"
            
            if zarr_path.exists():
                try:
                    store = zarr.open(str(zarr_path), mode='r')
                    
                    if level not in store:
                        continue
                    
                    seq_embeddings = store[level][:]
                    
                    # We want temporal data (2D)
                    if seq_embeddings.ndim == 2 and seq_embeddings.shape[0] == 40320:
                        temporal_data[key] = seq_embeddings
                    elif seq_embeddings.ndim == 1:
                        # If it's already aggregated, skip
                        continue
                        
                except Exception as e:
                    pass
        
        return temporal_data
    
    def compute_umap_for_level(self, level, strain_data):
        """Compute UMAP for a specific level and strain"""
        print(f"  Computing UMAP for {level}...")
        
        # Collect all temporal frames
        all_frames = []
        frame_metadata = []
        
        for entry in strain_data:
            key = entry['key']
            temporal_seq = entry['temporal_data']
            start_age = entry['start_age']
            cage_id = entry['cage_id']
            
            # Each frame gets an age based on its position
            for frame_idx in range(temporal_seq.shape[0]):
                frame_emb = temporal_seq[frame_idx]
                
                if not np.isnan(frame_emb).any():
                    # Calculate age: start_age + days_elapsed
                    days_elapsed = frame_idx // 1440
                    frame_age = start_age + days_elapsed
                    
                    all_frames.append(frame_emb)
                    frame_metadata.append({
                        'key': key,
                        'cage_id': cage_id,
                        'age': frame_age,
                        'frame_idx': frame_idx
                    })
        
        if len(all_frames) == 0:
            return None, None
        
        X = np.array(all_frames)
        
        print(f"    Total frames: {len(X):,}")
        
        # Standardize
        X_scaled = StandardScaler().fit_transform(X)
        
        # Compute UMAP
        reducer = umap.UMAP(
            n_neighbors=30, 
            min_dist=0.1,
            n_jobs=-1,
            verbose=False
        )
        X_umap = reducer.fit_transform(X_scaled)
        
        return X_umap, frame_metadata
    
    def process_strain(self, strain):
        """Process all embedding levels for a single strain"""
        print(f"\n{'='*70}")
        print(f"Processing strain: {strain}")
        print(f"{'='*70}")
        
        # Get all entries for this strain
        strain_metadata = self.metadata[self.metadata['strain'] == strain].copy()
        
        if len(strain_metadata) == 0:
            print(f"No data for strain {strain}")
            return
        
        keys = strain_metadata['key'].tolist()
        print(f"Found {len(keys)} entries for strain {strain}")
        
        # Get unique cages
        unique_cages = strain_metadata['cage_id'].unique()
        print(f"Cages: {len(unique_cages)} - {list(unique_cages)}")
        
        # Create cage-to-marker mapping
        cage_markers = {cage: self.markers[i % len(self.markers)] 
                       for i, cage in enumerate(unique_cages)}
        
        # Load temporal data for all levels
        level_temporal_data = {}
        
        for level in self.embedding_levels:
            print(f"\nLoading temporal data for {level}...")
            temporal_data = self.load_temporal_data_for_level(level, keys)
            
            if temporal_data:
                # Organize by cage
                strain_data = []
                for _, row in strain_metadata.iterrows():
                    key = row['key']
                    if key in temporal_data:
                        strain_data.append({
                            'key': key,
                            'cage_id': row['cage_id'],
                            'start_age': row['avg_age_days_chunk_start'],
                            'temporal_data': temporal_data[key]
                        })
                
                level_temporal_data[level] = strain_data
                print(f"  Loaded {len(strain_data)} entries with temporal data")
        
        if not level_temporal_data:
            print(f"No temporal data loaded for strain {strain}")
            return
        
        # Compute UMAP for each level
        umap_results = {}
        
        for level, strain_data in level_temporal_data.items():
            X_umap, frame_metadata = self.compute_umap_for_level(level, strain_data)
            if X_umap is not None:
                umap_results[level] = {
                    'umap': X_umap,
                    'metadata': frame_metadata
                }
        
        if not umap_results:
            print(f"No UMAP results for strain {strain}")
            return
        
        # Create visualization
        self.create_strain_visualization(strain, umap_results, cage_markers)
    
    def create_strain_visualization(self, strain, umap_results, cage_markers):
        """Create multi-level UMAP visualization for a strain"""
        print(f"\nCreating visualization for strain {strain}...")
        
        # Create figure with 2 rows × 4 columns
        fig, axes = plt.subplots(2, 4, figsize=(28, 14), facecolor='black')
        axes = axes.flatten()
        
        # Get global age range for this strain
        all_ages = []
        for level_data in umap_results.values():
            all_ages.extend([m['age'] for m in level_data['metadata']])
        
        age_min, age_max = min(all_ages), max(all_ages)
        
        # Plot each level
        for idx, level in enumerate(self.embedding_levels):
            ax = axes[idx]
            ax.set_facecolor('black')
            
            if level not in umap_results:
                ax.text(0.5, 0.5, f"No data\nfor {level}", 
                       transform=ax.transAxes, ha='center', va='center',
                       color='white', fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            level_data = umap_results[level]
            X_umap = level_data['umap']
            metadata = level_data['metadata']
            
            # Extract data
            ages = np.array([m['age'] for m in metadata])
            cage_ids = np.array([m['cage_id'] for m in metadata])
            
            # Plot by cage with different markers
            unique_cages = np.unique(cage_ids)
            
            for cage_id in unique_cages:
                mask = cage_ids == cage_id
                marker = cage_markers[cage_id]
                
                scatter = ax.scatter(
                    X_umap[mask, 0], 
                    X_umap[mask, 1],
                    c=ages[mask],
                    cmap='viridis',
                    vmin=age_min,
                    vmax=age_max,
                    s=8,
                    alpha=0.6,
                    marker=marker,
                    edgecolors='none',
                    label=f'Cage {cage_id}'
                )
            
            # Format level name
            if 'level_' in level:
                level_num = level.split('_')[1]
                display_name = f"LEVEL{level_num}"
            else:
                display_name = "COMBINED"
            
            # Estimate timespan (rough approximation)
            timespans = {
                'level_1_pooled': '(5min)',
                'level_2_pooled': '(30min)',
                'level_3_pooled': '(2hrs)',
                'level_4_pooled': '(6hrs)',
                'level_5_pooled': '(1day)',
                'level_6_pooled': '(2days)',
                'level_7_pooled': '(4days)',
                'combined': '(concatenated)'
            }
            timespan = timespans.get(level, '')
            
            ax.set_title(f"{display_name}\n{timespan}", 
                        fontsize=13, color='white', pad=10, fontweight='bold')
            
            # Only show axis labels on left and bottom
            if idx >= 4:  # Bottom row
                ax.set_xlabel('UMAP-1', fontsize=11, color='white')
            if idx % 4 == 0:  # Left column
                ax.set_ylabel('UMAP-2', fontsize=11, color='white')
            
            ax.tick_params(colors='white', labelsize=9)
            ax.grid(True, alpha=0.2, color='gray', linewidth=0.5)
            
            # Add legend for first subplot
            if idx == 0:
                legend = ax.legend(loc='upper left', fontsize=9, framealpha=0.8,
                                 facecolor='black', edgecolor='white')
                for text in legend.get_texts():
                    text.set_color('white')
        
        # Add single colorbar for the entire figure
        # Use the last scatter for colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = plt.colorbar(scatter, cax=cbar_ax, label='Age (days)')
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.yaxis.label.set_fontsize(12)
        cbar.ax.tick_params(colors='white', labelsize=10)
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Overall title
        cage_list = ', '.join([str(c) for c in sorted(cage_markers.keys())])
        fig.suptitle(
            f"Strain: {strain} | Cages ({len(cage_markers)}): {cage_list}\n"
            f"Temporal UMAP Across All Embedding Levels",
            fontsize=18, color='white', y=0.97, fontweight='bold'
        )
        
        plt.tight_layout(rect=[0, 0, 0.90, 0.95])
        
        # Save
        output_path = self.output_dir / f"temporal_umap_{strain}.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
        print(f"Saved: {output_path}")
    
    def run_analysis(self, strains_to_process=None):
        """Run analysis for all or specific strains"""
        print("\n" + "="*70)
        print("TEMPORAL STRAIN UMAP ANALYSIS")
        print("="*70)
        
        # Get strains to process
        if strains_to_process is None:
            strains = sorted(self.metadata['strain'].unique())
        else:
            strains = strains_to_process
        
        print(f"\nProcessing {len(strains)} strains...")
        
        # Process each strain
        for strain in strains:
            try:
                self.process_strain(strain)
            except Exception as e:
                print(f"ERROR processing strain {strain}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Temporal UMAP Analysis by Strain Across Embedding Levels'
    )
    parser.add_argument('--embeddings', required=True,
                       help='Path to embeddings directory')
    parser.add_argument('--metadata', required=True,
                       help='Path to metadata CSV')
    parser.add_argument('--aggregation', required=True,
                       help='Aggregation level (e.g., "4weeks")')
    parser.add_argument('--output-dir', default='temporal_strain_umaps',
                       help='Output directory')
    parser.add_argument('--strains', nargs='+', default=None,
                       help='Specific strains to process (default: all)')
    
    args = parser.parse_args()
    
    analyzer = TemporalStrainUMAPAnalyzer(
        embeddings_dir=args.embeddings,
        metadata_path=args.metadata,
        aggregation=args.aggregation,
        output_dir=args.output_dir
    )
    
    analyzer.run_analysis(strains_to_process=args.strains)


if __name__ == "__main__":
    main()
    
    
# # Process all strains
# python downstream_tasks/pca_analysis_new.py \
#   --embeddings /scratch/bhole/dvc_data/smoothed/models_4weeks/embeddings \
#   --metadata /scratch/bhole/dvc_data/smoothed/40320/summary_metadata_40320.csv \
#   --aggregation 4weeks \
#   --output-dir temporal_strain_umaps_4weeks

# # Process specific strains only
# python downstream_tasks/pca_analysis_new.py \
#   --embeddings /scratch/bhole/dvc_data/smoothed/models_4weeks/embeddings \
#   --metadata /scratch/bhole/dvc_data/smoothed/40320/summary_metadata_40320.csv \
#   --aggregation 4weeks \
#   --output-dir temporal_strain_umaps_4weeks \
#   --strains BXD102 BXD34 "C57BL/6J"