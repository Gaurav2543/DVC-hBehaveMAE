#!/usr/bin/env python3
"""
Enhanced PCA and UMAP Analysis with:
1. Per-cage temporal trajectories (all cages)
2. Multi-level UMAP comparisons (like Image 2)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
import umap
import zarr
from tqdm import tqdm
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

plt.style.use('dark_background')

class EnhancedEmbeddingAnalyzer:
    """Enhanced analyzer with comprehensive temporal and multi-level visualizations"""
    
    def __init__(self, embeddings_dir, metadata_path, aggregation, embedding_level, 
                 output_dir="enhanced_analysis_results"):
        self.embeddings_dir = Path(embeddings_dir)
        self.metadata_path = metadata_path
        self.aggregation = aggregation
        self.embedding_level = embedding_level
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.output_dir / "temporal_trajectories").mkdir(exist_ok=True)
        (self.output_dir / "multi_level_umaps").mkdir(exist_ok=True)
        (self.output_dir / "pca_analysis").mkdir(exist_ok=True)
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load embeddings and metadata"""
        print("Loading data...")
        
        core_emb_dir = self.embeddings_dir / self.aggregation / 'core_embeddings'
        if not core_emb_dir.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {core_emb_dir}")
        
        # Load metadata
        self.metadata = pd.read_csv(self.metadata_path)
        #    take the first 50 entries for testing
        self.metadata = self.metadata.head(50)  
        # take onyl those entries for which the column nb_tpt_activity_data is equal to 40320
        self.metadata = self.metadata[self.metadata['nb_tpt_activity_data'] == 40320]
        
        self.metadata['from_tpt'] = pd.to_datetime(self.metadata['from_tpt'])
        self.metadata['key'] = self.metadata.apply(
            lambda row: f"{row['cage_id']}_{row['from_tpt'].strftime('%Y-%m-%d')}", axis=1
        )
        
        # Load embeddings
        self.embeddings_dict = {}
        self.temporal_embeddings_dict = {}
        
        for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), 
                          desc="Loading embeddings"):
            key = row['key']
            zarr_path = core_emb_dir / f"{key}.zarr"
            
            if zarr_path.exists():
                try:
                    store = zarr.open(str(zarr_path), mode='r')
                    
                    if self.embedding_level not in store:
                        level_to_use = 'combined'
                    else:
                        level_to_use = self.embedding_level
                    
                    seq_embeddings = store[level_to_use][:]
                    
                    # Store both temporal and aggregated versions
                    if seq_embeddings.ndim == 2:
                        self.temporal_embeddings_dict[key] = seq_embeddings
                        daily_embedding = seq_embeddings.mean(axis=0)
                    else:
                        daily_embedding = seq_embeddings
                    
                    if not np.isnan(daily_embedding).any():
                        self.embeddings_dict[key] = daily_embedding
                        
                except Exception as e:
                    print(f"[WARN] Failed to load {zarr_path}: {e}")
        
        # Create aligned dataset
        self.create_aligned_dataset()
        
        print(f"Loaded {self.X.shape[0]} samples with {self.X.shape[1]} dimensions")
        print(f"Temporal embeddings available for {len(self.temporal_embeddings_dict)} samples")
    
    def create_aligned_dataset(self):
        """Create aligned embeddings and metadata"""
        df_with_embeddings = self.metadata[self.metadata['key'].isin(self.embeddings_dict.keys())].copy()
        
        aligned_embeddings = []
        aligned_metadata = []
        
        for _, row in df_with_embeddings.iterrows():
            key = row['key']
            embedding = self.embeddings_dict[key]
            
            if pd.notna(row.get('avg_age_days_chunk_start')) and pd.notna(row.get('strain')):
                aligned_embeddings.append(embedding)
                aligned_metadata.append(row)
        
        self.X = np.array(aligned_embeddings)
        self.metadata_aligned = pd.DataFrame(aligned_metadata).reset_index(drop=True)
    
    def plot_all_temporal_trajectories(self):
        """Plot temporal trajectories for ALL cages in a multi-page PDF"""
        print("\n" + "="*70)
        print("PLOTTING ALL TEMPORAL TRAJECTORIES")
        print("="*70)
        
        if not self.temporal_embeddings_dict:
            print("No temporal embeddings available")
            return
        
        # Collect all temporal frames
        print("Collecting temporal data...")
        all_temporal_frames = []
        frame_metadata = []
        
        for _, row in self.metadata_aligned.iterrows():
            key = row['key']
            if key in self.temporal_embeddings_dict:
                temporal_seq = self.temporal_embeddings_dict[key]
                for frame_idx, frame_emb in enumerate(temporal_seq):
                    if not np.isnan(frame_emb).any():
                        all_temporal_frames.append(frame_emb)
                        frame_metadata.append({
                            'key': key,
                            'cage_id': row['cage_id'],
                            'strain': row['strain'],
                            'age': row['avg_age_days_chunk_start'],
                            'frame_idx': frame_idx
                        })
        
        X_temporal = np.array(all_temporal_frames)
        
        # Compute UMAP for all temporal data
        print("Computing UMAP embedding (this may take a while)...")
        X_temporal_scaled = StandardScaler().fit_transform(X_temporal)
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_jobs=-1)
        X_umap = reducer.fit_transform(X_temporal_scaled)
        
        # Group by cage
        unique_cages = sorted(set([m['cage_id'] for m in frame_metadata]))
        n_cages = len(unique_cages)
        
        print(f"Creating trajectory plots for {n_cages} cages...")
        
        # Create multi-page PDF (6 cages per page)
        pdf_path = self.output_dir / "temporal_trajectories" / "all_cage_trajectories.pdf"
        
        with PdfPages(pdf_path) as pdf:
            plots_per_page = 6
            n_pages = int(np.ceil(n_cages / plots_per_page))
            
            for page_idx in tqdm(range(n_pages), desc="Generating pages"):
                fig, axes = plt.subplots(2, 3, figsize=(22, 14), facecolor='black')
                axes = axes.flatten()
                
                start_idx = page_idx * plots_per_page
                end_idx = min(start_idx + plots_per_page, n_cages)
                page_cages = unique_cages[start_idx:end_idx]
                
                for ax_idx, cage_id in enumerate(page_cages):
                    ax = axes[ax_idx]
                    ax.set_facecolor('black')
                    
                    # Plot all points in gray (background)
                    ax.scatter(X_umap[:, 0], X_umap[:, 1], c='gray', s=1, alpha=0.05)
                    
                    # Get this cage's trajectory
                    cage_mask = [m['cage_id'] == cage_id for m in frame_metadata]
                    cage_trajectory = X_umap[cage_mask]
                    
                    if len(cage_trajectory) == 0:
                        ax.text(0.5, 0.5, f"No data for cage {cage_id}", 
                               transform=ax.transAxes, ha='center', va='center',
                               color='white', fontsize=12)
                        continue
                    
                    # Color by time
                    n_frames = len(cage_trajectory)
                    colors = plt.cm.viridis(np.linspace(0, 1, n_frames))
                    
                    # Plot trajectory
                    ax.scatter(cage_trajectory[:, 0], cage_trajectory[:, 1], 
                              c=colors, s=15, alpha=0.8, edgecolors='white', linewidths=0.3)
                    ax.plot(cage_trajectory[:, 0], cage_trajectory[:, 1], 
                           'white', alpha=0.2, linewidth=1)
                    
                    # Mark start and end
                    ax.scatter(cage_trajectory[0, 0], cage_trajectory[0, 1], 
                              c='lime', s=200, marker='*', edgecolors='white', 
                              linewidths=2, zorder=10, label='Start')
                    ax.scatter(cage_trajectory[-1, 0], cage_trajectory[-1, 1], 
                              c='red', s=200, marker='X', edgecolors='white', 
                              linewidths=2, zorder=10, label='End')
                    
                    # Get cage info
                    cage_info = [m for m in frame_metadata if m['cage_id'] == cage_id][0]
                    
                    ax.set_title(f"Cage {cage_id} | Strain: {cage_info['strain']}\n"
                                f"Age: {cage_info['age']:.1f} days | Frames: {n_frames}", 
                                fontsize=11, color='white', pad=10)
                    ax.set_xlabel('UMAP-1', color='white', fontsize=10)
                    ax.set_ylabel('UMAP-2', color='white', fontsize=10)
                    ax.tick_params(colors='white', labelsize=8)
                    ax.grid(True, alpha=0.2, color='gray')
                    
                    # Add legend
                    legend = ax.legend(loc='upper right', fontsize=8, framealpha=0.7)
                    legend.get_frame().set_facecolor('black')
                    legend.get_frame().set_edgecolor('white')
                    for text in legend.get_texts():
                        text.set_color('white')
                
                # Hide unused subplots
                for ax_idx in range(len(page_cages), plots_per_page):
                    axes[ax_idx].axis('off')
                
                fig.suptitle(f'Temporal Trajectories in Embedding Space (Per-Day Progression)\n'
                            f'Page {page_idx + 1}/{n_pages}', 
                            fontsize=16, color='white', y=0.98)
                
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                pdf.savefig(fig, facecolor='black')
                plt.close(fig)
        
        print(f"Saved all trajectories to: {pdf_path}")
    
    def plot_multi_level_umaps_by_strain(self, embedding_levels=None):
        """
        Plot UMAP comparisons across multiple embedding levels (like Image 2)
        Groups by strain, shows multiple cages per strain
        """
        print("\n" + "="*70)
        print("CREATING MULTI-LEVEL UMAP COMPARISONS BY STRAIN")
        print("="*70)
        
        if embedding_levels is None:
            # Default: test a few representative levels
            embedding_levels = ['level_3_pooled', 'level_4_pooled', 'level_5_pooled', 'combined']
        
        # Load embeddings for all requested levels
        print("Loading embeddings for multiple levels...")
        multi_level_data = {}
        
        for level in embedding_levels:
            print(f"  Loading {level}...")
            level_embeddings = {}
            
            core_emb_dir = self.embeddings_dir / self.aggregation / 'core_embeddings'
            
            for _, row in tqdm(self.metadata_aligned.iterrows(), 
                             desc=f"Loading {level}", leave=False):
                key = row['key']
                zarr_path = core_emb_dir / f"{key}.zarr"
                
                if zarr_path.exists():
                    try:
                        store = zarr.open(str(zarr_path), mode='r')
                        
                        if level not in store:
                            continue
                        
                        seq_embeddings = store[level][:]
                        
                        if seq_embeddings.ndim == 2:
                            daily_embedding = seq_embeddings.mean(axis=0)
                        else:
                            daily_embedding = seq_embeddings
                        
                        if not np.isnan(daily_embedding).any():
                            level_embeddings[key] = daily_embedding
                            
                    except Exception as e:
                        pass
            
            if level_embeddings:
                multi_level_data[level] = level_embeddings
        
        if not multi_level_data:
            print("No multi-level data loaded")
            return
        
        # Compute UMAP for each level
        print("Computing UMAP embeddings for each level...")
        umap_projections = {}
        
        for level, embeddings_dict in multi_level_data.items():
            # Align with metadata
            aligned_embs = []
            aligned_meta = []
            
            for _, row in self.metadata_aligned.iterrows():
                key = row['key']
                if key in embeddings_dict:
                    aligned_embs.append(embeddings_dict[key])
                    aligned_meta.append(row)
            
            X_level = np.array(aligned_embs)
            meta_level = pd.DataFrame(aligned_meta)
            
            # Compute UMAP
            X_scaled = StandardScaler().fit_transform(X_level)
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
            X_umap = reducer.fit_transform(X_scaled)
            
            umap_projections[level] = {
                'umap': X_umap,
                'metadata': meta_level
            }
        
        # Group by strain and create plots
        print("Creating strain-based multi-level UMAP plots...")
        
        # Get strains with multiple cages
        strain_cage_counts = self.metadata_aligned.groupby('strain')['cage_id'].nunique()
        strains_with_multiple_cages = strain_cage_counts[strain_cage_counts >= 2].index.tolist()
        
        pdf_path = self.output_dir / "multi_level_umaps" / "multi_level_umap_by_strain.pdf"
        
        with PdfPages(pdf_path) as pdf:
            for strain in tqdm(strains_with_multiple_cages, desc="Processing strains"):
                # Get cages for this strain
                strain_cages = self.metadata_aligned[
                    self.metadata_aligned['strain'] == strain
                ]['cage_id'].unique()
                
                if len(strain_cages) < 2:
                    continue
                
                # Limit to first 2 cages for cleaner visualization
                selected_cages = strain_cages[:2]
                
                # Create figure
                fig, axes = plt.subplots(len(selected_cages), len(umap_projections), 
                                        figsize=(6*len(umap_projections), 5*len(selected_cages)),
                                        facecolor='black')
                
                if len(selected_cages) == 1:
                    axes = axes.reshape(1, -1)
                
                for cage_idx, cage_id in enumerate(selected_cages):
                    for level_idx, (level_name, proj_data) in enumerate(umap_projections.items()):
                        ax = axes[cage_idx, level_idx]
                        ax.set_facecolor('black')
                        
                        # Get data for this cage
                        cage_mask = proj_data['metadata']['cage_id'] == cage_id
                        cage_umap = proj_data['umap'][cage_mask]
                        cage_ages = proj_data['metadata'][cage_mask]['avg_age_days_chunk_start'].values
                        
                        if len(cage_umap) == 0:
                            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                                   ha='center', va='center', color='white')
                            continue
                        
                        # Plot all strain data in background
                        strain_mask = proj_data['metadata']['strain'] == strain
                        strain_umap = proj_data['umap'][strain_mask]
                        ax.scatter(strain_umap[:, 0], strain_umap[:, 1], 
                                  c='gray', s=10, alpha=0.2)
                        
                        # Plot this cage colored by age
                        scatter = ax.scatter(cage_umap[:, 0], cage_umap[:, 1], 
                                           c=cage_ages, cmap='viridis', 
                                           s=50, alpha=0.8, edgecolors='white', linewidths=0.5)
                        
                        # Format level name
                        if 'level_' in level_name:
                            level_num = level_name.split('_')[1]
                            display_name = f"LEVEL{level_num}_PERCENTILES"
                        else:
                            display_name = "COMB_PERCENTILES"
                        
                        # Add title with estimated timespan
                        timespans = {
                            'level_3_pooled': '(2hrs)',
                            'level_4_pooled': '(6hrs)', 
                            'level_5_pooled': '(12hrs)',
                            'combined': '(concatenated)'
                        }
                        timespan = timespans.get(level_name, '')
                        
                        if cage_idx == 0:
                            ax.set_title(f"{display_name}\n{timespan}", 
                                       fontsize=12, color='white', pad=10)
                        
                        ax.tick_params(colors='white', labelsize=8)
                        ax.grid(True, alpha=0.2, color='gray')
                        
                        # Add colorbar
                        if level_idx == len(umap_projections) - 1:
                            cbar = plt.colorbar(scatter, ax=ax)
                            cbar.set_label('Average Age (days)', color='white', fontsize=10)
                            cbar.ax.yaxis.set_tick_params(color='white', labelsize=8)
                            cbar.outline.set_edgecolor('white')
                            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
                        
                        # Y-label for first column
                        if level_idx == 0:
                            ax.set_ylabel(f'Cage {cage_id}', fontsize=12, color='white', 
                                        rotation=90, labelpad=10)
                
                # Overall title
                cage_list = ', '.join([f"EHDP-{str(c).zfill(5)}" for c in selected_cages])
                fig.suptitle(f"Strain: {strain} | Cages ({len(selected_cages)}): {cage_list}", 
                           fontsize=16, color='white', y=0.98)
                
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                pdf.savefig(fig, facecolor='black')
                plt.close(fig)
        
        print(f"Saved multi-level UMAPs to: {pdf_path}")
    
    def perform_pca_analysis(self):
        """Perform comprehensive PCA analysis"""
        print("\n" + "="*70)
        print("PERFORMING PCA ANALYSIS")
        print("="*70)
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # Fit PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        self.pca = pca
        self.scaler = scaler
        self.X_pca = X_pca
        
        # Analyze
        self.analyze_variance_explained()
        self.analyze_pc1_contributions()
        self.plot_pca_projections()
        
        return X_pca
    
    def analyze_variance_explained(self):
        """Analyze variance explained by PCs"""
        variance_ratios = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratios)
        
        n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
        
        print(f"\nFirst 10 PCs:")
        for i in range(min(10, len(variance_ratios))):
            print(f"  PC{i+1}: {variance_ratios[i]*100:.2f}% "
                  f"(Cumulative: {cumulative_variance[i]*100:.2f}%)")
        print(f"\nComponents for 90% variance: {n_components_90}")
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='black')
        
        for ax in axes:
            ax.set_facecolor('black')
        
        n_plot = min(50, len(variance_ratios))
        
        axes[0].bar(range(1, n_plot + 1), variance_ratios[:n_plot], alpha=0.8, color='cyan')
        axes[0].set_xlabel('Principal Component', fontsize=13, color='white')
        axes[0].set_ylabel('Variance Explained', fontsize=13, color='white')
        axes[0].set_title('Individual Variance Explained', fontsize=15, color='white', pad=15)
        axes[0].tick_params(colors='white', labelsize=10)
        axes[0].grid(True, alpha=0.3, color='gray')
        
        axes[1].plot(range(1, n_plot + 1), cumulative_variance[:n_plot], 
                    'o-', color='orange', linewidth=2, markersize=5)
        axes[1].axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='90% threshold')
        axes[1].axvline(x=n_components_90, color='red', linestyle='--', alpha=0.5, linewidth=2)
        axes[1].set_xlabel('Number of Components', fontsize=13, color='white')
        axes[1].set_ylabel('Cumulative Variance', fontsize=13, color='white')
        axes[1].set_title('Cumulative Variance Explained', fontsize=15, color='white', pad=15)
        axes[1].tick_params(colors='white', labelsize=10)
        axes[1].legend(fontsize=11, facecolor='black', edgecolor='white', 
                      labelcolor='white')
        axes[1].grid(True, alpha=0.3, color='gray')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pca_analysis' / 'variance_explained.png', 
                   dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
    
    def analyze_pc1_contributions(self):
        """Analyze PC1 contributions from embedding dimensions"""
        pc1_loadings = self.pca.components_[0]
        abs_loadings = np.abs(pc1_loadings)
        top_indices = np.argsort(abs_loadings)[::-1][:20]
        
        print("\nTop 20 dimensions for PC1:")
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. Dim {idx}: {pc1_loadings[idx]:.4f}")
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='black')
        
        for ax in axes:
            ax.set_facecolor('black')
        
        axes[0].bar(range(len(pc1_loadings)), pc1_loadings, alpha=0.8, color='cyan')
        axes[0].set_xlabel('Embedding Dimension', fontsize=13, color='white')
        axes[0].set_ylabel('PC1 Loading', fontsize=13, color='white')
        axes[0].set_title('PC1 Loadings (All Dimensions)', fontsize=15, color='white', pad=15)
        axes[0].tick_params(colors='white', labelsize=10)
        axes[0].grid(True, alpha=0.3, color='gray')
        
        top_dims = top_indices[:30]
        top_vals = pc1_loadings[top_dims]
        colors = ['lime' if x > 0 else 'red' for x in top_vals]
        
        axes[1].barh(range(len(top_dims)), top_vals, color=colors, alpha=0.8)
        axes[1].set_yticks(range(len(top_dims)))
        axes[1].set_yticklabels([f'Dim {i}' for i in top_dims], fontsize=9, color='white')
        axes[1].set_xlabel('PC1 Loading', fontsize=13, color='white')
        axes[1].set_title('Top 30 Contributors to PC1', fontsize=15, color='white', pad=15)
        axes[1].tick_params(colors='white', labelsize=10)
        axes[1].grid(True, alpha=0.3, axis='x', color='gray')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pca_analysis' / 'pc1_contributions.png', 
                   dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
    
    def plot_pca_projections(self):
        """Plot PCA projections"""
        y_age = self.metadata_aligned['avg_age_days_chunk_start'].values
        y_strain = self.metadata_aligned['strain'].values
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor='black')
        
        for ax in axes:
            ax.set_facecolor('black')
        
        # By age
        scatter1 = axes[0].scatter(self.X_pca[:, 0], self.X_pca[:, 1], 
                                   c=y_age, cmap='viridis', s=30, alpha=0.7, 
                                   edgecolors='white', linewidths=0.3)
        axes[0].set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)', 
                          fontsize=13, color='white')
        axes[0].set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)', 
                          fontsize=13, color='white')
        axes[0].set_title('PCA Projection (Colored by Age)', fontsize=15, color='white', pad=15)
        axes[0].tick_params(colors='white', labelsize=10)
        axes[0].grid(True, alpha=0.3, color='gray')
        
        cbar1 = plt.colorbar(scatter1, ax=axes[0])
        cbar1.set_label('Age (days)', fontsize=12, color='white')
        cbar1.ax.yaxis.set_tick_params(color='white', labelsize=10)
        cbar1.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='white')
        
        # By strain
        unique_strains = np.unique(y_strain)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_strains)))
        
        for strain, color in zip(unique_strains, colors):
            mask = y_strain == strain
            axes[1].scatter(self.X_pca[mask, 0], self.X_pca[mask, 1], 
                          c=[color], label=strain, s=30, alpha=0.7,
                          edgecolors='white', linewidths=0.3)
        
        axes[1].set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)', 
                          fontsize=13, color='white')
        axes[1].set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)', 
                          fontsize=13, color='white')
        axes[1].set_title('PCA Projection (Colored by Strain)', fontsize=15, color='white', pad=15)
        axes[1].tick_params(colors='white', labelsize=10)
        axes[1].grid(True, alpha=0.3, color='gray')
        
        legend = axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8,
                               facecolor='black', edgecolor='white', labelcolor='white')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pca_analysis' / 'pca_projections.png', 
                   dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()

    def run_complete_analysis(self):
        """Run all enhanced analyses"""
        print("\n" + "="*70)
        print("STARTING ENHANCED ANALYSIS PIPELINE")
        print("="*70)
        
        # 1. PCA Analysis
        print("\n[1/3] PCA Analysis...")
        self.perform_pca_analysis()
        
        # 2. All Temporal Trajectories
        print("\n[2/3] Temporal Trajectories for All Cages...")
        self.plot_all_temporal_trajectories()
        
        # 3. Multi-Level UMAP Comparisons
        print("\n[3/3] Multi-Level UMAP Comparisons...")
        # You can customize which levels to compare
        embedding_levels = ['level_3_pooled', 'level_4_pooled', 'level_5_pooled', 'combined']
        self.plot_multi_level_umaps_by_strain(embedding_levels=embedding_levels)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print(f"Results saved to: {self.output_dir}")
        print("\nOutput directories:")
        print(f"  - {self.output_dir}/temporal_trajectories/")
        print(f"  - {self.output_dir}/multi_level_umaps/")
        print(f"  - {self.output_dir}/pca_analysis/")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced PCA and UMAP Analysis with Temporal Trajectories'
    )
    parser.add_argument('--embeddings', required=True, 
                       help='Path to embeddings directory')
    parser.add_argument('--metadata', required=True, 
                       help='Path to metadata CSV')
    parser.add_argument('--aggregation', required=True, 
                       help='Aggregation level (e.g., "4weeks")')
    parser.add_argument('--embedding-level', default='combined', 
                       help='Embedding level to use for main analysis')
    parser.add_argument('--output-dir', default='enhanced_analysis_results', 
                       help='Output directory')
    parser.add_argument('--comparison-levels', nargs='+',
                       default=['level_3_pooled', 'level_4_pooled', 'level_5_pooled', 'combined'],
                       help='Embedding levels to compare in multi-level UMAP plots')
    
    args = parser.parse_args()
    
    analyzer = EnhancedEmbeddingAnalyzer(
        embeddings_dir=args.embeddings,
        metadata_path=args.metadata,
        aggregation=args.aggregation,
        embedding_level=args.embedding_level,
        output_dir=args.output_dir
    )
    
    # You can also run individual analyses:
    # analyzer.perform_pca_analysis()
    # analyzer.plot_all_temporal_trajectories()
    # analyzer.plot_multi_level_umaps_by_strain(embedding_levels=args.comparison_levels)
    
    # Or run everything:
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()





















# #!/usr/bin/env python3
# """
# PCA and UMAP Analysis of Behavioral Embeddings with Temporal Tracking
# - Visualize daily trajectory in embedding space
# - PCA analysis with component importance
# - Identify which embedding dimensions contribute to PC1
# - Use PCA features for prediction tasks
# """

# import argparse
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import LeaveOneGroupOut
# from sklearn.linear_model import Ridge, LinearRegression
# from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score, f1_score
# import umap
# import zarr
# from tqdm import tqdm
# import json

# plt.style.use('dark_background')

# class EmbeddingPCAAnalyzer:
#     """Analyze embeddings using PCA and temporal visualization"""
    
#     def __init__(self, embeddings_dir, metadata_path, aggregation, embedding_level, 
#                  output_dir="pca_analysis_results"):
#         self.embeddings_dir = Path(embeddings_dir)
#         self.metadata_path = metadata_path
#         self.aggregation = aggregation
#         self.embedding_level = embedding_level
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(exist_ok=True, parents=True)
        
#         # Load data
#         self.load_data()
        
#     def load_data(self):
#         """Load embeddings and metadata"""
#         print("Loading data...")
        
#         core_emb_dir = self.embeddings_dir / self.aggregation / 'core_embeddings'
#         if not core_emb_dir.exists():
#             raise FileNotFoundError(f"Embeddings directory not found: {core_emb_dir}")
        
#         # Load metadata
#         self.metadata = pd.read_csv(self.metadata_path)
#         # take the first 50 entries for testing
#         self.metadata = self.metadata.head(50)  
#         # take onyl those entries for which the column nb_tpt_activity_data is equal to 40320
#         self.metadata = self.metadata[self.metadata['nb_tpt_activity_data'] == 40320]
        
#         self.metadata['from_tpt'] = pd.to_datetime(self.metadata['from_tpt'])
#         self.metadata['key'] = self.metadata.apply(
#             lambda row: f"{row['cage_id']}_{row['from_tpt'].strftime('%Y-%m-%d')}", axis=1
#         )
        
#         # Load embeddings with temporal information
#         self.embeddings_dict = {}
#         self.temporal_embeddings_dict = {}
        
#         for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), 
#                           desc="Loading embeddings"):
#             key = row['key']
#             zarr_path = core_emb_dir / f"{key}.zarr"
            
#             if zarr_path.exists():
#                 try:
#                     store = zarr.open(str(zarr_path), mode='r')
                    
#                     if self.embedding_level not in store:
#                         level_to_use = 'combined'
#                     else:
#                         level_to_use = self.embedding_level
                    
#                     seq_embeddings = store[level_to_use][:]
                    
#                     # Store both temporal and aggregated versions
#                     if seq_embeddings.ndim == 2:
#                         self.temporal_embeddings_dict[key] = seq_embeddings
#                         daily_embedding = seq_embeddings.mean(axis=0)
#                     else:
#                         daily_embedding = seq_embeddings
                    
#                     if not np.isnan(daily_embedding).any():
#                         self.embeddings_dict[key] = daily_embedding
                        
#                 except Exception as e:
#                     print(f"[WARN] Failed to load {zarr_path}: {e}")
        
#         # Create aligned dataset
#         self.create_aligned_dataset()
        
#         print(f"Loaded {self.X.shape[0]} samples with {self.X.shape[1]} dimensions")
#         print(f"Temporal embeddings available for {len(self.temporal_embeddings_dict)} samples")
    
#     def create_aligned_dataset(self):
#         """Create aligned embeddings and metadata"""
#         df_with_embeddings = self.metadata[self.metadata['key'].isin(self.embeddings_dict.keys())].copy()
        
#         aligned_embeddings = []
#         aligned_metadata = []
        
#         for _, row in df_with_embeddings.iterrows():
#             key = row['key']
#             embedding = self.embeddings_dict[key]
            
#             # Check for required fields
#             if pd.notna(row.get('avg_age_days_chunk_start')) and pd.notna(row.get('strain')):
#                 aligned_embeddings.append(embedding)
#                 aligned_metadata.append(row)
        
#         self.X = np.array(aligned_embeddings)
#         self.metadata_aligned = pd.DataFrame(aligned_metadata).reset_index(drop=True)
    
#     def perform_pca_analysis(self):
#         """Perform comprehensive PCA analysis"""
#         print("\n" + "="*70)
#         print("PERFORMING PCA ANALYSIS")
#         print("="*70)
        
#         # Standardize data
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(self.X)
        
#         # Fit PCA with all components
#         pca = PCA()
#         X_pca = pca.fit_transform(X_scaled)
        
#         self.pca = pca
#         self.scaler = scaler
#         self.X_pca = X_pca
        
#         # Analyze variance explained
#         self.analyze_variance_explained()
        
#         # Analyze PC1 contributions
#         self.analyze_pc1_contributions()
        
#         # Plot PCA projections
#         self.plot_pca_projections()
        
#         return X_pca
    
#     def analyze_variance_explained(self):
#         """Analyze and plot variance explained by each PC"""
#         print("\nVariance explained by principal components:")
        
#         variance_ratios = self.pca.explained_variance_ratio_
#         cumulative_variance = np.cumsum(variance_ratios)
        
#         # Print top 10 components
#         for i in range(min(10, len(variance_ratios))):
#             print(f"  PC{i+1}: {variance_ratios[i]*100:.2f}% "
#                   f"(Cumulative: {cumulative_variance[i]*100:.2f}%)")
        
#         # Find number of components for 90% variance
#         n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
#         print(f"\nComponents needed for 90% variance: {n_components_90}")
        
#         # Plot variance explained
#         fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
#         # Individual variance
#         n_plot = min(50, len(variance_ratios))
#         axes[0].bar(range(1, n_plot + 1), variance_ratios[:n_plot], alpha=0.7, color='cyan')
#         axes[0].set_xlabel('Principal Component', fontsize=12)
#         axes[0].set_ylabel('Variance Explained', fontsize=12)
#         axes[0].set_title('Individual Variance Explained', fontsize=14)
#         axes[0].grid(True, alpha=0.3)
        
#         # Cumulative variance
#         axes[1].plot(range(1, n_plot + 1), cumulative_variance[:n_plot], 
#                     'o-', color='orange', linewidth=2, markersize=4)
#         axes[1].axhline(y=0.9, color='red', linestyle='--', label='90% threshold')
#         axes[1].axvline(x=n_components_90, color='red', linestyle='--', alpha=0.5)
#         axes[1].set_xlabel('Number of Components', fontsize=12)
#         axes[1].set_ylabel('Cumulative Variance Explained', fontsize=12)
#         axes[1].set_title('Cumulative Variance Explained', fontsize=14)
#         axes[1].legend()
#         axes[1].grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig(self.output_dir / 'pca_variance_explained.png', dpi=300, bbox_inches='tight')
#         plt.savefig(self.output_dir / 'pca_variance_explained.pdf', bbox_inches='tight')
#         plt.close()
        
#         # Save variance data
#         variance_data = {
#             'variance_ratios': variance_ratios.tolist(),
#             'cumulative_variance': cumulative_variance.tolist(),
#             'n_components_90': int(n_components_90)
#         }
        
#         with open(self.output_dir / 'pca_variance_data.json', 'w') as f:
#             json.dump(variance_data, f, indent=2)
    
#     def analyze_pc1_contributions(self):
#         """Analyze which embedding dimensions contribute most to PC1"""
#         print("\nAnalyzing PC1 contributions...")
        
#         # Get loadings for PC1
#         pc1_loadings = self.pca.components_[0]
        
#         # Get top contributing dimensions
#         abs_loadings = np.abs(pc1_loadings)
#         top_indices = np.argsort(abs_loadings)[::-1][:20]
        
#         print("\nTop 20 dimensions contributing to PC1:")
#         for i, idx in enumerate(top_indices):
#             print(f"  {i+1}. Dimension {idx}: {pc1_loadings[idx]:.4f} "
#                   f"(|loading|: {abs_loadings[idx]:.4f})")
        
#         # Plot loading distribution
#         fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
#         # All loadings
#         axes[0].bar(range(len(pc1_loadings)), pc1_loadings, alpha=0.7, color='cyan')
#         axes[0].set_xlabel('Embedding Dimension', fontsize=12)
#         axes[0].set_ylabel('PC1 Loading', fontsize=12)
#         axes[0].set_title('PC1 Loadings Across All Dimensions', fontsize=14)
#         axes[0].grid(True, alpha=0.3)
        
#         # Top contributors
#         top_dims = top_indices[:30]
#         top_vals = pc1_loadings[top_dims]
#         colors = ['green' if x > 0 else 'red' for x in top_vals]
        
#         axes[1].barh(range(len(top_dims)), top_vals, color=colors, alpha=0.7)
#         axes[1].set_yticks(range(len(top_dims)))
#         axes[1].set_yticklabels([f'Dim {i}' for i in top_dims], fontsize=8)
#         axes[1].set_xlabel('PC1 Loading', fontsize=12)
#         axes[1].set_title('Top 30 Contributing Dimensions to PC1', fontsize=14)
#         axes[1].grid(True, alpha=0.3, axis='x')
#         axes[1].invert_yaxis()
        
#         plt.tight_layout()
#         plt.savefig(self.output_dir / 'pc1_contributions.png', dpi=300, bbox_inches='tight')
#         plt.savefig(self.output_dir / 'pc1_contributions.pdf', bbox_inches='tight')
#         plt.close()
        
#         # Save PC1 loading data
#         pc1_data = {
#             'all_loadings': pc1_loadings.tolist(),
#             'top_20_dimensions': top_indices[:20].tolist(),
#             'top_20_loadings': [float(pc1_loadings[i]) for i in top_indices[:20]]
#         }
        
#         with open(self.output_dir / 'pc1_loadings.json', 'w') as f:
#             json.dump(pc1_data, f, indent=2)
    
#     def plot_pca_projections(self):
#         """Plot PCA projections colored by age and strain"""
#         print("\nCreating PCA projection plots...")
        
#         y_age = self.metadata_aligned['avg_age_days_chunk_start'].values
#         y_strain = self.metadata_aligned['strain'].values
        
#         # Plot PC1 vs PC2 colored by age
#         fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
#         # Colored by age
#         scatter1 = axes[0].scatter(self.X_pca[:, 0], self.X_pca[:, 1], 
#                                    c=y_age, cmap='viridis', s=20, alpha=0.6)
#         axes[0].set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
#         axes[0].set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
#         axes[0].set_title('PCA Projection (Colored by Age)', fontsize=14)
#         cbar1 = plt.colorbar(scatter1, ax=axes[0])
#         cbar1.set_label('Age (days)', fontsize=12)
#         axes[0].grid(True, alpha=0.3)
        
#         # Colored by strain
#         unique_strains = np.unique(y_strain)
#         colors = plt.cm.tab20(np.linspace(0, 1, len(unique_strains)))
        
#         for strain, color in zip(unique_strains, colors):
#             mask = y_strain == strain
#             axes[1].scatter(self.X_pca[mask, 0], self.X_pca[mask, 1], 
#                           c=[color], label=strain, s=20, alpha=0.6)
        
#         axes[1].set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
#         axes[1].set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
#         axes[1].set_title('PCA Projection (Colored by Strain)', fontsize=14)
#         axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
#         axes[1].grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig(self.output_dir / 'pca_projections.png', dpi=300, bbox_inches='tight')
#         plt.savefig(self.output_dir / 'pca_projections.pdf', bbox_inches='tight')
#         plt.close()
    
#     def plot_temporal_trajectories(self):
#         """Plot per-day temporal trajectories in embedding space"""
#         print("\n" + "="*70)
#         print("PLOTTING TEMPORAL TRAJECTORIES")
#         print("="*70)
        
#         if not self.temporal_embeddings_dict:
#             print("No temporal embeddings available")
#             return
        
#         # Select a few representative samples
#         cage_ids = self.metadata_aligned['cage_id'].unique()
#         n_samples = min(5, len(cage_ids))
#         sample_cages = np.random.choice(cage_ids, n_samples, replace=False)
        
#         # Create UMAP embedding for temporal data
#         print("Computing UMAP for temporal trajectories...")
        
#         # Collect all temporal frames
#         all_temporal_frames = []
#         frame_metadata = []
        
#         for _, row in self.metadata_aligned.iterrows():
#             key = row['key']
#             if key in self.temporal_embeddings_dict:
#                 temporal_seq = self.temporal_embeddings_dict[key]
#                 for frame_idx, frame_emb in enumerate(temporal_seq):
#                     all_temporal_frames.append(frame_emb)
#                     frame_metadata.append({
#                         'key': key,
#                         'cage_id': row['cage_id'],
#                         'strain': row['strain'],
#                         'age': row['avg_age_days_chunk_start'],
#                         'frame_idx': frame_idx
#                     })
        
#         X_temporal = np.array(all_temporal_frames)
        
#         # Standardize and compute UMAP
#         X_temporal_scaled = StandardScaler().fit_transform(X_temporal)
#         reducer = umap.UMAP(n_neighbors=30, min_dist=0.1)
#         X_umap = reducer.fit_transform(X_temporal_scaled)
        
#         # Plot trajectories for sample cages
#         fig, axes = plt.subplots(2, 3, figsize=(18, 12))
#         axes = axes.flatten()
        
#         for idx, cage_id in tqdm(enumerate(sample_cages), total=n_samples, desc="Plotting trajectories"):
#             ax = axes[idx]
            
#             # Plot all points in gray
#             ax.scatter(X_umap[:, 0], X_umap[:, 1], c='gray', s=1, alpha=0.1)
            
#             # Highlight this cage's trajectory
#             cage_mask = [m['cage_id'] == cage_id for m in frame_metadata]
#             cage_trajectory = X_umap[cage_mask]
            
#             # Color by time
#             n_frames = len(cage_trajectory)
#             colors = plt.cm.viridis(np.linspace(0, 1, n_frames))
            
#             ax.scatter(cage_trajectory[:, 0], cage_trajectory[:, 1], 
#                       c=colors, s=10, alpha=0.7)
#             ax.plot(cage_trajectory[:, 0], cage_trajectory[:, 1], 
#                    'white', alpha=0.3, linewidth=0.5)
            
#             # Mark start and end
#             ax.scatter(cage_trajectory[0, 0], cage_trajectory[0, 1], 
#                       c='green', s=100, marker='*', edgecolors='white', linewidths=1)
#             ax.scatter(cage_trajectory[-1, 0], cage_trajectory[-1, 1], 
#                       c='red', s=100, marker='X', edgecolors='white', linewidths=1)
            
#             cage_info = [m for m in frame_metadata if m['cage_id'] == cage_id][0]
#             ax.set_title(f"Cage {cage_id} | Strain: {cage_info['strain']}\n"
#                         f"Age: {cage_info['age']:.1f} days | Frames: {n_frames}", 
#                         fontsize=10)
#             ax.set_xlabel('UMAP-1')
#             ax.set_ylabel('UMAP-2')
#             ax.grid(True, alpha=0.3)
        
#         # Hide extra subplot
#         axes[-1].axis('off')
        
#         plt.suptitle('Temporal Trajectories in Embedding Space (Per-Day Progression)', 
#                     fontsize=16, y=0.995)
#         plt.tight_layout()
#         plt.savefig(self.output_dir / 'temporal_trajectories_umap.png', dpi=300, bbox_inches='tight')
#         plt.savefig(self.output_dir / 'temporal_trajectories_umap.pdf', bbox_inches='tight')
#         plt.close()
        
#         print(f"Plotted trajectories for {n_samples} sample cages")
    
#     def predict_with_pca_features(self, n_components_list=[10, 20, 50, 100]):
#         """Use PCA features for prediction tasks"""
#         print("\n" + "="*70)
#         print("PREDICTION WITH PCA FEATURES")
#         print("="*70)
        
#         y_age = self.metadata_aligned['avg_age_days_chunk_start'].values
#         y_strain = self.metadata_aligned['strain'].values
#         groups = self.metadata_aligned['strain'].values
        
#         results = {
#             'age_regression': {},
#             'strain_classification': {}
#         }
        
#         # Also test with original embeddings
#         n_components_list = ['original'] + list(n_components_list)
        
#         for n_comp in n_components_list:
#             print(f"\nTesting with {n_comp} components...")
            
#             if n_comp == 'original':
#                 X_features = self.X
#                 feature_name = 'Original Embeddings'
#             else:
#                 X_features = self.X_pca[:, :n_comp]
#                 feature_name = f'{n_comp} PCs'
            
#             # Age regression
#             print(f"  Age regression with {feature_name}...")
#             age_results = self.evaluate_age_regression(X_features, y_age, groups)
#             results['age_regression'][str(n_comp)] = age_results
            
#             # Strain classification
#             print(f"  Strain classification with {feature_name}...")
#             strain_results = self.evaluate_strain_classification(X_features, y_strain)
#             results['strain_classification'][str(n_comp)] = strain_results
        
#         # Plot comparison
#         self.plot_pca_prediction_comparison(results)
        
#         # Save results
#         with open(self.output_dir / 'pca_prediction_results.json', 'w') as f:
#             json.dump(results, f, indent=2)
        
#         return results
    
#     def evaluate_age_regression(self, X, y, groups):
#         """Evaluate age regression with LOGO CV"""
#         logo = LeaveOneGroupOut()
#         model = Ridge(alpha=1.0)
        
#         rmse_scores = []
#         r2_scores = []
        
#         for train_idx, test_idx in logo.split(X, y, groups):
#             X_train, X_test = X[train_idx], X[test_idx]
#             y_train, y_test = y[train_idx], y[test_idx]
            
#             scaler = StandardScaler()
#             X_train_scaled = scaler.fit_transform(X_train)
#             X_test_scaled = scaler.transform(X_test)
            
#             model.fit(X_train_scaled, y_train)
#             y_pred = model.predict(X_test_scaled)
            
#             rmse_scores.append(root_mean_squared_error(y_test, y_pred))
#             r2_scores.append(r2_score(y_test, y_pred))
        
#         return {
#             'mean_rmse': float(np.mean(rmse_scores)),
#             'std_rmse': float(np.std(rmse_scores)),
#             'mean_r2': float(np.mean(r2_scores)),
#             'std_r2': float(np.std(r2_scores))
#         }
    
#     def evaluate_strain_classification(self, X, y):
#         """Evaluate strain classification"""
#         from sklearn.model_selection import train_test_split
#         from sklearn.preprocessing import LabelEncoder
        
#         # Encode labels
#         le = LabelEncoder()
#         y_encoded = le.fit_transform(y)
        
#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
#         )
        
#         # Scale and train
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
        
#         model = LinearRegression(n_jobs=-1)
#         model.fit(X_train_scaled, y_train)
        
#         y_pred = model.predict(X_test_scaled)
        
#         return {
#             'accuracy': float(accuracy_score(y_test, y_pred)),
#             'f1_weighted': float(f1_score(y_test, y_pred, average='weighted'))
#         }
    
#     def plot_pca_prediction_comparison(self, results):
#         """Plot comparison of predictions with different numbers of PCs"""
#         print("\nCreating PCA prediction comparison plots...")
        
#         fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
#         # Extract data
#         n_components = list(results['age_regression'].keys())
#         age_rmse = [results['age_regression'][n]['mean_rmse'] for n in n_components]
#         age_r2 = [results['age_regression'][n]['mean_r2'] for n in n_components]
#         strain_acc = [results['strain_classification'][n]['accuracy'] for n in n_components]
#         strain_f1 = [results['strain_classification'][n]['f1_weighted'] for n in n_components]
        
#         # Age RMSE
#         axes[0, 0].plot(range(len(n_components)), age_rmse, 'o-', linewidth=2, markersize=8)
#         axes[0, 0].set_xticks(range(len(n_components)))
#         axes[0, 0].set_xticklabels(n_components, rotation=45)
#         axes[0, 0].set_ylabel('RMSE (days)', fontsize=12)
#         axes[0, 0].set_title('Age Regression: RMSE', fontsize=14)
#         axes[0, 0].grid(True, alpha=0.3)
        
#         # Age R²
#         axes[0, 1].plot(range(len(n_components)), age_r2, 'o-', linewidth=2, markersize=8, color='orange')
#         axes[0, 1].set_xticks(range(len(n_components)))
#         axes[0, 1].set_xticklabels(n_components, rotation=45)
#         axes[0, 1].set_ylabel('R² Score', fontsize=12)
#         axes[0, 1].set_title('Age Regression: R²', fontsize=14)
#         axes[0, 1].grid(True, alpha=0.3)
        
#         # Strain Accuracy
#         axes[1, 0].plot(range(len(n_components)), strain_acc, 'o-', linewidth=2, markersize=8, color='green')
#         axes[1, 0].set_xticks(range(len(n_components)))
#         axes[1, 0].set_xticklabels(n_components, rotation=45)
#         axes[1, 0].set_ylabel('Accuracy', fontsize=12)
#         axes[1, 0].set_title('Strain Classification: Accuracy', fontsize=14)
#         axes[1, 0].grid(True, alpha=0.3)
        
#         # Strain F1
#         axes[1, 1].plot(range(len(n_components)), strain_f1, 'o-', linewidth=2, markersize=8, color='red')
#         axes[1, 1].set_xticks(range(len(n_components)))
#         axes[1, 1].set_xticklabels(n_components, rotation=45)
#         axes[1, 1].set_ylabel('F1 Score (weighted)', fontsize=12)
#         axes[1, 1].set_title('Strain Classification: F1', fontsize=14)
#         axes[1, 1].grid(True, alpha=0.3)
        
#         plt.suptitle('Prediction Performance with Different Numbers of Principal Components', 
#                     fontsize=16, y=0.995)
#         plt.tight_layout()
#         plt.savefig(self.output_dir / 'pca_prediction_comparison.png', dpi=300, bbox_inches='tight')
#         plt.savefig(self.output_dir / 'pca_prediction_comparison.pdf', bbox_inches='tight')
#         plt.close()
    
#     def run_complete_analysis(self):
#         """Run all analyses"""
#         print("\n" + "="*70)
#         print("STARTING COMPLETE PCA ANALYSIS PIPELINE")
#         print("="*70)
        
#         # 1. PCA Analysis
#         self.perform_pca_analysis()
        
#         # 2. Temporal Trajectories
#         self.plot_temporal_trajectories()
        
#         # 3. Prediction with PCA features
#         self.predict_with_pca_features()
        
#         print("\n" + "="*70)
#         print("ANALYSIS COMPLETE!")
#         print(f"Results saved to: {self.output_dir}")
#         print("="*70)


# def main():
#     parser = argparse.ArgumentParser(description='PCA Analysis of Behavioral Embeddings')
#     parser.add_argument('--embeddings', required=True, help='Path to embeddings directory')
#     parser.add_argument('--metadata', required=True, help='Path to metadata CSV')
#     parser.add_argument('--aggregation', required=True, help='Aggregation level (e.g., "4weeks")')
#     parser.add_argument('--embedding-level', default='combined', 
#                        help='Embedding level to use')
#     parser.add_argument('--output-dir', default='pca_analysis_results', 
#                        help='Output directory')
    
#     args = parser.parse_args()
    
#     analyzer = EmbeddingPCAAnalyzer(
#         embeddings_dir=args.embeddings,
#         metadata_path=args.metadata,
#         aggregation=args.aggregation,
#         embedding_level=args.embedding_level,
#         output_dir=args.output_dir
#     )
    
#     analyzer.run_complete_analysis()


# if __name__ == "__main__":
#     main()
    
    
# # python pca_temporal_analysis.py \
# #   --embeddings /scratch/bhole/dvc_data/smoothed/models_4weeks/embeddings \
# #   --metadata /scratch/bhole/dvc_data/smoothed/40320/summary_metadata_40320.csv \
# #   --aggregation 4weeks \
# #   --embedding-level level_6_pooled \
# #   --output-dir pca_analysis_4weeks