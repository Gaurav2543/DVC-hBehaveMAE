#!/usr/bin/env python3
"""
CPU-based biological interpretability analyses for hBehaveMAE embeddings.
Implements visualization and statistical analyses that don't require GPU computation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm
import json
import imageio.v2 as imageio
import glob
from collections import defaultdict
import time
import datetime

# ML libraries
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Specialized libraries
import umap.umap_ as umap
from pygam import LinearGAM, s
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

class BiologicalInterpretabilityAnalyzer:
    def __init__(self, 
                 embeddings_dir: str,
                 labels_path: str,
                 summary_csv: str,
                 output_dir: str = "biological_interpretability_cpu"):
        
        print("🧬 Initializing Biological Interpretability Analyzer...")
        print(f"📁 Embeddings directory: {embeddings_dir}")
        print(f"📊 Labels path: {labels_path}")
        print(f"📋 Summary CSV: {summary_csv}")
        print(f"💾 Output directory: {output_dir}")
        
        self.embeddings_dir = Path(embeddings_dir)
        self.labels_path = labels_path
        self.summary_csv = summary_csv
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = ["umap_visualizations", "clustering_analysis", "age_trajectories", 
                  "phylogenetic_analysis", "animations", "strain_analysis"]
        print("📂 Creating output subdirectories...")
        for subdir in tqdm(subdirs, desc="Creating directories"):
            (self.output_dir / subdir).mkdir(exist_ok=True)
            time.sleep(0.1)  # Small delay for visual progress
        
        self.load_data()
        
    def load_data(self):
        """Load embeddings, labels, and metadata"""
        print("\n🔄 Loading data...")
        
        # Load embeddings
        print("📊 Loading embeddings...")
        self.embeddings = {}
        embedding_levels = ['low', 'mid', 'high', 'comb']
        
        for level in tqdm(embedding_levels, desc="Loading embedding levels"):
            emb_path = self.embeddings_dir / f"test_{level}.npy"
            if emb_path.exists():
                print(f"   Loading {level} embeddings from {emb_path}")
                data = np.load(emb_path, allow_pickle=True)
                if isinstance(data, np.ndarray) and data.size == 1:
                    data = data.item()
                self.embeddings[level] = data['embeddings'].astype(np.float32)
                print(f"   ✅ Loaded {level} embeddings: {self.embeddings[level].shape}")
            else:
                print(f"   ⚠️  {level} embeddings not found at {emb_path}")
        
        if not self.embeddings:
            raise FileNotFoundError("No embedding files found!")
        
        # Load labels
        print(f"🏷️  Loading labels from {self.labels_path}")
        if os.path.exists(self.labels_path):
            labels_data = np.load(self.labels_path, allow_pickle=True)
            if isinstance(labels_data, np.ndarray) and labels_data.size == 1:
                labels_data = labels_data.item()
            self.label_array = labels_data['label_array']
            self.vocabulary = labels_data['vocabulary']
            print(f"   ✅ Loaded labels: {len(self.vocabulary)} vocabulary terms")
            print(f"   📝 Vocabulary: {self.vocabulary}")
        else:
            print(f"   ⚠️  Labels file not found, creating dummy labels")
            # Create dummy labels based on embedding size
            first_emb = list(self.embeddings.values())[0]
            self.vocabulary = ['Age_Days', 'strain', 'cage', 'body_weight']
            self.label_array = np.random.randint(0, 100, (len(self.vocabulary), first_emb.shape[0]))
        
        # Load metadata CSV
        print(f"📋 Loading metadata from {self.summary_csv}")
        if os.path.exists(self.summary_csv):
            self.metadata_df = pd.read_csv(self.summary_csv)
            print(f"   ✅ Loaded metadata: {self.metadata_df.shape}")
            print(f"   📊 Columns: {list(self.metadata_df.columns)}")
            
            # Filter for test set (sets == 1)
            test_mask = self.metadata_df['sets'] == 1
            self.test_metadata = self.metadata_df[test_mask].copy().reset_index(drop=True)
            print(f"   🧪 Test samples: {len(self.test_metadata)} (sets == 1)")
            
            # Show sample of test data
            print("   📋 Sample test metadata:")
            print(self.test_metadata.head()[['day', 'cage', 'strain', 'Age_Days']].to_string())
        else:
            raise FileNotFoundError(f"Summary CSV not found: {self.summary_csv}")
        
        # Align data lengths
        min_length = min(
            min(emb.shape[0] for emb in self.embeddings.values()),
            len(self.test_metadata)
        )
        print(f"🔧 Aligning data to minimum length: {min_length}")
        
        # Truncate embeddings and metadata to match
        for level in self.embeddings:
            if self.embeddings[level].shape[0] > min_length:
                self.embeddings[level] = self.embeddings[level][:min_length]
                print(f"   Truncated {level} embeddings to {self.embeddings[level].shape}")
        
        if len(self.test_metadata) > min_length:
            self.test_metadata = self.test_metadata.iloc[:min_length].copy()
            print(f"   Truncated metadata to {len(self.test_metadata)} rows")
        
        print(f"✅ Data loading completed!")
        print(f"   📊 Final embedding shapes: {[(k, v.shape) for k, v in self.embeddings.items()]}")
        print(f"   📋 Final metadata shape: {self.test_metadata.shape}")
    
    def preprocess_embeddings(self, embeddings: np.ndarray, level_name: str) -> np.ndarray:
        """Preprocess embeddings for analysis"""
        print(f"🔧 Preprocessing {level_name} embeddings...")
        original_shape = embeddings.shape
        
        embeddings = embeddings.astype(np.float32)
        
        # Check for inf/nan values
        inf_count = np.isinf(embeddings).sum()
        nan_count = np.isnan(embeddings).sum()
        print(f"   🔍 Found {inf_count} inf values, {nan_count} nan values")
        
        embeddings = np.where(np.isinf(embeddings), np.nan, embeddings)
        
        if nan_count > 0 or inf_count > 0:
            imputer = SimpleImputer(strategy='mean')
            embeddings = imputer.fit_transform(embeddings)
            print(f"   🔧 Applied mean imputation")
        
        embeddings = np.clip(embeddings, -65504, 65504)
        print(f"   ✅ Preprocessing complete: {original_shape} -> {embeddings.shape}")
        
        return embeddings
    
    def plot_umap_visualization(self, embeddings_level: str = 'comb'):
        """Generate UMAP visualizations colored by different metadata"""
        print(f"\n🗺️  Generating UMAP visualizations for {embeddings_level} level...")
        
        if embeddings_level not in self.embeddings:
            print(f"   ❌ {embeddings_level} level not available")
            return
        
        embeddings = self.preprocess_embeddings(self.embeddings[embeddings_level], embeddings_level)
        
        # Reduce dimensionality with PCA first if needed
        print("🔍 Checking if PCA dimensionality reduction is needed...")
        if embeddings.shape[1] > 50:
            print(f"   📉 Reducing from {embeddings.shape[1]} to 50 dimensions with PCA")
            pca = PCA(n_components=50)
            embeddings_pca = pca.fit_transform(embeddings)
            explained_var = pca.explained_variance_ratio_.sum()
            print(f"   📊 PCA explained variance: {explained_var:.3f}")
        else:
            embeddings_pca = embeddings
            print(f"   ✅ Using original {embeddings.shape[1]} dimensions")
        
        # Fit UMAP
        print("🗺️  Fitting UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, verbose=True)
        embedding_2d = reducer.fit_transform(embeddings_pca)
        print(f"   ✅ UMAP completed: {embedding_2d.shape}")
        
        # Metadata columns to visualize
        metadata_cols = ['Age_Days', 'strain', 'cage']
        available_cols = [col for col in metadata_cols if col in self.test_metadata.columns]
        print(f"📊 Available metadata columns: {available_cols}")
        
        # Create subplot for each metadata column
        n_cols = len(available_cols)
        if n_cols == 0:
            print("   ⚠️  No metadata columns available for visualization")
            return
        
        n_rows = (n_cols + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(15, 6 * n_rows))
        if n_cols == 1:
            axes = [axes] if n_rows == 1 else axes.ravel()
        else:
            axes = axes.ravel() if n_rows > 1 else [axes[0], axes[1]]
        
        print("🎨 Creating visualizations...")
        for i, col in enumerate(tqdm(available_cols, desc="Plotting metadata")):
            ax = axes[i]
            values = self.test_metadata[col].values[:len(embedding_2d)]
            
            if pd.api.types.is_numeric_dtype(values):
                print(f"   📈 Plotting numeric column: {col}")
                scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                                   c=values, cmap='viridis', alpha=0.6, s=10)
                plt.colorbar(scatter, ax=ax)
                ax.set_title(f'{embeddings_level.upper()} - Colored by {col}\n'
                           f'Range: {values.min():.1f} - {values.max():.1f}')
            else:
                # Categorical data
                print(f"   🏷️  Plotting categorical column: {col}")
                unique_vals = np.unique(values)
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_vals)))
                
                for j, val in enumerate(unique_vals):
                    mask = values == val
                    ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                             c=[colors[j]], label=str(val), alpha=0.6, s=10)
                
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                ax.set_title(f'{embeddings_level.upper()} - Colored by {col}\n'
                           f'Categories: {len(unique_vals)}')
            
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
        
        # Hide unused subplots
        for i in range(len(available_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        save_path = self.output_dir / "umap_visualizations" / f"umap_{embeddings_level}_metadata.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved: {save_path}")
    
    def clustering_analysis(self, embeddings_level: str = 'comb', n_clusters: int = 8):
        """Perform clustering analysis and characterize behavioral phenotypes"""
        print(f"\n🔬 Performing clustering analysis for {embeddings_level} level...")
        
        if embeddings_level not in self.embeddings:
            print(f"   ❌ {embeddings_level} level not available")
            return None, None
        
        embeddings = self.preprocess_embeddings(self.embeddings[embeddings_level], embeddings_level)
        
        # Perform clustering
        print(f"🎯 Running K-means clustering with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        with tqdm(total=100, desc="K-means fitting") as pbar:
            cluster_labels = kmeans.fit_predict(embeddings)
            pbar.update(100)
        
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        print(f"   📊 Silhouette score: {silhouette_avg:.3f}")
        
        # Show cluster sizes
        unique, counts = np.unique(cluster_labels, return_counts=True)
        print(f"   📈 Cluster sizes:")
        for cluster_id, count in zip(unique, counts):
            print(f"      Cluster {cluster_id}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")
        
        # Characterize clusters
        print("🔍 Characterizing clusters...")
        results = self._characterize_clusters(cluster_labels, embeddings_level)
        
        # Save results
        results_path = self.output_dir / "clustering_analysis" / f"cluster_results_{embeddings_level}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"   💾 Saved cluster results: {results_path}")
        
        # Plot cluster phenotypes
        self._plot_cluster_phenotypes(cluster_labels, embeddings_level)
        
        # Plot cluster metadata
        self._plot_cluster_metadata(cluster_labels, embeddings_level)
        
        return cluster_labels, results
    
    def _characterize_clusters(self, labels: np.ndarray, embeddings_level: str) -> Dict:
        """Characterize clusters by computing phenotype summaries"""
        print("   📊 Computing cluster characteristics...")
        results = {}
        
        # Numeric columns for analysis
        numeric_cols = ['Age_Days']
        available_numeric = [col for col in numeric_cols if col in self.test_metadata.columns]
        
        # Get cluster data
        cluster_data = self.test_metadata.iloc[:len(labels)].copy()
        cluster_data['cluster'] = labels
        
        for cluster_id in tqdm(np.unique(labels), desc="Characterizing clusters"):
            mask = labels == cluster_id
            cluster_subset = cluster_data[mask]
            
            result = {
                'size': int(mask.sum()),
                'percentage': float(mask.mean() * 100)
            }
            
            # Numeric phenotypes
            for col in available_numeric:
                if len(cluster_subset[col].dropna()) > 0:
                    result[f'{col}_mean'] = float(cluster_subset[col].mean())
                    result[f'{col}_std'] = float(cluster_subset[col].std())
                    result[f'{col}_min'] = float(cluster_subset[col].min())
                    result[f'{col}_max'] = float(cluster_subset[col].max())
            
            # Categorical distributions
            categorical_cols = ['strain', 'cage']
            for col in categorical_cols:
                if col in cluster_subset.columns:
                    dist = cluster_subset[col].value_counts().to_dict()
                    result[f'{col}_distribution'] = {str(k): int(v) for k, v in dist.items()}
                    result[f'{col}_most_common'] = str(cluster_subset[col].mode().iloc[0] if len(cluster_subset[col].mode()) > 0 else 'N/A')
            
            results[f'cluster_{cluster_id}'] = result
            
        return results
    
    def _plot_cluster_phenotypes(self, labels: np.ndarray, embeddings_level: str):
        """Plot phenotype means per cluster"""
        print("   📊 Plotting cluster phenotypes...")
        
        numeric_cols = ['Age_Days']
        available_cols = [col for col in numeric_cols if col in self.test_metadata.columns]
        
        if not available_cols:
            print("      ⚠️  No numeric columns available for phenotype plotting")
            return
        
        cluster_data = self.test_metadata.iloc[:len(labels)].copy()
        cluster_data['cluster'] = labels
        
        means = cluster_data.groupby('cluster')[available_cols].mean()
        stds = cluster_data.groupby('cluster')[available_cols].std()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        means.plot(kind='bar', yerr=stds, ax=ax, capsize=4)
        ax.set_title(f'Phenotype Means per Cluster ({embeddings_level} level)')
        ax.set_ylabel('Mean Value')
        ax.set_xlabel('Cluster')
        ax.tick_params(axis='x', rotation=0)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / "clustering_analysis" / f"cluster_phenotypes_{embeddings_level}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      💾 Saved: {save_path}")
    
    def _plot_cluster_metadata(self, labels: np.ndarray, embeddings_level: str):
        """Plot metadata distributions per cluster"""
        print("   📊 Plotting cluster metadata distributions...")
        
        cluster_data = self.test_metadata.iloc[:len(labels)].copy()
        cluster_data['cluster'] = labels
        
        metadata_cols = ['Age_Days', 'strain', 'cage']
        available_cols = [col for col in metadata_cols if col in cluster_data.columns]
        
        if not available_cols:
            print("      ⚠️  No metadata columns available")
            return
        
        n_cols = len(available_cols)
        n_rows = (n_cols + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4 * n_rows))
        
        if n_cols == 1:
            axes = [axes] if n_rows == 1 else axes.ravel()
        else:
            axes = axes.ravel() if n_rows > 1 else [axes[0], axes[1]]
        
        for i, col in enumerate(tqdm(available_cols, desc="Plotting metadata distributions")):
            ax = axes[i]
            
            if pd.api.types.is_numeric_dtype(cluster_data[col]):
                sns.boxplot(data=cluster_data, x='cluster', y=col, ax=ax)
                ax.set_title(f'{col} Distribution by Cluster')
            else:
                # Stacked bar for categorical
                ct = pd.crosstab(cluster_data['cluster'], cluster_data[col], normalize='index')
                ct.plot(kind='bar', stacked=True, ax=ax)
                ax.set_title(f'{col} Distribution by Cluster (Normalized)')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                ax.tick_params(axis='x', rotation=0)
        
        # Hide unused subplots
        for i in range(len(available_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        save_path = self.output_dir / "clustering_analysis" / f"cluster_metadata_{embeddings_level}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      💾 Saved: {save_path}")
    
    def age_trajectory_analysis(self, embeddings_level: str = 'comb'):
        """Analyze age trajectories in embedding space"""
        print(f"\n📈 Analyzing age trajectories for {embeddings_level} level...")
        
        if embeddings_level not in self.embeddings:
            print(f"   ❌ {embeddings_level} level not available")
            return
        
        if 'Age_Days' not in self.test_metadata.columns:
            print("   ⚠️  Age_Days not found in metadata")
            return
        
        embeddings = self.preprocess_embeddings(self.embeddings[embeddings_level], embeddings_level)
        ages = self.test_metadata['Age_Days'].values[:len(embeddings)]
        strains = self.test_metadata['strain'].values[:len(embeddings)] if 'strain' in self.test_metadata.columns else None
        
        print(f"   📊 Age range: {ages.min():.1f} - {ages.max():.1f} days")
        if strains is not None:
            unique_strains = np.unique(strains)
            print(f"   🧬 Strains: {len(unique_strains)} unique ({list(unique_strains[:5])}{'...' if len(unique_strains) > 5 else ''})")
        
        # Fit UMAP for 2D visualization
        print("🗺️  Fitting UMAP for trajectory visualization...")
        if embeddings.shape[1] > 50:
            pca = PCA(n_components=50)
            embeddings_pca = pca.fit_transform(embeddings)
        else:
            embeddings_pca = embeddings
        
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings_pca)
        
        # Plot age trajectory
        print("🎨 Creating age trajectory plot...")
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=ages,
                            cmap='plasma', alpha=0.6, s=15)
        plt.colorbar(scatter, label='Age (Days)')
        
        # Fit trajectories per strain if available
        if strains is not None:
            unique_strains = np.unique(strains)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_strains)))
            
            print("   🧬 Fitting strain trajectories...")
            for strain, color in zip(tqdm(unique_strains, desc="Processing strains"), colors):
                strain_mask = strains == strain
                if np.sum(strain_mask) > 10:  # Only if sufficient samples
                    X = embedding_2d[strain_mask]
                    y = ages[strain_mask]
                    
                    # Sort by age for trajectory
                    sort_idx = np.argsort(y)
                    plt.plot(X[sort_idx, 0], X[sort_idx, 1],
                            color=color, linewidth=2, alpha=0.8, label=f'{strain}')
        
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title(f'Age Trajectories in {embeddings_level.upper()} Embedding Space')
        if strains is not None:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = self.output_dir / "age_trajectories" / f"age_trajectory_{embeddings_level}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved: {save_path}")
        
        # GAM analysis for individual dimensions
        self._gam_dimension_analysis(embeddings, ages, embeddings_level)
    
    def _gam_dimension_analysis(self, embeddings: np.ndarray, ages: np.ndarray, embeddings_level: str):
        """Perform GAM analysis on individual embedding dimensions"""
        print("   📈 Performing GAM analysis on embedding dimensions...")
        
        n_dims_to_analyze = min(10, embeddings.shape[1])  # Analyze first 10 dimensions
        print(f"      Analyzing {n_dims_to_analyze} dimensions")
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.ravel()
        
        gam_results = {}
        
        for dim in tqdm(range(n_dims_to_analyze), desc="GAM fitting"):
            ax = axes[dim]
            
            try:
                # Fit GAM
                gam = LinearGAM(s(0)).fit(ages.reshape(-1, 1), embeddings[:, dim])
                age_range = np.linspace(ages.min(), ages.max(), 200)
                predictions = gam.predict(age_range)
                
                # Calculate R-squared equivalent
                y_pred = gam.predict(ages.reshape(-1, 1))
                ss_res = np.sum((embeddings[:, dim] - y_pred) ** 2)
                ss_tot = np.sum((embeddings[:, dim] - np.mean(embeddings[:, dim])) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                gam_results[f'dim_{dim}'] = {
                    'r_squared': float(r_squared),
                    'deviance': float(gam.statistics_['deviance']),
                    'n_splines': int(gam.terms[0].n_splines)
                }
                
                ax.plot(age_range, predictions, 'r-', linewidth=2, label=f'GAM fit (R²={r_squared:.3f})')
                ax.scatter(ages, embeddings[:, dim], alpha=0.3, s=5)
                ax.set_xlabel('Age (days)')
                ax.set_ylabel(f'Dim {dim}')
                ax.set_title(f'Age Trajectory - Dimension {dim}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"         ⚠️  GAM failed for dimension {dim}: {str(e)[:50]}")
                ax.text(0.5, 0.5, f'GAM failed:\n{str(e)[:50]}...', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'Dimension {dim} - Failed')
        
        plt.tight_layout()
        save_path = self.output_dir / "age_trajectories" / f"gam_dimensions_{embeddings_level}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      💾 Saved: {save_path}")
        
        # Save GAM results
        gam_results_path = self.output_dir / "age_trajectories" / f"gam_results_{embeddings_level}.json"
        with open(gam_results_path, 'w') as f:
            json.dump(gam_results, f, indent=2)
        print(f"      💾 Saved GAM results: {gam_results_path}")
    
    def strain_specific_analysis(self, embeddings_level: str = 'comb'):
        """Analyze strain-specific patterns"""
        print(f"\n🧬 Performing strain-specific analysis for {embeddings_level} level...")
        
        if embeddings_level not in self.embeddings:
            print(f"   ❌ {embeddings_level} level not available")
            return
        
        if 'strain' not in self.test_metadata.columns:
            print("   ⚠️  Strain information not found in metadata")
            return
        
        embeddings = self.preprocess_embeddings(self.embeddings[embeddings_level], embeddings_level)
        strains = self.test_metadata['strain'].values[:len(embeddings)]
        ages = self.test_metadata['Age_Days'].values[:len(embeddings)] if 'Age_Days' in self.test_metadata.columns else None
        
        unique_strains = np.unique(strains)
        print(f"   🧬 Found {len(unique_strains)} unique strains")
        
        # Strain silhouette analysis
        print("   📊 Computing strain separation quality...")
        le = LabelEncoder()
        strain_encoded = le.fit_transform(strains)
        silhouette_strain = silhouette_score(embeddings, strain_encoded)
        print(f"      Strain silhouette score: {silhouette_strain:.3f}")
        
        # Per-strain statistics
        strain_stats = {}
        print("   📈 Computing per-strain statistics...")
        
        for strain in tqdm(unique_strains, desc="Processing strains"):
            strain_mask = strains == strain
            strain_embeddings = embeddings[strain_mask]
            
            stats = {
                'n_samples': int(strain_mask.sum()),
                'embedding_mean': strain_embeddings.mean(axis=0).tolist(),
                'embedding_std': strain_embeddings.std(axis=0).tolist(),
            }
            
            if ages is not None:
                strain_ages = ages[strain_mask]
                stats.update({
                    'age_mean': float(strain_ages.mean()),
                    'age_std': float(strain_ages.std()),
                    'age_range': [float(strain_ages.min()), float(strain_ages.max())]
                })
            
            strain_stats[strain] = stats
        
        # Save strain statistics
        strain_stats_path = self.output_dir / "strain_analysis" / f"strain_stats_{embeddings_level}.json"
        with open(strain_stats_path, 'w') as f:
            json.dump(strain_stats, f, indent=2)
        print(f"   💾 Saved strain statistics: {strain_stats_path}")
        
        # Plot strain comparison
        self._plot_strain_comparison(embeddings, strains, ages, embeddings_level)
        
        return strain_stats
    
    def _plot_strain_comparison(self, embeddings, strains, ages, embeddings_level):
        """Plot strain comparison visualizations"""
        print("   🎨 Creating strain comparison plots...")
        
        unique_strains = np.unique(strains)
        
        # UMAP colored by strain
        if embeddings.shape[1] > 50:
            pca = PCA(n_components=50)
            embeddings_pca = pca.fit_transform(embeddings)
        else:
            embeddings_pca = embeddings
        
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings_pca)
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Strain UMAP
        ax1 = axes[0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_strains)))
        for i, strain in enumerate(unique_strains):
            mask = strains == strain
            ax1.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                       c=[colors[i]], label=strain, alpha=0.6, s=15)
        
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.set_title(f'Strain Clustering ({embeddings_level.upper()})')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Age distribution by strain
        if ages is not None:
            ax2 = axes[1]
            strain_age_data = []
            strain_labels = []
            
            for strain in unique_strains:
                mask = strains == strain
                strain_ages = ages[mask]
                strain_age_data.append(strain_ages)
                strain_labels.append(strain)
            
            ax2.boxplot(strain_age_data, labels=strain_labels)
            ax2.set_xlabel('Strain')
            ax2.set_ylabel('Age (Days)')
            ax2.set_title('Age Distribution by Strain')
            ax2.tick_params(axis='x', rotation=45)
        else:
            axes[1].text(0.5, 0.5, 'Age data not available', 
                        transform=axes[1].transAxes, ha='center', va='center')
            axes[1].set_title('Age Analysis - No Data')
        
        plt.tight_layout()
        save_path = self.output_dir / "strain_analysis" / f"strain_comparison_{embeddings_level}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      💾 Saved: {save_path}")
    
    def create_animated_trajectory(self, embeddings_level: str = 'comb'):
        """Create animated GIF of age trajectories"""
        print(f"\n🎬 Creating animated trajectory for {embeddings_level} level...")
        
        if embeddings_level not in self.embeddings:
            print(f"   ❌ {embeddings_level} level not available")
            return
        
        if 'Age_Days' not in self.test_metadata.columns:
            print("   ⚠️  Age_Days not found in metadata")
            return
        
        embeddings = self.preprocess_embeddings(self.embeddings[embeddings_level], embeddings_level)
        ages = self.test_metadata['Age_Days'].values[:len(embeddings)]
        
        # Fit UMAP
        print("   🗺️  Fitting UMAP for animation...")
        if embeddings.shape[1] > 50:
            pca = PCA(n_components=50)
            embeddings_pca = pca.fit_transform(embeddings)
        else:
            embeddings_pca = embeddings
        
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings_pca)
        
        # Create frames for different age ranges
        n_frames = 20
        age_ranges = np.linspace(ages.min(), ages.max(), n_frames)
        frames_dir = self.output_dir / "animations" / "frames" / embeddings_level
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"   🎞️  Creating {n_frames} animation frames...")
        
        for i, age_threshold in enumerate(tqdm(age_ranges, desc="Creating frames")):
            plt.figure(figsize=(10, 8))
            
            # Plot all points in gray
            plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                       c='lightgray', alpha=0.3, s=10, label='Future')
            
            # Highlight points up to current age threshold
            mask = ages <= age_threshold
            if np.sum(mask) > 0:
                plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                           c=ages[mask], cmap='plasma', s=15, alpha=0.8, label='Current')
            
            plt.colorbar(label='Age (Days)')
            plt.title(f'Age Progression in {embeddings_level.upper()} Space\n'
                     f'Age ≤ {age_threshold:.0f} days ({np.sum(mask)} samples)')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.legend()
            
            frame_path = frames_dir / f"frame_{i:03d}.png"
            plt.savefig(frame_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        # Create GIF
        print("   🎬 Assembling GIF...")
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        if frame_files:
            images = []
            for f in tqdm(frame_files, desc="Loading frames"):
                images.append(imageio.imread(str(f)))
            
            gif_path = self.output_dir / "animations" / f"age_trajectory_{embeddings_level}.gif"
            imageio.mimsave(str(gif_path), images, fps=2)
            print(f"   🎬 Saved animated GIF: {gif_path}")
            
            # Clean up frame files
            print("   🧹 Cleaning up temporary frames...")
            for f in frame_files:
                f.unlink()
            frames_dir.rmdir()
        else:
            print("   ❌ No frames created for GIF")
    
    def run_all_analyses(self):
        """Run all CPU-based interpretability analyses"""
        start_time = time.time()
        print("\n" + "="*60)
        print("🧬 STARTING COMPREHENSIVE BIOLOGICAL INTERPRETABILITY ANALYSIS")
        print("="*60)
        
        # Analysis for each embedding level
        levels_to_analyze = ['low', 'mid', 'high', 'comb']
        available_levels = [level for level in levels_to_analyze if level in self.embeddings]
        
        print(f"📊 Available embedding levels: {available_levels}")
        
        analysis_results = {}
        
        for level in available_levels:
            level_start = time.time()
            print(f"\n{'='*50}")
            print(f"🔍 ANALYZING {level.upper()} LEVEL EMBEDDINGS")
            print(f"{'='*50}")
            
            level_results = {}
            
            # UMAP visualization
            try:
                print(f"🗺️  [1/6] UMAP Visualization...")
                self.plot_umap_visualization(level)
                level_results['umap'] = True
            except Exception as e:
                print(f"   ❌ UMAP failed: {e}")
                level_results['umap'] = False
            
            # Clustering analysis
            try:
                print(f"🔬 [2/6] Clustering Analysis...")
                cluster_labels, cluster_results = self.clustering_analysis(level)
                level_results['clustering'] = cluster_results is not None
            except Exception as e:
                print(f"   ❌ Clustering failed: {e}")
                level_results['clustering'] = False
            
            # Age trajectory analysis
            try:
                print(f"📈 [3/6] Age Trajectory Analysis...")
                if 'Age_Days' in self.test_metadata.columns:
                    self.age_trajectory_analysis(level)
                    level_results['age_trajectories'] = True
                else:
                    print("   ⚠️  Skipped - Age_Days not available")
                    level_results['age_trajectories'] = False
            except Exception as e:
                print(f"   ❌ Age trajectory failed: {e}")
                level_results['age_trajectories'] = False
            
            # Strain analysis
            try:
                print(f"🧬 [4/6] Strain-Specific Analysis...")
                strain_stats = self.strain_specific_analysis(level)
                level_results['strain_analysis'] = strain_stats is not None
            except Exception as e:
                print(f"   ❌ Strain analysis failed: {e}")
                level_results['strain_analysis'] = False
            
            # Animated trajectory
            try:
                print(f"🎬 [5/6] Animated Trajectory...")
                if 'Age_Days' in self.test_metadata.columns:
                    self.create_animated_trajectory(level)
                    level_results['animation'] = True
                else:
                    print("   ⚠️  Skipped - Age_Days not available")
                    level_results['animation'] = False
            except Exception as e:
                print(f"   ❌ Animation failed: {e}")
                level_results['animation'] = False
            
            level_time = time.time() - level_start
            print(f"✅ {level.upper()} level completed in {level_time:.1f}s")
            
            level_results['processing_time'] = level_time
            analysis_results[level] = level_results
        
        # Save comprehensive results
        total_time = time.time() - start_time
        
        summary = {
            'timestamp': datetime.datetime.now().isoformat(),
            'total_processing_time': total_time,
            'embedding_levels_processed': available_levels,
            'data_info': {
                'n_test_samples': len(self.test_metadata),
                'embedding_shapes': {k: list(v.shape) for k, v in self.embeddings.items()},
                'available_metadata': list(self.test_metadata.columns),
                'age_range': [float(self.test_metadata['Age_Days'].min()), 
                             float(self.test_metadata['Age_Days'].max())] if 'Age_Days' in self.test_metadata.columns else None,
                'n_strains': len(self.test_metadata['strain'].unique()) if 'strain' in self.test_metadata.columns else 0
            },
            'analysis_results': analysis_results
        }
        
        summary_path = self.output_dir / "analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Final summary
        print(f"\n{'='*60}")
        print("📊 ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"📊 Levels processed: {len(available_levels)}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"💾 Summary saved: {summary_path}")
        
        success_count = sum(sum(level_results.values()) for level_results in analysis_results.values() if isinstance(level_results, dict))
        total_analyses = len(available_levels) * 5  # 5 analyses per level
        print(f"✅ Success rate: {success_count}/{total_analyses} analyses completed")
        
        print(f"\n🎉 CPU-BASED ANALYSIS COMPLETED!")
        return analysis_results

def main():
    # Configuration
    EMBEDDINGS_DIR = "/scratch/bhole/dvc_hbehave_others/extracted_embeddings/extracted_embeddings2_new_lz"
    LABELS_PATH = "../../dvc_project/hbehavemae/original/dvc-data/arrays_sub20_with_cage_complete_correct_strains.npy"
    SUMMARY_CSV = "../../dvc_project/summary_table_imputed_with_sets_sub_20_CompleteAge_Strains.csv"
    OUTPUT_DIR = "biological_interpretability_cpu"
    
    print("🚀 Starting Biological Interpretability Analysis")
    print(f"📅 Timestamp: {datetime.datetime.now()}")
    
    # Initialize analyzer
    try:
        analyzer = BiologicalInterpretabilityAnalyzer(
            embeddings_dir=EMBEDDINGS_DIR,
            labels_path=LABELS_PATH,
            summary_csv=SUMMARY_CSV,
            output_dir=OUTPUT_DIR
        )
        
        # Run all analyses
        results = analyzer.run_all_analyses()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())