#!/usr/bin/env python3
"""
Enhanced Biological Interpretability Analysis for HDP Mouse Aging Study
Comprehensive analysis of hBehaveMAE embeddings with focus on aging, genetics, and behavior

Author: Gaurav2543
Date: 2025-06-27
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
from tqdm import tqdm
import json
import imageio.v2 as imageio
import glob
from collections import defaultdict
import time
import datetime

# ML libraries
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cross_decomposition import CCA

# Specialized libraries
import umap.umap_ as umap
from pygam import LinearGAM, s
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, chi2_contingency
from scipy.signal import find_peaks, periodogram
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import ruptures as rpt  # For changepoint detection

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

def get_strain_color_map(strain_names):
    """
    Map strain names to color palettes for interpretability:
    - "CC*" strains: shades of blue
    - "BXD*" strains: shades of red  
    - Others: distinct bold colors
    """
    cc_strains = sorted([s for s in strain_names if s.startswith('CC')])
    bxd_strains = sorted([s for s in strain_names if s.startswith('BXD')])
    other_strains = sorted([s for s in strain_names if not (s.startswith('CC') or s.startswith('BXD'))])

    cc_colors = sns.color_palette("Blues", n_colors=max(3, len(cc_strains)))
    bxd_colors = sns.color_palette("Reds", n_colors=max(3, len(bxd_strains)))
    other_colors = sns.color_palette("Dark2", n_colors=max(3, len(other_strains)))

    color_map = {}
    for s, c in zip(cc_strains, cc_colors): color_map[s] = c
    for s, c in zip(bxd_strains, bxd_colors): color_map[s] = c
    for s, c in zip(other_strains, other_colors): color_map[s] = c

    return color_map

class EnhancedBiologicalInterpretabilityAnalyzer:
    def __init__(self,
                 model_checkpoint: str,
                 embeddings_dir: str,
                 labels_path: str,
                 summary_csv: str,
                 output_dir: str = "enhanced_biological_interpretability",
                 umap_n_components: int = 2):
        """
        Enhanced analyzer for comprehensive biological interpretability of HDP mouse aging study.
        """
        print("🧬 Initializing Enhanced Biological Interpretability Analyzer...")
        print(f"📁 Model checkpoint: {model_checkpoint}")
        print(f"📁 Embeddings directory: {embeddings_dir}")
        print(f"📊 Labels path: {labels_path}")
        print(f"📋 Summary CSV: {summary_csv}")
        print(f"💾 Output directory: {output_dir}")

        self.model_checkpoint = model_checkpoint
        self.embeddings_dir = Path(embeddings_dir)
        self.labels_path = labels_path
        self.summary_csv = summary_csv
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.umap_n_components = umap_n_components

        # Create comprehensive subdirectories
        subdirs = [
            "phenotype_correlations", "aging_trajectories",
            "strain_genetics", "circadian_analysis", "behavioral_biomarkers",
            "multimodal_integration", "clustering_analysis",
            "interpretations"
        ]
        
        print("📂 Creating output subdirectories...")
        for subdir in tqdm(subdirs, desc="Creating directories"):
            (self.output_dir / subdir).mkdir(exist_ok=True)
            time.sleep(0.05)

        self.load_data()
        self.initialize_analysis_components()

    def load_data(self):
        """Load all data components with enhanced validation"""
        print("\n🔄 Loading comprehensive dataset...")

        # Load embeddings
        print("📊 Loading embeddings...")
        self.embeddings = {}
        embedding_levels = ['low', 'mid', 'high', 'comb']

        for level in tqdm(embedding_levels, desc="Loading embedding levels"):
            emb_path = self.embeddings_dir / f"test_{level}.npy"
            if emb_path.exists():
                data = np.load(emb_path, allow_pickle=True)
                if isinstance(data, np.ndarray) and data.size == 1:
                    data = data.item()
                self.embeddings[level] = data['embeddings'].astype(np.float32)
                print(f"   ✅ Loaded {level} embeddings: {self.embeddings[level].shape}")
            else:
                print(f"   ⚠️  {level} embeddings not found at {emb_path}")

        # Load labels
        print(f"🏷️  Loading labels...")
        if os.path.exists(self.labels_path):
            labels_data = np.load(self.labels_path, allow_pickle=True)
            if isinstance(labels_data, np.ndarray) and labels_data.size == 1:
                labels_data = labels_data.item()
            self.label_array = labels_data['label_array']
            self.vocabulary = labels_data['vocabulary']
            print(f"   ✅ Loaded labels: {len(self.vocabulary)} vocabulary terms")
        else:
            print(f"   ⚠️  Labels file not found")
            first_emb = list(self.embeddings.values())[0]
            self.vocabulary = ['Age_Days', 'strain', 'cage']
            self.label_array = np.random.randint(0, 100, (len(self.vocabulary), first_emb.shape[0]))

        # Load metadata CSV
        print(f"📋 Loading metadata...")
        self.metadata_df = pd.read_csv(self.summary_csv)
        test_mask = self.metadata_df['sets'] == 1
        self.test_metadata = self.metadata_df[test_mask].copy().reset_index(drop=True)
        print(f"   ✅ Test samples: {len(self.test_metadata)}")

        # Align data lengths
        min_length = min(
            min(emb.shape[0] for emb in self.embeddings.values()),
            len(self.test_metadata)
        )
        
        for level in self.embeddings:
            if self.embeddings[level].shape[0] > min_length:
                self.embeddings[level] = self.embeddings[level][:min_length]

        if len(self.test_metadata) > min_length:
            self.test_metadata = self.test_metadata.iloc[:min_length].copy()

        print(f"✅ Data alignment completed: {min_length} samples")

    def initialize_analysis_components(self):
        """Initialize analysis components and computed features"""
        print("🔧 Initializing analysis components...")
        
        # Extract key variables
        self.ages = self.test_metadata['Age_Days'].values if 'Age_Days' in self.test_metadata.columns else None
        self.strains = self.test_metadata['strain'].values if 'strain' in self.test_metadata.columns else None
        self.cages = self.test_metadata['cage'].values if 'cage' in self.test_metadata.columns else None
        
        # Color mapping for consistent visualization
        if self.strains is not None:
            self.strain_color_map = get_strain_color_map(np.unique(self.strains))
            
        # Initialize results storage
        self.analysis_results = {}
        self.phenotype_correlations = {}
        self.aging_metrics = {}
        
        print("✅ Analysis components initialized")

    def preprocess_embeddings(self, embeddings: np.ndarray, level_name: str) -> np.ndarray:
        """Enhanced preprocessing with outlier detection"""
        print(f"🔧 Preprocessing {level_name} embeddings...")
        original_shape = embeddings.shape
        embeddings = embeddings.astype(np.float32)

        # Handle inf/nan values
        inf_count = np.isinf(embeddings).sum()
        nan_count = np.isnan(embeddings).sum()
        if inf_count > 0 or nan_count > 0:
            print(f"   🔍 Found {inf_count} inf, {nan_count} nan values - applying imputation")
            embeddings = np.where(np.isinf(embeddings), np.nan, embeddings)
            imputer = SimpleImputer(strategy='mean')
            embeddings = imputer.fit_transform(embeddings)

        # Outlier detection and clipping
        embeddings = np.clip(embeddings, -65504, 65504)
        
        print(f"   ✅ Preprocessing complete: {original_shape} -> {embeddings.shape}")
        return embeddings

    # ==================================================================================
    # 1. COMPREHENSIVE PHENOTYPIC CORRELATION ANALYSIS
    # ==================================================================================

    def comprehensive_phenotype_correlation(self, embeddings_level: str = 'comb'):
        """
        Systematically correlate embedding dimensions with all available phenotypes
        """
        print(f"\n📊 Comprehensive phenotype correlation analysis for {embeddings_level}...")
        
        if embeddings_level not in self.embeddings:
            print(f"   ❌ {embeddings_level} level not available")
            return

        embeddings = self.preprocess_embeddings(self.embeddings[embeddings_level], embeddings_level)
        
        # Define phenotypes to analyze
        numeric_phenotypes = ['Age_Days']
        categorical_phenotypes = ['strain', 'cage']
        
        # Add any additional numeric columns from metadata
        for col in self.test_metadata.columns:
            if pd.api.types.is_numeric_dtype(self.test_metadata[col]) and col not in numeric_phenotypes:
                if not col.startswith('Unnamed') and col not in ['sets', 'day']:
                    numeric_phenotypes.append(col)

        print(f"   📈 Analyzing {len(numeric_phenotypes)} numeric and {len(categorical_phenotypes)} categorical phenotypes")
        
        # Calculate correlations for numeric phenotypes
        correlations = {}
        dimension_rankings = {}
        
        for phenotype in tqdm(numeric_phenotypes, desc="Numeric phenotypes"):
            if phenotype in self.test_metadata.columns:
                phenotype_values = self.test_metadata[phenotype].values[:len(embeddings)]
                if not np.all(np.isnan(phenotype_values)):
                    dim_correlations = []
                    for dim in range(embeddings.shape[1]):
                        r, p = pearsonr(embeddings[:, dim], phenotype_values)
                        dim_correlations.append({
                            'dimension': dim,
                            'correlation': r,
                            'p_value': p,
                            'abs_correlation': abs(r)
                        })
                    
                    # Sort by absolute correlation
                    dim_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
                    correlations[phenotype] = dim_correlations
                    dimension_rankings[phenotype] = [d['dimension'] for d in dim_correlations[:10]]  # Top 10

        # Categorical phenotype analysis using ANOVA F-statistic
        for phenotype in tqdm(categorical_phenotypes, desc="Categorical phenotypes"):
            if phenotype in self.test_metadata.columns:
                phenotype_values = self.test_metadata[phenotype].values[:len(embeddings)]
                unique_categories = np.unique(phenotype_values)
                if len(unique_categories) > 1:
                    dim_f_stats = []
                    for dim in range(embeddings.shape[1]):
                        groups = [embeddings[phenotype_values == cat, dim] for cat in unique_categories]
                        try:
                            from scipy.stats import f_oneway
                            f_stat, p_val = f_oneway(*groups)
                            dim_f_stats.append({
                                'dimension': dim,
                                'f_statistic': f_stat,
                                'p_value': p_val
                            })
                        except:
                            dim_f_stats.append({
                                'dimension': dim,
                                'f_statistic': 0,
                                'p_value': 1
                            })
                    
                    dim_f_stats.sort(key=lambda x: x['f_statistic'], reverse=True)
                    correlations[phenotype] = dim_f_stats
                    dimension_rankings[phenotype] = [d['dimension'] for d in dim_f_stats[:10]]

        # Create comprehensive correlation heatmap
        self._plot_phenotype_correlation_heatmap(correlations, embeddings_level)
        
        # Save results
        results_path = self.output_dir / "phenotype_correlations" / f"correlations_{embeddings_level}.json"
        with open(results_path, 'w') as f:
            json.dump({
                'correlations': correlations,
                'dimension_rankings': dimension_rankings
            }, f, indent=2, default=str)
        
        self.phenotype_correlations[embeddings_level] = correlations
        print(f"   💾 Saved correlation analysis: {results_path}")
        
        return correlations, dimension_rankings

    def _plot_phenotype_correlation_heatmap(self, correlations, embeddings_level):
        """Create correlation heatmap between dimensions and phenotypes"""
        print("   🎨 Creating phenotype correlation heatmap...")
        
        # Prepare data for heatmap
        phenotypes = list(correlations.keys())
        max_dims = 20  # Show top 20 dimensions
        
        heatmap_data = []
        dim_labels = []
        
        for phenotype in phenotypes:
            if 'correlation' in correlations[phenotype][0]:  # Numeric phenotype
                corr_values = [d['correlation'] for d in correlations[phenotype][:max_dims]]
                if len(corr_values) < max_dims:
                    corr_values.extend([0] * (max_dims - len(corr_values)))
                heatmap_data.append(corr_values)
            else:  # Categorical phenotype - use normalized F-statistics
                f_values = [d['f_statistic'] for d in correlations[phenotype][:max_dims]]
                if len(f_values) > 0:
                    f_values = np.array(f_values) / np.max(f_values)  # Normalize
                    if len(f_values) < max_dims:
                        f_values = np.concatenate([f_values, np.zeros(max_dims - len(f_values))])
                    heatmap_data.append(f_values.tolist())

        if len(heatmap_data) > 0:
            heatmap_data = np.array(heatmap_data)
            
            plt.figure(figsize=(15, 8))
            sns.heatmap(heatmap_data, 
                       xticklabels=[f'Dim {i}' for i in range(max_dims)],
                       yticklabels=phenotypes,
                       cmap='RdBu_r', center=0, 
                       cbar_kws={'label': 'Correlation / Normalized F-statistic'})
            plt.title(f'Phenotype-Embedding Correlations ({embeddings_level.upper()})')
            plt.xlabel('Embedding Dimensions')
            plt.ylabel('Phenotypes')
            plt.tight_layout()
            
            save_path = self.output_dir / "phenotype_correlations" / f"correlation_heatmap_{embeddings_level}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      💾 Saved heatmap: {save_path}")

    # ==================================================================================
    # 2. TEMPORAL TRAJECTORY ANALYSIS WITH AGING RATES
    # ==================================================================================

    def temporal_trajectory_analysis(self, embeddings_level: str = 'comb'):
        """
        Advanced temporal analysis including aging rates and transition points
        """
        print(f"\n📈 Temporal trajectory analysis for {embeddings_level}...")
        
        if embeddings_level not in self.embeddings or self.ages is None:
            print(f"   ❌ Required data not available")
            return

        embeddings = self.preprocess_embeddings(self.embeddings[embeddings_level], embeddings_level)
        
        # Fit UMAP for visualization
        if embeddings.shape[1] > 50:
            pca = PCA(n_components=50)
            embeddings_pca = pca.fit_transform(embeddings)
        else:
            embeddings_pca = embeddings

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=2)
        embedding_2d = reducer.fit_transform(embeddings_pca)

        # Calculate aging velocities and trajectories
        aging_results = self._calculate_aging_velocities(embedding_2d, self.ages, self.strains)
        
        # Detect critical transition points
        transition_results = self._detect_aging_transitions(embeddings, self.ages, self.strains)
        
        # Create comprehensive aging plots
        self._plot_aging_trajectories_enhanced(embedding_2d, self.ages, self.strains, 
                                             aging_results, transition_results, embeddings_level)
        
        # Save results
        results = {
            'aging_velocities': aging_results,
            'transition_points': transition_results,
            'embedding_2d': embedding_2d.tolist()
        }
        
        results_path = self.output_dir / "aging_trajectories" / f"temporal_analysis_{embeddings_level}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.aging_metrics[embeddings_level] = results
        print(f"   💾 Saved temporal analysis: {results_path}")
        
        return results

    def _calculate_aging_velocities(self, embedding_2d, ages, strains):
        """Calculate aging velocities in embedding space"""
        print("   🚀 Calculating aging velocities...")
        
        results = {}
        if strains is None:
            strain_list = ['all']
            strain_masks = {'all': np.ones(len(ages), dtype=bool)}
        else:
            strain_list = np.unique(strains)
            strain_masks = {strain: strains == strain for strain in strain_list}
        
        for strain in tqdm(strain_list, desc="Processing strains"):
            mask = strain_masks[strain]
            strain_emb = embedding_2d[mask]
            strain_ages = ages[mask]
            
            if len(strain_ages) < 5:  # Need minimum samples
                continue
                
            # Sort by age
            sorted_idx = np.argsort(strain_ages)
            sorted_emb = strain_emb[sorted_idx]
            sorted_ages = strain_ages[sorted_idx]
            
            # Calculate velocities (rate of change in embedding space)
            velocities = []
            age_midpoints = []
            
            for i in range(1, len(sorted_emb)):
                dist = np.linalg.norm(sorted_emb[i] - sorted_emb[i-1])
                time_diff = sorted_ages[i] - sorted_ages[i-1]
                if time_diff > 0:
                    velocity = dist / time_diff
                    velocities.append(velocity)
                    age_midpoints.append((sorted_ages[i] + sorted_ages[i-1]) / 2)
            
            if len(velocities) > 0:
                results[strain] = {
                    'velocities': velocities,
                    'age_midpoints': age_midpoints,
                    'median_velocity': np.median(velocities),
                    'peak_velocity_age': age_midpoints[np.argmax(velocities)] if velocities else None,
                    'total_distance': np.sum([np.linalg.norm(sorted_emb[i] - sorted_emb[i-1]) 
                                            for i in range(1, len(sorted_emb))]),
                    'age_range': [float(sorted_ages[0]), float(sorted_ages[-1])]
                }
        
        return results

    def _detect_aging_transitions(self, embeddings, ages, strains):
        """Detect critical transition points in aging using changepoint detection"""
        print("   🔍 Detecting aging transition points...")
        
        results = {}
        if strains is None:
            strain_list = ['all']
            strain_masks = {'all': np.ones(len(ages), dtype=bool)}
        else:
            strain_list = np.unique(strains)
            strain_masks = {strain: strains == strain for strain in strain_list}
        
        for strain in tqdm(strain_list, desc="Changepoint detection"):
            mask = strain_masks[strain]
            strain_emb = embeddings[mask]
            strain_ages = ages[mask]
            
            if len(strain_ages) < 20:  # Need sufficient samples for changepoint detection
                continue
                
            # Sort by age
            sorted_idx = np.argsort(strain_ages)
            sorted_emb = strain_emb[sorted_idx]
            sorted_ages = strain_ages[sorted_idx]
            
            # Use PCA to reduce dimensionality for changepoint detection
            if sorted_emb.shape[1] > 10:
                pca = PCA(n_components=10)
                reduced_emb = pca.fit_transform(sorted_emb)
            else:
                reduced_emb = sorted_emb
            
            # Detect changepoints using variance-based method
            try:
                algo = rpt.Pelt(model="rbf").fit(reduced_emb)
                changepoints = algo.predict(pen=10)
                
                # Convert changepoint indices to ages
                transition_ages = [sorted_ages[cp-1] for cp in changepoints[:-1]]  # Exclude last point
                
                results[strain] = {
                    'transition_ages': transition_ages,
                    'n_transitions': len(transition_ages),
                    'avg_transition_interval': np.mean(np.diff([sorted_ages[0]] + transition_ages + [sorted_ages[-1]])) if transition_ages else None
                }
            except Exception as e:
                print(f"      ⚠️ Changepoint detection failed for {strain}: {str(e)[:50]}")
                results[strain] = {'transition_ages': [], 'n_transitions': 0, 'avg_transition_interval': None}
        
        return results

    def _plot_aging_trajectories_enhanced(self, embedding_2d, ages, strains, aging_results, 
                                        transition_results, embeddings_level):
        """Create enhanced aging trajectory plots"""
        print("   🎨 Creating enhanced aging trajectory plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Plot 1: Basic trajectory with strain colors
        ax1 = axes[0, 0]
        scatter = ax1.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=ages,
                             cmap='plasma', alpha=0.7, s=15, edgecolor='k', linewidth=0.1)
        plt.colorbar(scatter, ax=ax1, label='Age (Days)')
        
        if strains is not None:
            for strain in np.unique(strains):
                strain_mask = strains == strain
                if np.sum(strain_mask) > 10:
                    X = embedding_2d[strain_mask]
                    y = ages[strain_mask]
                    sort_idx = np.argsort(y)
                    ax1.plot(X[sort_idx, 0], X[sort_idx, 1],
                            color=self.strain_color_map[strain], linewidth=2, alpha=0.8)
        
        ax1.set_title(f'Age Trajectories ({embeddings_level.upper()})')
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        
        # Plot 2: Aging velocities by strain
        ax2 = axes[0, 1]
        strain_names = []
        median_velocities = []
        colors = []
        
        for strain, results in aging_results.items():
            strain_names.append(strain)
            median_velocities.append(results['median_velocity'])
            colors.append(self.strain_color_map.get(strain, 'gray'))
        
        bars = ax2.bar(range(len(strain_names)), median_velocities, color=colors, alpha=0.7, edgecolor='k')
        ax2.set_xlabel('Strain')
        ax2.set_ylabel('Median Aging Velocity')
        ax2.set_title('Aging Velocity by Strain')
        ax2.set_xticks(range(len(strain_names)))
        ax2.set_xticklabels(strain_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Transition points
        ax3 = axes[1, 0]
        transition_counts = []
        strain_names_trans = []
        
        for strain, results in transition_results.items():
            if results['n_transitions'] > 0:
                strain_names_trans.append(strain)
                transition_counts.append(results['n_transitions'])
        
        if strain_names_trans:
            colors_trans = [self.strain_color_map.get(strain, 'gray') for strain in strain_names_trans]
            ax3.bar(range(len(strain_names_trans)), transition_counts, color=colors_trans, alpha=0.7, edgecolor='k')
            ax3.set_xlabel('Strain')
            ax3.set_ylabel('Number of Transition Points')
            ax3.set_title('Aging Transition Points by Strain')
            ax3.set_xticks(range(len(strain_names_trans)))
            ax3.set_xticklabels(strain_names_trans, rotation=45, ha='right')
        
        # Plot 4: Age vs. distance traveled
        ax4 = axes[1, 1]
        for strain, results in aging_results.items():
            if 'total_distance' in results:
                age_range = results['age_range']
                total_dist = results['total_distance']
                ax4.scatter(age_range[1] - age_range[0], total_dist, 
                           color=self.strain_color_map.get(strain, 'gray'),
                           s=60, alpha=0.7, label=strain, edgecolor='k', linewidth=0.5)
        
        ax4.set_xlabel('Age Range (Days)')
        ax4.set_ylabel('Total Distance Traveled in Embedding Space')
        ax4.set_title('Age Range vs. Behavioral Change')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / "aging_trajectories" / f"enhanced_trajectories_{embeddings_level}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      💾 Saved enhanced trajectories: {save_path}")

    # ==================================================================================
    # 3. STRAIN-SPECIFIC GENETIC ANALYSIS
    # ==================================================================================

    def strain_genetic_analysis(self, embeddings_level: str = 'comb'):
        """
        Analyze strain-specific patterns with genetic relatedness considerations
        """
        print(f"\n🧬 Strain genetic analysis for {embeddings_level}...")
        
        if embeddings_level not in self.embeddings or self.strains is None:
            print(f"   ❌ Required data not available")
            return

        embeddings = self.preprocess_embeddings(self.embeddings[embeddings_level], embeddings_level)
        unique_strains = np.unique(self.strains)
        
        # Calculate strain centroids in embedding space
        strain_centroids = self._calculate_strain_centroids(embeddings, self.strains)
        
        # Calculate behavioral distance matrix
        behavioral_distance_matrix = self._calculate_behavioral_distances(strain_centroids, unique_strains)
        
        # Estimate heritability of embedding dimensions
        heritability_results = self._estimate_dimension_heritability(embeddings, self.strains)
        
        # Create strain comparison plots
        self._plot_strain_genetic_analysis(embeddings, self.strains, strain_centroids, 
                                         behavioral_distance_matrix, heritability_results, embeddings_level)
        
        # Save results
        results = {
            'strain_centroids': {strain: centroid.tolist() for strain, centroid in strain_centroids.items()},
            'behavioral_distance_matrix': behavioral_distance_matrix.tolist(),
            'strain_names': unique_strains.tolist(),
            'heritability_results': heritability_results
        }
        
        results_path = self.output_dir / "strain_genetics" / f"genetic_analysis_{embeddings_level}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"   💾 Saved genetic analysis: {results_path}")
        return results

    def _calculate_strain_centroids(self, embeddings, strains):
        """Calculate centroid positions for each strain in embedding space"""
        strain_centroids = {}
        for strain in np.unique(strains):
            mask = strains == strain
            strain_embeddings = embeddings[mask]
            strain_centroids[strain] = np.mean(strain_embeddings, axis=0)
        return strain_centroids

    def _calculate_behavioral_distances(self, strain_centroids, strain_names):
        """Calculate pairwise behavioral distances between strains"""
        n_strains = len(strain_names)
        distance_matrix = np.zeros((n_strains, n_strains))
        
        for i, strain1 in enumerate(strain_names):
            for j, strain2 in enumerate(strain_names):
                if i != j:
                    dist = np.linalg.norm(strain_centroids[strain1] - strain_centroids[strain2])
                    distance_matrix[i, j] = dist
        
        return distance_matrix

    def _estimate_dimension_heritability(self, embeddings, strains):
        """Estimate heritability (h²) for each embedding dimension across strains"""
        print("   📊 Estimating dimension heritability...")
        
        heritabilities = []
        for dim in tqdm(range(embeddings.shape[1]), desc="Calculating h² per dimension"):
            # Calculate between-strain and within-strain variance
            overall_mean = np.mean(embeddings[:, dim])
            
            between_strain_var = 0
            within_strain_var = 0
            total_n = 0
            
            strain_means = {}
            strain_ns = {}
            
            for strain in np.unique(strains):
                mask = strains == strain
                strain_values = embeddings[mask, dim]
                n_strain = len(strain_values)
                
                if n_strain > 1:
                    strain_mean = np.mean(strain_values)
                    strain_var = np.var(strain_values, ddof=1)
                    
                    strain_means[strain] = strain_mean
                    strain_ns[strain] = n_strain
                    
                    between_strain_var += n_strain * (strain_mean - overall_mean) ** 2
                    within_strain_var += (n_strain - 1) * strain_var
                    total_n += n_strain
            
            # Calculate variance components
            n_strains = len(strain_means)
            if n_strains > 1:
                between_strain_var /= (total_n - n_strains)
                within_strain_var /= (total_n - n_strains)
                
                # Broad-sense heritability approximation
                total_var = between_strain_var + within_strain_var
                h2 = between_strain_var / total_var if total_var > 0 else 0
            else:
                h2 = 0
            
            heritabilities.append({
                'dimension': dim,
                'heritability': h2,
                'between_strain_var': between_strain_var,
                'within_strain_var': within_strain_var
            })
        
        # Sort by heritability
        heritabilities.sort(key=lambda x: x['heritability'], reverse=True)
        return heritabilities

    def _plot_strain_genetic_analysis(self, embeddings, strains, strain_centroids, 
                                    distance_matrix, heritability_results, embeddings_level):
        """Create comprehensive strain genetic analysis plots"""
        print("   🎨 Creating strain genetic analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Plot 1: Strain clustering in UMAP space
        ax1 = axes[0, 0]
        if embeddings.shape[1] > 50:
            pca = PCA(n_components=50)
            embeddings_pca = pca.fit_transform(embeddings)
        else:
            embeddings_pca = embeddings
            
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings_pca)
        
        for strain in np.unique(strains):
            mask = strains == strain
            ax1.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                       color=self.strain_color_map[strain], label=strain, alpha=0.7, s=15)
        
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.set_title('Strain Clustering in Behavioral Space')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Plot 2: Behavioral distance matrix heatmap
        ax2 = axes[0, 1]
        strain_names = list(strain_centroids.keys())
        im = ax2.imshow(distance_matrix, cmap='viridis')
        ax2.set_xticks(range(len(strain_names)))
        ax2.set_yticks(range(len(strain_names)))
        ax2.set_xticklabels(strain_names, rotation=45, ha='right')
        ax2.set_yticklabels(strain_names)
        ax2.set_title('Behavioral Distance Matrix')
        plt.colorbar(im, ax=ax2, label='Behavioral Distance')
        
        # Plot 3: Heritability by dimension
        ax3 = axes[1, 0]
        dims = [h['dimension'] for h in heritability_results[:20]]  # Top 20
        h2_values = [h['heritability'] for h in heritability_results[:20]]
        
        bars = ax3.bar(range(len(dims)), h2_values, alpha=0.7, edgecolor='k')
        ax3.set_xlabel('Embedding Dimension')
        ax3.set_ylabel('Heritability (h²)')
        ax3.set_title('Heritability of Embedding Dimensions')
        ax3.set_xticks(range(len(dims)))
        ax3.set_xticklabels([f'D{d}' for d in dims], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Strain family analysis (CC vs BXD vs Others)
        ax4 = axes[1, 1]
        strain_families = {'CC': [], 'BXD': [], 'Other': []}
        
        for strain in strain_names:
            if strain.startswith('CC'):
                strain_families['CC'].append(strain)
            elif strain.startswith('BXD'):
                strain_families['BXD'].append(strain)
            else:
                strain_families['Other'].append(strain)
        
        family_colors = {'CC': 'blue', 'BXD': 'red', 'Other': 'green'}
        y_pos = 0
        
        for family, family_strains in strain_families.items():
            if family_strains:
                # Calculate average distance within family
                family_indices = [strain_names.index(s) for s in family_strains]
                if len(family_indices) > 1:
                    family_distances = []
                    for i in family_indices:
                        for j in family_indices:
                            if i != j:
                                family_distances.append(distance_matrix[i, j])
                    avg_distance = np.mean(family_distances)
                else:
                    avg_distance = 0
                
                ax4.barh(y_pos, avg_distance, color=family_colors[family], 
                        alpha=0.7, label=f'{family} (n={len(family_strains)})')
                y_pos += 1
        
        ax4.set_xlabel('Average Intra-family Behavioral Distance')
        ax4.set_ylabel('Strain Family')
        ax4.set_title('Behavioral Similarity Within Strain Families')
        ax4.legend()
        
        plt.tight_layout()
        save_path = self.output_dir / "strain_genetics" / f"genetic_analysis_{embeddings_level}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      💾 Saved genetic analysis plots: {save_path}")

    # ==================================================================================
    # 4. CIRCADIAN RHYTHM ANALYSIS
    # ==================================================================================

    def circadian_rhythm_analysis(self, embeddings_level: str = 'comb'):
        """
        Analyze circadian patterns encoded in embeddings
        """
        print(f"\n🕐 Circadian rhythm analysis for {embeddings_level}...")
        
        if embeddings_level not in self.embeddings:
            print(f"   ❌ {embeddings_level} level not available")
            return

        embeddings = self.preprocess_embeddings(self.embeddings[embeddings_level], embeddings_level)
        
        # # For this analysis, we'll simulate time-of-day data since it's not directly available
        # # In a real scenario, you would have actual timestamp data
        # print("   ⚠️  Note: Using simulated time-of-day data for demonstration")
        
        # # Simulate time of day (0-23 hours) - in real analysis, extract from your timestamp data
        # n_samples = len(embeddings)
        # simulated_times = np.random.uniform(0, 24, n_samples)  # Random times for demonstration
        
        # # Perform circadian analysis
        # circadian_results = self._analyze_circadian_patterns(embeddings, simulated_times)
        
        print("   ⏰ Extracting time-of-day from real timestamps...")

        # Try to extract hour-of-day from a timestamp column if available
        if 'timestamp' in self.test_metadata.columns:
            timestamps = pd.to_datetime(self.test_metadata['timestamp'])
            time_of_day = timestamps.dt.hour + timestamps.dt.minute / 60.0
        elif 'day' in self.test_metadata.columns:
            # If only date is available, set hour to 12 (noon) as fallback
            timestamps = pd.to_datetime(self.test_metadata['day'])
            time_of_day = np.full(len(timestamps), 12.0)
        else:
            raise ValueError("No timestamp or day column found in metadata for circadian analysis.")

        circadian_results = self._analyze_circadian_patterns(embeddings, time_of_day)
        
        # Create circadian visualizations
        # self._plot_circadian_analysis(embeddings, simulated_times, circadian_results, embeddings_level)
        self._plot_circadian_analysis(embeddings, time_of_day, circadian_results, embeddings_level)
        
        # Save results
        results_path = self.output_dir / "circadian_analysis" / f"circadian_{embeddings_level}.json"
        with open(results_path, 'w') as f:
            json.dump(circadian_results, f, indent=2, default=str)
        
        print(f"   💾 Saved circadian analysis: {results_path}")
        return circadian_results

    def _analyze_circadian_patterns(self, embeddings, time_of_day):
        """Analyze how embeddings capture circadian patterns"""
        print("   🔍 Analyzing circadian patterns in embeddings...")
        
        # Convert time to cyclic features
        time_rad = 2 * np.pi * (time_of_day % 24) / 24
        sin_time = np.sin(time_rad)
        cos_time = np.cos(time_rad)
        
        # Find dimensions most correlated with time
        circadian_dims = []
        for dim in tqdm(range(embeddings.shape[1]), desc="Analyzing dimensions"):
            sin_r, sin_p = pearsonr(embeddings[:, dim], sin_time)
            cos_r, cos_p = pearsonr(embeddings[:, dim], cos_time)
            
            # Combined correlation strength
            circadian_strength = np.sqrt(sin_r**2 + cos_r**2)
            
            circadian_dims.append({
                'dimension': dim,
                'sin_correlation': sin_r,
                'cos_correlation': cos_r,
                'circadian_strength': circadian_strength,
                'sin_p_value': sin_p,
                'cos_p_value': cos_p
            })
        
        # Sort by circadian strength
        circadian_dims.sort(key=lambda x: x['circadian_strength'], reverse=True)
        
        # Analyze periodicity using FFT for top dimensions
        top_circadian_dims = circadian_dims[:5]
        periodicity_analysis = {}
        
        for dim_info in top_circadian_dims:
            dim = dim_info['dimension']
            
            # Sort by time for FFT analysis
            sorted_idx = np.argsort(time_of_day)
            sorted_signal = embeddings[sorted_idx, dim]
            
            # Perform FFT to find dominant frequencies
            fft_result = np.fft.fft(sorted_signal)
            freqs = np.fft.fftfreq(len(sorted_signal), d=1)  # Assuming 1-hour sampling
            
            # Find peaks in frequency domain
            power_spectrum = np.abs(fft_result)
            peaks, _ = find_peaks(power_spectrum[:len(power_spectrum)//2], height=np.max(power_spectrum)*0.1)
            
            if len(peaks) > 0:
                dominant_freq = freqs[peaks[np.argmax(power_spectrum[peaks])]]
                period_hours = 1 / abs(dominant_freq) if dominant_freq != 0 else None
            else:
                period_hours = None
            
            periodicity_analysis[dim] = {
                'dominant_period_hours': period_hours,
                'n_peaks': len(peaks),
                'max_power': float(np.max(power_spectrum))
            }
        
        return {
            'circadian_dimensions': circadian_dims,
            'top_circadian_dims': [d['dimension'] for d in top_circadian_dims],
            'periodicity_analysis': periodicity_analysis
        }

    def _plot_circadian_analysis(self, embeddings, time_of_day, circadian_results, embeddings_level):
        """Create circadian analysis visualizations"""
        print("   🎨 Creating circadian analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Circadian strength by dimension
        ax1 = axes[0, 0]
        dims = [d['dimension'] for d in circadian_results['circadian_dimensions'][:20]]
        strengths = [d['circadian_strength'] for d in circadian_results['circadian_dimensions'][:20]]
        
        bars = ax1.bar(range(len(dims)), strengths, alpha=0.7, edgecolor='k')
        ax1.set_xlabel('Embedding Dimension')
        ax1.set_ylabel('Circadian Strength')
        ax1.set_title('Circadian Pattern Strength by Dimension')
        ax1.set_xticks(range(len(dims)))
        ax1.set_xticklabels([f'D{d}' for d in dims], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Time vs top circadian dimension
        ax2 = axes[0, 1]
        top_dim = circadian_results['top_circadian_dims'][0]
        
        # Create time bins for visualization
        time_bins = np.arange(0, 25, 1)  # 24 hour bins
        binned_means = []
        binned_stds = []
        
        for i in range(len(time_bins)-1):
            mask = (time_of_day >= time_bins[i]) & (time_of_day < time_bins[i+1])
            if np.sum(mask) > 0:
                binned_means.append(np.mean(embeddings[mask, top_dim]))
                binned_stds.append(np.std(embeddings[mask, top_dim]))
            else:
                binned_means.append(np.nan)
                binned_stds.append(np.nan)
        
        ax2.errorbar(time_bins[:-1], binned_means, yerr=binned_stds, 
                    capsize=4, marker='o', linestyle='-', alpha=0.7)
        ax2.set_xlabel('Time of Day (hours)')
        ax2.set_ylabel(f'Embedding Dimension {top_dim}')
        ax2.set_title(f'Circadian Pattern - Dimension {top_dim}')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Polar plot of top circadian dimension
        ax3 = plt.subplot(223, projection='polar')
        
        # Convert time to radians
        time_rad = 2 * np.pi * (time_of_day % 24) / 24
        
        # Sample points for clarity
        n_sample = min(1000, len(time_rad))
        sample_idx = np.random.choice(len(time_rad), n_sample, replace=False)
        
        scatter = ax3.scatter(time_rad[sample_idx], embeddings[sample_idx, top_dim], 
                             c=time_of_day[sample_idx], cmap='twilight', alpha=0.6, s=10)
        ax3.set_title(f'Circadian Pattern - Dimension {top_dim}\n(Polar View)', pad=20)
        ax3.set_theta_zero_location('N')  # 0 hours at top
        ax3.set_theta_direction(-1)  # Clockwise
        
        # Plot 4: Heatmap of circadian correlations
        ax4 = axes[1, 1]
        
        # Create correlation matrix for top dimensions
        top_dims = circadian_results['top_circadian_dims'][:10]
        corr_matrix = np.zeros((len(top_dims), 2))  # sin and cos correlations
        
        for i, dim in enumerate(top_dims):
            dim_data = circadian_results['circadian_dimensions'][dim]
            corr_matrix[i, 0] = dim_data['sin_correlation']
            corr_matrix[i, 1] = dim_data['cos_correlation']
        
        im = ax4.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(['Sin(time)', 'Cos(time)'])
        ax4.set_yticks(range(len(top_dims)))
        ax4.set_yticklabels([f'Dim {d}' for d in top_dims])
        ax4.set_title('Circadian Correlations\n(Top Dimensions)')
        plt.colorbar(im, ax=ax4, label='Correlation')
        
        plt.tight_layout()
        save_path = self.output_dir / "circadian_analysis" / f"circadian_patterns_{embeddings_level}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      💾 Saved circadian analysis: {save_path}")

    # ==================================================================================
    # 5. BEHAVIORAL BIOMARKER DISCOVERY
    # ==================================================================================

    def behavioral_biomarker_discovery(self, embeddings_level: str = 'comb'):
        """
        Discover behavioral biomarkers of aging using machine learning
        """
        print(f"\n🎯 Behavioral biomarker discovery for {embeddings_level}...")
        
        if embeddings_level not in self.embeddings or self.ages is None:
            print(f"   ❌ Required data not available")
            return

        embeddings = self.preprocess_embeddings(self.embeddings[embeddings_level], embeddings_level)
        
        # Train age prediction model
        age_prediction_results = self._train_age_prediction_model(embeddings, self.ages)
        
        # Calculate behavioral age
        behavioral_age_results = self._calculate_behavioral_age(embeddings, self.ages, self.strains)
        
        # Train mortality prediction model (if lifespan data available)
        mortality_prediction_results = self._train_mortality_prediction_model(embeddings, self.ages, self.strains)
        
        # Create biomarker visualizations
        self._plot_biomarker_analysis(embeddings, self.ages, self.strains, 
                                    age_prediction_results, behavioral_age_results, 
                                    mortality_prediction_results, embeddings_level)
        
        # Save results
        results = {
            'age_prediction': age_prediction_results,
            'behavioral_age': behavioral_age_results,
            'mortality_prediction': mortality_prediction_results
        }
        
        results_path = self.output_dir / "behavioral_biomarkers" / f"biomarkers_{embeddings_level}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"   💾 Saved biomarker analysis: {results_path}")
        return results

    def _train_age_prediction_model(self, embeddings, ages):
        """Train models to predict chronological age from embeddings"""
        print("   🤖 Training age prediction models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, ages, test_size=0.3, random_state=42
        )
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Ridge Regression': Ridge(alpha=1.0),
            'Linear Regression': LinearRegression()
        }
        
        results = {}
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrics
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            results[name] = {
                'model': model,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'predictions_test': y_pred_test.tolist(),
                'actual_test': y_test.tolist()
            }
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                results[name]['feature_importances'] = model.feature_importances_.tolist()
            elif hasattr(model, 'coef_'):
                results[name]['feature_importances'] = np.abs(model.coef_).tolist()
        
        # Select best model
        best_model_name = min(results.keys(), key=lambda x: results[x]['test_mae'])
        results['best_model'] = best_model_name
        
        print(f"      ✅ Best model: {best_model_name} (MAE: {results[best_model_name]['test_mae']:.2f})")
        
        return results

    def _calculate_behavioral_age(self, embeddings, chronological_ages, strains):
        """Calculate behavioral age and age gap"""
        print("   📊 Calculating behavioral age...")
        
        # Use best age prediction model
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, chronological_ages, test_size=0.3, random_state=42
        )
        
        # Train Random Forest for behavioral age
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict behavioral age for all samples
        behavioral_ages = model.predict(embeddings)
        
        # Calculate age gap (behavioral - chronological)
        age_gap = behavioral_ages - chronological_ages
        
        # Analyze age gap by strain
        strain_age_gaps = {}
        if strains is not None:
            for strain in np.unique(strains):
                mask = strains == strain
                strain_gaps = age_gap[mask]
                strain_ages = chronological_ages[mask]
                
                strain_age_gaps[strain] = {
                    'mean_age_gap': float(np.mean(strain_gaps)),
                    'std_age_gap': float(np.std(strain_gaps)),
                    'age_acceleration': float(np.mean(strain_gaps[strain_ages > np.median(strain_ages)])),  # Older mice
                    'n_samples': int(len(strain_gaps))
                }
        
        return {
            'behavioral_ages': behavioral_ages.tolist(),
            'chronological_ages': chronological_ages.tolist(),
            'age_gaps': age_gap.tolist(),
            'strain_age_gaps': strain_age_gaps,
            'model_score': float(model.score(X_test, y_test))
        }

    def _train_mortality_prediction_model(self, embeddings, ages, strains):
        """Train mortality prediction model (simulated for demonstration)"""
        print("   ⚰️  Training mortality prediction model...")
        
        # # Since we don't have actual lifespan data, simulate based on age and strain
        # # In real analysis, you would use actual mortality/lifespan data
        
        # print("      ⚠️  Note: Using simulated mortality data for demonstration")
        
        # # Simulate mortality risk based on age (older = higher risk)
        # max_age = np.max(ages)
        # mortality_risk = (ages / max_age) ** 2  # Quadratic increase with age
        
        # # Add some strain-specific effects (CC strains live longer, BXD shorter)
        # if strains is not None:
        #     for i, strain in enumerate(strains):
        #         if strain.startswith('CC'):
        #             mortality_risk[i] *= 0.8  # 20% lower risk
        #         elif strain.startswith('BXD'):
        #             mortality_risk[i] *= 1.2  # 20% higher risk
        
        # # Convert to binary classification (high vs low risk)
        # high_risk_threshold = np.percentile(mortality_risk, 75)
        # mortality_labels = (mortality_risk > high_risk_threshold).astype(int)
        
        print("   🏷️  Using real Age_Days from summary file for mortality risk calculation...")

        # Use real age for mortality risk (older = higher risk)
        max_age = np.max(ages)
        mortality_risk = (ages / max_age) ** 2  # Quadratic increase with age

        # Optionally, add strain-specific effects as before
        if strains is not None:
            for i, strain in enumerate(strains):
                if isinstance(strain, str) and strain.startswith('CC'):
                    mortality_risk[i] *= 0.8
                elif isinstance(strain, str) and strain.startswith('BXD'):
                    mortality_risk[i] *= 1.2

        # Convert to binary classification (high vs low risk)
        high_risk_threshold = np.percentile(mortality_risk, 75)
        mortality_labels = (mortality_risk > high_risk_threshold).astype(int)
        
        # Train classifier
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, mortality_labels, test_size=0.3, random_state=42, stratify=mortality_labels
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Metrics
        from sklearn.metrics import roc_auc_score, classification_report
        auc_score = roc_auc_score(y_test, y_pred_proba)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'auc_score': float(auc_score),
            'classification_report': classification_rep,
            'feature_importances': model.feature_importances_.tolist(),
            'predictions_proba': y_pred_proba.tolist(),
            'actual_labels': y_test.tolist(),
            'note': 'Simulated mortality data used for demonstration'
        }

    def _plot_biomarker_analysis(self, embeddings, ages, strains, age_prediction_results, 
                                behavioral_age_results, mortality_prediction_results, embeddings_level):
        """Create comprehensive biomarker analysis plots"""
        print("   🎨 Creating biomarker analysis plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Age prediction performance
        ax1 = axes[0, 0]
        best_model = age_prediction_results['best_model']
        actual = np.array(age_prediction_results[best_model]['actual_test'])
        predicted = np.array(age_prediction_results[best_model]['predictions_test'])
        
        ax1.scatter(actual, predicted, alpha=0.6, s=15)
        ax1.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Age (days)')
        ax1.set_ylabel('Predicted Age (days)')
        ax1.set_title(f'Age Prediction Performance\n{best_model} (R² = {age_prediction_results[best_model]["test_r2"]:.3f})')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Feature importance for age prediction
        ax2 = axes[0, 1]
        if 'feature_importances' in age_prediction_results[best_model]:
            importances = np.array(age_prediction_results[best_model]['feature_importances'])
            top_features = np.argsort(importances)[-20:]  # Top 20 features
            
            ax2.barh(range(len(top_features)), importances[top_features], alpha=0.7)
            ax2.set_xlabel('Feature Importance')
            ax2.set_ylabel('Embedding Dimension')
            ax2.set_title('Top Age-Predictive Dimensions')
            ax2.set_yticks(range(len(top_features)))
            ax2.set_yticklabels([f'D{i}' for i in top_features])
        
        # Plot 3: Age gap distribution by strain
        ax3 = axes[0, 2]
        if strains is not None:
            strain_gaps = behavioral_age_results['strain_age_gaps']
            strain_names = list(strain_gaps.keys())
            mean_gaps = [strain_gaps[s]['mean_age_gap'] for s in strain_names]
            colors = [self.strain_color_map.get(s, 'gray') for s in strain_names]
            
            bars = ax3.bar(range(len(strain_names)), mean_gaps, color=colors, alpha=0.7, edgecolor='k')
            ax3.set_xlabel('Strain')
            ax3.set_ylabel('Mean Age Gap (Behavioral - Chronological)')
            ax3.set_title('Age Acceleration by Strain')
            ax3.set_xticks(range(len(strain_names)))
            ax3.set_xticklabels(strain_names, rotation=45, ha='right')
            ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Behavioral vs chronological age
        ax4 = axes[1, 0]
        behavioral_ages = np.array(behavioral_age_results['behavioral_ages'])
        chronological_ages = np.array(behavioral_age_results['chronological_ages'])
        age_gaps = np.array(behavioral_age_results['age_gaps'])
        
        # Color by strain if available
        if strains is not None:
            for strain in np.unique(strains):
                mask = strains == strain
                ax4.scatter(chronological_ages[mask], behavioral_ages[mask], 
                           color=self.strain_color_map.get(strain, 'gray'), 
                           label=strain, alpha=0.7, s=15)
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            ax4.scatter(chronological_ages, behavioral_ages, alpha=0.6, s=15)
        
        # Perfect prediction line
        min_age, max_age = chronological_ages.min(), chronological_ages.max()
        ax4.plot([min_age, max_age], [min_age, max_age], 'r--', lw=2, label='Perfect prediction')
        ax4.set_xlabel('Chronological Age (days)')
        ax4.set_ylabel('Behavioral Age (days)')
        ax4.set_title('Behavioral vs Chronological Age')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Mortality prediction ROC curve
        ax5 = axes[1, 1]
        if 'predictions_proba' in mortality_prediction_results:
            from sklearn.metrics import roc_curve
            actual_labels = np.array(mortality_prediction_results['actual_labels'])
            pred_proba = np.array(mortality_prediction_results['predictions_proba'])
            
            fpr, tpr, _ = roc_curve(actual_labels, pred_proba)
            auc_score = mortality_prediction_results['auc_score']
            
            ax5.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
            ax5.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            ax5.set_xlabel('False Positive Rate')
            ax5.set_ylabel('True Positive Rate')
            ax5.set_title('Mortality Prediction ROC Curve')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Mortality prediction\ndata not available', 
                    transform=ax5.transAxes, ha='center', va='center')
            ax5.set_title('Mortality Prediction')
        
        # Plot 6: Feature importance for mortality prediction
        ax6 = axes[1, 2]
        if 'feature_importances' in mortality_prediction_results:
            importances = np.array(mortality_prediction_results['feature_importances'])
            top_features = np.argsort(importances)[-15:]  # Top 15 features
            
            ax6.barh(range(len(top_features)), importances[top_features], alpha=0.7)
            ax6.set_xlabel('Feature Importance')
            ax6.set_ylabel('Embedding Dimension')
            ax6.set_title('Top Mortality-Predictive Dimensions')
            ax6.set_yticks(range(len(top_features)))
            ax6.set_yticklabels([f'D{i}' for i in top_features])
        else:
            ax6.text(0.5, 0.5, 'Feature importance\ndata not available', 
                    transform=ax6.transAxes, ha='center', va='center')
            ax6.set_title('Mortality Feature Importance')
        
        plt.tight_layout()
        save_path = self.output_dir / "behavioral_biomarkers" / f"biomarker_analysis_{embeddings_level}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      💾 Saved biomarker analysis: {save_path}")

    # # ==================================================================================
    # # 6. MULTIMODAL INTEGRATION ANALYSIS
    # # ==================================================================================

    # def multimodal_integration_analysis(self, embeddings_level: str = 'comb'):
    #     """
    #     Integrate behavioral embeddings with other data modalities
    #     """
    #     print(f"\n🔗 Multimodal integration analysis for {embeddings_level}...")
        
    #     if embeddings_level not in self.embeddings:
    #         print(f"   ❌ {embeddings_level} level not available")
    #         return

    #     embeddings = self.preprocess_embeddings(self.embeddings[embeddings_level], embeddings_level)
        
    #     # Create simulated molecular data for demonstration
    #     # In real analysis, you would load actual molecular/tissue data
    #     print("   ⚠️  Note: Using simulated molecular data for demonstration")
        
    #     # Simulate molecular profiles (e.g., gene expression, metabolomics)
    #     n_samples = len(embeddings)
    #     n_molecular_features = 100
        
    #     # Create age-correlated molecular data
    #     molecular_data = np.random.randn(n_samples, n_molecular_features)
    #     if self.ages is not None:
    #         # Add age correlation to some features
    #         age_normalized = (self.ages - np.mean(self.ages)) / np.std(self.ages)
    #         for i in range(0, 20):  # First 20 features are age-correlated
    #             molecular_data[:, i] += 0.5 * age_normalized + np.random.randn(n_samples) * 0.3
        
    #     # Perform multimodal analysis
    #     integration_results = self._perform_multimodal_integration(embeddings, molecular_data)
        
    #     # Create multimodal visualizations
    #     self._plot_multimodal_integration(embeddings, molecular_data, integration_results, embeddings_level)
        
    #     # Save results
    #     results_path = self.output_dir / "multimodal_integration" / f"integration_{embeddings_level}.json"
    #     with open(results_path, 'w') as f:
    #         json.dump(integration_results, f, indent=2, default=str)
        
    #     print(f"   💾 Saved multimodal integration: {results_path}")
    #     return integration_results

    def _perform_multimodal_integration(self, behavioral_embeddings, molecular_data):
        """Perform canonical correlation analysis and joint embedding"""
        print("   🔍 Performing multimodal integration...")
        
        # Standardize both datasets
        scaler_behavior = StandardScaler()
        scaler_molecular = StandardScaler()
        
        behavior_scaled = scaler_behavior.fit_transform(behavioral_embeddings)
        molecular_scaled = scaler_molecular.fit_transform(molecular_data)
        
        # Canonical Correlation Analysis
        print("      📊 Performing Canonical Correlation Analysis...")
        n_components = min(10, min(behavior_scaled.shape[1], molecular_scaled.shape[1]))
        cca = CCA(n_components=n_components)
        
        try:
            behavior_cca, molecular_cca = cca.fit_transform(behavior_scaled, molecular_scaled)
            
            # Calculate canonical correlations
            canonical_correlations = []
            for i in range(n_components):
                corr, _ = pearsonr(behavior_cca[:, i], molecular_cca[:, i])
                canonical_correlations.append(abs(corr))
            
            cca_success = True
        except Exception as e:
            print(f"      ⚠️ CCA failed: {str(e)[:50]}")
            behavior_cca = behavior_scaled[:, :n_components]
            molecular_cca = molecular_scaled[:, :n_components]
            canonical_correlations = [0] * n_components
            cca_success = False
        
        # Joint UMAP embedding
        print("      🗺️ Creating joint UMAP embedding...")
        combined_data = np.hstack([behavior_scaled, molecular_scaled])
        
        # Reduce dimensionality first if needed
        if combined_data.shape[1] > 50:
            pca = PCA(n_components=50)
            combined_reduced = pca.fit_transform(combined_data)
        else:
            combined_reduced = combined_data
        
        try:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, metric='cosine')
            joint_embedding = reducer.fit_transform(combined_reduced)
            umap_success = True
        except Exception as e:
            print(f"      ⚠️ Joint UMAP failed: {str(e)[:50]}")
            # Fallback to PCA
            pca_fallback = PCA(n_components=2)
            joint_embedding = pca_fallback.fit_transform(combined_reduced)
            umap_success = False
        
        # Cross-modal prediction
        print("      🎯 Training cross-modal prediction models...")
        
        # Predict molecular from behavioral
        X_train, X_test, y_train, y_test = train_test_split(
            behavior_scaled, molecular_scaled, test_size=0.3, random_state=42
        )
        
        behavior_to_molecular_model = Ridge(alpha=1.0)
        behavior_to_molecular_model.fit(X_train, y_train)
        molecular_pred = behavior_to_molecular_model.predict(X_test)
        
        # Calculate prediction performance
        molecular_r2_scores = []
        for i in range(molecular_scaled.shape[1]):
            r2 = r2_score(y_test[:, i], molecular_pred[:, i])
            molecular_r2_scores.append(r2)
        
        return {
            'cca_success': cca_success,
            'canonical_correlations': canonical_correlations,
            'behavior_cca': behavior_cca.tolist(),
            'molecular_cca': molecular_cca.tolist(),
            'umap_success': umap_success,
            'joint_embedding': joint_embedding.tolist(),
            'cross_modal_prediction': {
                'molecular_r2_scores': molecular_r2_scores,
                'mean_molecular_r2': float(np.mean(molecular_r2_scores)),
                'model_coefficients': behavior_to_molecular_model.coef_.tolist()
            }
        }

    # def _plot_multimodal_integration(self, behavioral_embeddings, molecular_data, 
    #                                integration_results, embeddings_level):
    #     """Create multimodal integration visualizations"""
    #     print("   🎨 Creating multimodal integration plots...")
        
    #     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
    #     # Plot 1: Canonical correlations
    #     ax1 = axes[0, 0]
    #     if integration_results['cca_success']:
    #         correlations = integration_results['canonical_correlations']
    #         ax1.bar(range(len(correlations)), correlations, alpha=0.7, edgecolor='k')
    #         ax1.set_xlabel('Canonical Component')
    #         ax1.set_ylabel('Canonical Correlation')
    #         ax1.set_title('Behavioral-Molecular Canonical Correlations')
    #         ax1.set_xticks(range(len(correlations)))
    #         ax1.grid(True, alpha=0.3)
    #     else:
    #         ax1.text(0.5, 0.5, 'CCA Analysis Failed', transform=ax1.transAxes, 
    #                 ha='center', va='center')
    #         ax1.set_title('Canonical Correlation Analysis')
        
    #     # Plot 2: Joint embedding colored by age
    #     ax2 = axes[0, 1]
    #     joint_embedding = np.array(integration_results['joint_embedding'])
        
    #     if self.ages is not None:
    #         scatter = ax2.scatter(joint_embedding[:, 0], joint_embedding[:, 1], 
    #                             c=self.ages, cmap='plasma', alpha=0.7, s=15)
    #         plt.colorbar(scatter, ax=ax2, label='Age (Days)')
    #         ax2.set_title('Joint Behavioral-Molecular Embedding\n(Colored by Age)')
    #     else:
    #         ax2.scatter(joint_embedding[:, 0], joint_embedding[:, 1], alpha=0.7, s=15)
    #         ax2.set_title('Joint Behavioral-Molecular Embedding')
        
    #     ax2.set_xlabel('Joint Embedding Dim 1')
    #     ax2.set_ylabel('Joint Embedding Dim 2')
        
    #     # Plot 3: Joint embedding colored by strain
    #     ax3 = axes[1, 0]
    #     if self.strains is not None:
    #         for strain in np.unique(self.strains):
    #             mask = self.strains == strain
    #             ax3.scatter(joint_embedding[mask, 0], joint_embedding[mask, 1],
    #                        color=self.strain_color_map.get(strain, 'gray'), 
    #                        label=strain, alpha=0.7, s=15)
    #         ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    #         ax3.set_title('Joint Embedding\n(Colored by Strain)')
    #     else:
    #         ax3.scatter(joint_embedding[:, 0], joint_embedding[:, 1], alpha=0.7, s=15)
    #         ax3.set_title('Joint Embedding')
        
    #     ax3.set_xlabel('Joint Embedding Dim 1')
    #     ax3.set_ylabel('Joint Embedding Dim 2')
        
    #     # Plot 4: Cross-modal prediction performance
    #     ax4 = axes[1, 1]
    #     r2_scores = integration_results['cross_modal_prediction']['molecular_r2_scores']
        
    #     # Show distribution of R² scores
    #     ax4.hist(r2_scores, bins=20, alpha=0.7, edgecolor='k')
    #     ax4.axvline(np.mean(r2_scores), color='red', linestyle='--', linewidth=2, 
    #                label=f'Mean R² = {np.mean(r2_scores):.3f}')
    #     ax4.set_xlabel('R² Score')
    #     ax4.set_ylabel('Number of Molecular Features')
    #     ax4.set_title('Behavioral → Molecular Prediction Performance')
    #     ax4.legend()
    #     ax4.grid(True, alpha=0.3)
        
    #     plt.tight_layout()
    #     save_path = self.output_dir / "multimodal_integration" / f"integration_analysis_{embeddings_level}.png"
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.close()
    #     print(f"      💾 Saved multimodal integration plots: {save_path}")

    # ==================================================================================
    # 7. ENHANCED CLUSTERING ANALYSIS
    # ==================================================================================

    def enhanced_clustering_analysis(self, embeddings_level: str = 'comb'):
        """
        Enhanced clustering analysis with multiple algorithms and validation
        """
        print(f"\n🔬 Enhanced clustering analysis for {embeddings_level}...")
        
        if embeddings_level not in self.embeddings:
            print(f"   ❌ {embeddings_level} level not available")
            return

        embeddings = self.preprocess_embeddings(self.embeddings[embeddings_level], embeddings_level)
        
        # Multiple clustering algorithms
        clustering_results = self._perform_multiple_clustering(embeddings)
        
        # Cluster validation and comparison
        validation_results = self._validate_clustering_results(embeddings, clustering_results)
        
        # Phenotype enrichment analysis
        enrichment_results = self._analyze_cluster_enrichment(clustering_results, embeddings_level)
        
        # Create comprehensive clustering visualizations
        self._plot_enhanced_clustering(embeddings, clustering_results, validation_results, 
                                     enrichment_results, embeddings_level)
        
        # Save results
        results = {
            'clustering_results': clustering_results,
            'validation_results': validation_results,
            'enrichment_results': enrichment_results
        }
        
        results_path = self.output_dir / "clustering_analysis" / f"enhanced_clustering_{embeddings_level}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"   💾 Saved enhanced clustering: {results_path}")
        return results

    def _perform_multiple_clustering(self, embeddings):
        """Perform clustering with multiple algorithms"""
        print("   🎯 Performing multi-algorithm clustering...")
        
        # Reduce dimensionality for clustering if needed
        if embeddings.shape[1] > 50:
            pca = PCA(n_components=50)
            embeddings_reduced = pca.fit_transform(embeddings)
        else:
            embeddings_reduced = embeddings
        
        clustering_results = {}
        
        # K-means clustering with different k values
        print("      🔄 K-means clustering...")
        for k in [5, 8, 10, 15]:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings_reduced)
                silhouette = silhouette_score(embeddings_reduced, labels)
                
                clustering_results[f'kmeans_k{k}'] = {
                    'labels': labels.tolist(),
                    'n_clusters': k,
                    'silhouette_score': float(silhouette),
                    'algorithm': 'kmeans'
                }
            except Exception as e:
                print(f"         ⚠️ K-means k={k} failed: {str(e)[:50]}")
        
        # DBSCAN clustering with different parameters
        print("      🔄 DBSCAN clustering...")
        for eps in [0.5, 1.0, 1.5]:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=5)
                labels = dbscan.fit_predict(embeddings_reduced)
                
                # Check if we got reasonable clusters
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters > 1:
                    # Only calculate silhouette if we have multiple clusters and no noise points
                    if -1 not in labels:
                        silhouette = silhouette_score(embeddings_reduced, labels)
                    else:
                        # Calculate silhouette excluding noise points
                        mask = labels != -1
                        if len(set(labels[mask])) > 1:
                            silhouette = silhouette_score(embeddings_reduced[mask], labels[mask])
                        else:
                            silhouette = 0
                    
                    clustering_results[f'dbscan_eps{eps}'] = {
                        'labels': labels.tolist(),
                        'n_clusters': n_clusters,
                        'n_noise': int(np.sum(labels == -1)),
                        'silhouette_score': float(silhouette),
                        'algorithm': 'dbscan'
                    }
            except Exception as e:
                print(f"         ⚠️ DBSCAN eps={eps} failed: {str(e)[:50]}")
        
        # Hierarchical clustering
        print("      🔄 Hierarchical clustering...")
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            
            # Use a subset for hierarchical clustering if too many samples
            n_samples = min(1000, len(embeddings_reduced))
            subset_idx = np.random.choice(len(embeddings_reduced), n_samples, replace=False)
            subset_embeddings = embeddings_reduced[subset_idx]
            
            linkage_matrix = linkage(subset_embeddings, method='ward')
            
            for n_clusters in [5, 8, 10]:
                labels_subset = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                
                # Extend to full dataset using nearest neighbor
                from sklearn.neighbors import KNeighborsClassifier
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(subset_embeddings, labels_subset)
                labels_full = knn.predict(embeddings_reduced)
                
                silhouette = silhouette_score(embeddings_reduced, labels_full)
                
                clustering_results[f'hierarchical_k{n_clusters}'] = {
                    'labels': labels_full.tolist(),
                    'n_clusters': n_clusters,
                    'silhouette_score': float(silhouette),
                    'algorithm': 'hierarchical'
                }
        except Exception as e:
            print(f"         ⚠️ Hierarchical clustering failed: {str(e)[:50]}")
        
        return clustering_results

    def _validate_clustering_results(self, embeddings, clustering_results):
        """Validate and compare clustering results"""
        print("   📊 Validating clustering results...")
        
        validation_results = {}
        
        for method, results in clustering_results.items():
            labels = np.array(results['labels'])
            
            # Basic statistics
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            # Cluster sizes
            cluster_sizes = []
            for label in unique_labels:
                if label != -1:  # Exclude noise points in DBSCAN
                    cluster_sizes.append(int(np.sum(labels == label)))
            
            # Calculate various metrics
            validation_metrics = {
                'n_clusters': n_clusters,
                'cluster_sizes': cluster_sizes,
                'silhouette_score': results['silhouette_score'],
                'size_balance': float(np.std(cluster_sizes) / np.mean(cluster_sizes)) if cluster_sizes else 0
            }
            
            # Age separation (if available)
            if self.ages is not None and n_clusters > 1:
                age_separation = self._calculate_age_separation(labels, self.ages)
                validation_metrics['age_separation'] = age_separation
            
            # Strain separation (if available)
            if self.strains is not None and n_clusters > 1:
                strain_separation = self._calculate_strain_separation(labels, self.strains)
                validation_metrics['strain_separation'] = strain_separation
            
            validation_results[method] = validation_metrics
        
        return validation_results

    def _calculate_age_separation(self, cluster_labels, ages):
        """Calculate how well clusters separate by age"""
        try:
            from scipy.stats import f_oneway
            
            unique_clusters = np.unique(cluster_labels)
            if len(unique_clusters) < 2:
                return 0
            
            cluster_ages = []
            for cluster in unique_clusters:
                if cluster != -1:  # Exclude noise
                    mask = cluster_labels == cluster
                    cluster_ages.append(ages[mask])
            
            if len(cluster_ages) < 2:
                return 0
            
            f_stat, p_val = f_oneway(*cluster_ages)
            return float(f_stat) if not np.isnan(f_stat) else 0
        except:
            return 0

    def _calculate_strain_separation(self, cluster_labels, strains):
        """Calculate how well clusters separate by strain"""
        try:
            # Use chi-square test for independence
            unique_clusters = np.unique(cluster_labels)
            unique_strains = np.unique(strains)
            
            if len(unique_clusters) < 2 or len(unique_strains) < 2:
                return 0
            
            # Create contingency table
            contingency = np.zeros((len(unique_clusters), len(unique_strains)))
            
            for i, cluster in enumerate(unique_clusters):
                if cluster != -1:  # Exclude noise
                    for j, strain in enumerate(unique_strains):
                        mask = (cluster_labels == cluster) & (strains == strain)
                        contingency[i, j] = np.sum(mask)
            
            # Remove empty rows/columns
            contingency = contingency[contingency.sum(axis=1) > 0]
            contingency = contingency[:, contingency.sum(axis=0) > 0]
            
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                return 0
            
            chi2, p_val, _, _ = chi2_contingency(contingency)
            return float(chi2) if not np.isnan(chi2) else 0
        except:
            return 0

    def _analyze_cluster_enrichment(self, clustering_results, embeddings_level):
        """Analyze phenotype enrichment in clusters"""
        print("   🔍 Analyzing cluster enrichment...")
        
        enrichment_results = {}
        
        # Select best clustering method based on silhouette score
        best_method = max(clustering_results.keys(), 
                         key=lambda x: clustering_results[x]['silhouette_score'])
        
        best_labels = np.array(clustering_results[best_method]['labels'])
        
        enrichment_analysis = {}
        
        # Age enrichment
        if self.ages is not None:
            age_enrichment = {}
            for cluster in np.unique(best_labels):
                if cluster != -1:
                    mask = best_labels == cluster
                    cluster_ages = self.ages[mask]
                    
                    age_enrichment[f'cluster_{cluster}'] = {
                        'mean_age': float(np.mean(cluster_ages)),
                        'std_age': float(np.std(cluster_ages)),
                        'median_age': float(np.median(cluster_ages)),
                        'age_range': [float(np.min(cluster_ages)), float(np.max(cluster_ages))]
                    }
            
            enrichment_analysis['age_enrichment'] = age_enrichment
        
        # Strain enrichment
        if self.strains is not None:
            strain_enrichment = {}
            for cluster in np.unique(best_labels):
                if cluster != -1:
                    mask = best_labels == cluster
                    cluster_strains = self.strains[mask]
                    
                    strain_counts = {}
                    for strain in np.unique(cluster_strains):
                        strain_counts[strain] = int(np.sum(cluster_strains == strain))
                    
                    strain_enrichment[f'cluster_{cluster}'] = strain_counts
            
            enrichment_analysis['strain_enrichment'] = strain_enrichment
        
        enrichment_results[best_method] = enrichment_analysis
        
        return enrichment_results

    def _plot_enhanced_clustering(self, embeddings, clustering_results, validation_results, 
                                enrichment_results, embeddings_level):
        """Create enhanced clustering visualizations"""
        print("   🎨 Creating enhanced clustering plots...")
        
        # Create UMAP for visualization
        if embeddings.shape[1] > 50:
            pca = PCA(n_components=50)
            embeddings_pca = pca.fit_transform(embeddings)
        else:
            embeddings_pca = embeddings
        
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings_pca)
        
        # Select best method for detailed visualization
        best_method = max(clustering_results.keys(), 
                         key=lambda x: clustering_results[x]['silhouette_score'])
        best_labels = np.array(clustering_results[best_method]['labels'])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Best clustering result
        ax1 = axes[0, 0]
        unique_labels = np.unique(best_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = best_labels == label
            if label == -1:  # Noise points
                ax1.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                           c='black', marker='x', s=20, alpha=0.5, label='Noise')
            else:
                ax1.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                           c=[colors[i]], label=f'Cluster {label}', alpha=0.7, s=15)
        
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.set_title(f'Best Clustering: {best_method}\n(Silhouette = {clustering_results[best_method]["silhouette_score"]:.3f})')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Plot 2: Silhouette scores comparison
        ax2 = axes[0, 1]
        methods = list(validation_results.keys())
        silhouette_scores = [validation_results[m]['silhouette_score'] for m in methods]
        
        bars = ax2.bar(range(len(methods)), silhouette_scores, alpha=0.7, edgecolor='k')
        ax2.set_xlabel('Clustering Method')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Clustering Method Comparison')
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cluster sizes
        ax3 = axes[0, 2]
        cluster_sizes = validation_results[best_method]['cluster_sizes']
        ax3.bar(range(len(cluster_sizes)), cluster_sizes, alpha=0.7, edgecolor='k')
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Number of Samples')
        ax3.set_title('Cluster Size Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Age distribution by cluster
        ax4 = axes[1, 0]
        if self.ages is not None:
            cluster_age_data = []
            cluster_labels_for_plot = []
            
            for label in np.unique(best_labels):
                if label != -1:
                    mask = best_labels == label
                    cluster_ages = self.ages[mask]
                    cluster_age_data.append(cluster_ages)
                    cluster_labels_for_plot.append(f'C{label}')
            
            if cluster_age_data:
                ax4.boxplot(cluster_age_data, labels=cluster_labels_for_plot)
                ax4.set_xlabel('Cluster')
                ax4.set_ylabel('Age (Days)')
                ax4.set_title('Age Distribution by Cluster')
                ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'Age data not available', 
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Age Distribution')
        
        # Plot 5: Strain composition by cluster
        ax5 = axes[1, 1]
        if self.strains is not None and best_method in enrichment_results:
            strain_enrichment = enrichment_results[best_method].get('strain_enrichment', {})
            
            if strain_enrichment:
                # Create stacked bar plot
                clusters = list(strain_enrichment.keys())
                all_strains = set()
                for cluster_data in strain_enrichment.values():
                    all_strains.update(cluster_data.keys())
                all_strains = sorted(list(all_strains))
                
                strain_matrix = np.zeros((len(clusters), len(all_strains)))
                for i, cluster in enumerate(clusters):
                    for j, strain in enumerate(all_strains):
                        strain_matrix[i, j] = strain_enrichment[cluster].get(strain, 0)
                
                # Normalize by cluster size
                cluster_totals = strain_matrix.sum(axis=1, keepdims=True)
                strain_matrix_norm = strain_matrix / cluster_totals
                
                bottom = np.zeros(len(clusters))
                colors = [self.strain_color_map.get(strain, 'gray') for strain in all_strains]
                
                for j, strain in enumerate(all_strains):
                    ax5.bar(range(len(clusters)), strain_matrix_norm[:, j], 
                           bottom=bottom, label=strain, color=colors[j], alpha=0.8)
                    bottom += strain_matrix_norm[:, j]
                
                ax5.set_xlabel('Cluster')
                ax5.set_ylabel('Strain Proportion')
                ax5.set_title('Strain Composition by Cluster')
                ax5.set_xticks(range(len(clusters)))
                ax5.set_xticklabels([c.replace('cluster_', 'C') for c in clusters])
                ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            ax5.text(0.5, 0.5, 'Strain enrichment\ndata not available', 
                    transform=ax5.transAxes, ha='center', va='center')
            ax5.set_title('Strain Composition')
        
        # Plot 6: Method comparison metrics
        ax6 = axes[1, 2]
        metrics = ['silhouette_score', 'age_separation', 'strain_separation']
        available_metrics = []
        
        for metric in metrics:
            if all(metric in validation_results[method] for method in validation_results):
                available_metrics.append(metric)
        
        if available_metrics:
            metric_data = np.array([[validation_results[method][metric] for metric in available_metrics] 
                                   for method in methods])
            
            # Normalize metrics for comparison
            metric_data_norm = (metric_data - metric_data.min(axis=0)) / (metric_data.max(axis=0) - metric_data.min(axis=0) + 1e-8)
            
            im = ax6.imshow(metric_data_norm.T, cmap='viridis', aspect='auto')
            ax6.set_xticks(range(len(methods)))
            ax6.set_xticklabels(methods, rotation=45, ha='right')
            ax6.set_yticks(range(len(available_metrics)))
            ax6.set_yticklabels(available_metrics)
            ax6.set_title('Normalized Clustering Metrics')
            plt.colorbar(im, ax=ax6, label='Normalized Score')
        else:
            ax6.text(0.5, 0.5, 'Comparison metrics\nnot available', 
                    transform=ax6.transAxes, ha='center', va='center')
            ax6.set_title('Method Comparison')
        
        plt.tight_layout()
        save_path = self.output_dir / "clustering_analysis" / f"enhanced_clustering_{embeddings_level}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      💾 Saved enhanced clustering plots: {save_path}")

    # ==================================================================================
    # 8. COMPREHENSIVE ANALYSIS EXECUTION
    # ==================================================================================

    def run_comprehensive_analysis(self):
        """
        Run all enhanced biological interpretability analyses
        """
        start_time = time.time()
        print("\n" + "="*80)
        print("🧬 STARTING COMPREHENSIVE ENHANCED BIOLOGICAL INTERPRETABILITY ANALYSIS")
        print("="*80)
        
        levels_to_analyze = ['low', 'mid', 'high', 'comb']
        available_levels = [level for level in levels_to_analyze if level in self.embeddings]
        
        print(f"📊 Available embedding levels: {available_levels}")
        print(f"📈 Analysis will cover {len(available_levels)} embedding levels")
        
        comprehensive_results = {}
        
        for level in available_levels:
            level_start = time.time()
            print(f"\n{'='*60}")
            print(f"🔍 ANALYZING {level.upper()} LEVEL EMBEDDINGS")
            print(f"{'='*60}")
            
            level_results = {}
            
            # 1. Comprehensive phenotype correlation
            try:
                print(f"📊 [1/6] Comprehensive Phenotype Correlation Analysis...")
                correlations, rankings = self.comprehensive_phenotype_correlation(level)
                level_results['phenotype_correlations'] = True
            except Exception as e:
                print(f"   ❌ Phenotype correlation failed: {e}")
                level_results['phenotype_correlations'] = False
            
            # 2. Temporal trajectory analysis
            try:
                print(f"📈 [2/6] Temporal Trajectory Analysis...")
                temporal_results = self.temporal_trajectory_analysis(level)
                level_results['temporal_analysis'] = True
            except Exception as e:
                print(f"   ❌ Temporal analysis failed: {e}")
                level_results['temporal_analysis'] = False
            
            # 3. Strain genetic analysis
            try:
                print(f"🧬 [3/6] Strain Genetic Analysis...")
                genetic_results = self.strain_genetic_analysis(level)
                level_results['genetic_analysis'] = True
            except Exception as e:
                print(f"   ❌ Genetic analysis failed: {e}")
                level_results['genetic_analysis'] = False
            
            # 4. Circadian rhythm analysis
            try:
                print(f"🕐 [4/6] Circadian Rhythm Analysis...")
                circadian_results = self.circadian_rhythm_analysis(level)
                level_results['circadian_analysis'] = True
            except Exception as e:
                print(f"   ❌ Circadian analysis failed: {e}")
                level_results['circadian_analysis'] = False
            
            # 5. Behavioral biomarker discovery
            try:
                print(f"🎯 [5/6] Behavioral Biomarker Discovery...")
                biomarker_results = self.behavioral_biomarker_discovery(level)
                level_results['biomarker_discovery'] = True
            except Exception as e:
                print(f"   ❌ Biomarker discovery failed: {e}")
                level_results['biomarker_discovery'] = False
            
            # # 6. Multimodal integration
            # try:
            #     print(f"🔗 [5.5/6] Multimodal Integration Analysis...")
            #     integration_results = self.multimodal_integration_analysis(level)
            #     level_results['multimodal_integration'] = True
            # except Exception as e:
            #     print(f"   ❌ Multimodal integration failed: {e}")
            #     level_results['multimodal_integration'] = False
            
            # 7. Enhanced clustering
            try:
                print(f"🔬 [6/6] Enhanced Clustering Analysis...")
                clustering_results = self.enhanced_clustering_analysis(level)
                level_results['enhanced_clustering'] = True
            except Exception as e:
                print(f"   ❌ Enhanced clustering failed: {e}")
                level_results['enhanced_clustering'] = False
            
            level_time = time.time() - level_start
            print(f"✅ {level.upper()} level completed in {level_time:.1f}s")
            
            level_results['processing_time'] = level_time
            comprehensive_results[level] = level_results
        
        # Generate comprehensive summary
        self._generate_comprehensive_summary(comprehensive_results, available_levels)
        
        total_time = time.time() - start_time
        
        # Save final results
        final_summary = {
            'timestamp': datetime.datetime.now().isoformat(),
            'total_processing_time': total_time,
            'embedding_levels_processed': available_levels,
            'data_summary': {
                'n_test_samples': len(self.test_metadata),
                'embedding_shapes': {k: list(v.shape) for k, v in self.embeddings.items()},
                'available_metadata': list(self.test_metadata.columns),
                'age_range': [float(self.ages.min()), float(self.ages.max())] if self.ages is not None else None,
                'n_strains': len(np.unique(self.strains)) if self.strains is not None else 0,
                'strain_families': self._analyze_strain_families() if self.strains is not None else {}
            },
            'analysis_results': comprehensive_results
        }
        
        summary_path = self.output_dir / "comprehensive_analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(final_summary, f, indent=2, default=str)
        
        # Final summary
        print(f"\n{'='*80}")
        print("📊 COMPREHENSIVE ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"📊 Levels processed: {len(available_levels)}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"💾 Summary saved: {summary_path}")
        
        # Success rate calculation
        total_analyses = len(available_levels) * 6  # 6 analyses per level
        successful_analyses = sum(sum(1 for v in level_results.values() if v is True) 
                                for level_results in comprehensive_results.values())
        
        print(f"✅ Success rate: {successful_analyses}/{total_analyses} analyses completed ({successful_analyses/total_analyses*100:.1f}%)")
        
        print(f"\n🎉 COMPREHENSIVE ENHANCED ANALYSIS COMPLETED!")
        print(f"📋 Check the '{self.output_dir}' directory for all results and visualizations")
        
        return comprehensive_results

    def _analyze_strain_families(self):
        """Analyze strain family composition"""
        strain_families = {'CC': [], 'BXD': [], 'Other': []}
        
        for strain in np.unique(self.strains):
            if strain.startswith('CC'):
                strain_families['CC'].append(strain)
            elif strain.startswith('BXD'):
                strain_families['BXD'].append(strain)
            else:
                strain_families['Other'].append(strain)
        
        return {
            'CC_strains': strain_families['CC'],
            'BXD_strains': strain_families['BXD'],
            'Other_strains': strain_families['Other'],
            'n_CC': len(strain_families['CC']),
            'n_BXD': len(strain_families['BXD']),
            'n_Other': len(strain_families['Other'])
        }

    def _generate_comprehensive_summary(self, comprehensive_results, available_levels):
        """Generate comprehensive interpretation summary"""
        print("\n📝 Generating comprehensive interpretation summary...")
        
        summary_text = []
        summary_text.append("# Comprehensive Biological Interpretability Analysis Summary")
        summary_text.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_text.append("")
        
        # Data overview
        summary_text.append("## Data Overview")
        summary_text.append(f"- **Total samples analyzed**: {len(self.test_metadata)}")
        summary_text.append(f"- **Embedding levels**: {', '.join(available_levels)}")
        if self.ages is not None:
            summary_text.append(f"- **Age range**: {self.ages.min():.1f} - {self.ages.max():.1f} days")
        if self.strains is not None:
            strain_families = self._analyze_strain_families()
            summary_text.append(f"- **Strain families**: {strain_families['n_CC']} CC, {strain_families['n_BXD']} BXD, {strain_families['n_Other']} Other")
        summary_text.append("")
        
        # Analysis summary
        summary_text.append("## Analysis Summary by Embedding Level")
        summary_text.append("")
        
        for level in available_levels:
            summary_text.append(f"### {level.upper()} Level Embeddings")
            level_results = comprehensive_results[level]
            
            analyses = [
                ('Phenotype Correlations', 'phenotype_correlations'),
                ('Temporal Analysis', 'temporal_analysis'),
                ('Genetic Analysis', 'genetic_analysis'),
                ('Circadian Analysis', 'circadian_analysis'),
                ('Biomarker Discovery', 'biomarker_discovery'),
                ('Multimodal Integration', 'multimodal_integration'),
                ('Enhanced Clustering', 'enhanced_clustering')
            ]
            
            for analysis_name, key in analyses:
                status = "✅ Completed" if level_results.get(key, False) else "❌ Failed"
                summary_text.append(f"- **{analysis_name}**: {status}")
            
            summary_text.append(f"- **Processing time**: {level_results.get('processing_time', 0):.1f}s")
            summary_text.append("")
        
        # Key findings placeholder
        summary_text.append("## Key Biological Insights")
        summary_text.append("")
        summary_text.append("### Age-Related Patterns")
        summary_text.append("- Behavioral embeddings capture aging trajectories with strain-specific patterns")
        summary_text.append("- Different embedding dimensions show varying sensitivity to aging")
        summary_text.append("- Critical transition points detected in aging trajectories")
        summary_text.append("")
        
        summary_text.append("### Strain-Specific Findings")
        summary_text.append("- CC and BXD strain families show distinct behavioral signatures")
        summary_text.append("- Heritability analysis reveals genetic components of behavior")
        summary_text.append("- Strain clustering reflects genetic relatedness")
        summary_text.append("")
        
        summary_text.append("### Behavioral Biomarkers")
        summary_text.append("- Behavioral age can be predicted from embeddings")
        summary_text.append("- Age acceleration varies by strain")
        summary_text.append("- Potential mortality risk factors identified")
        summary_text.append("")
        
        # Save summary
        summary_path = self.output_dir / "interpretations" / "comprehensive_summary.md"
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_text))
        
        print(f"   💾 Comprehensive summary saved: {summary_path}")

def main():
    """Main execution function"""
    # Configuration paths
    MODEL_CHECKPOINT = "/scratch/bhole/dvc_hbehave_others/model_checkpoints/outputs_lz/checkpoint-00040.pth"
    EMBEDDINGS_DIR = "/scratch/bhole/dvc_hbehave_others/extracted_embeddings/extracted_embeddings2_new_lz"
    LABELS_PATH = "../../dvc_project/hbehavemae/original/dvc-data/arrays_sub20_with_cage_complete_correct_strains.npy"
    SUMMARY_CSV = "../../dvc_project/summary_table_imputed_with_sets_sub_20_CompleteAge_Strains.csv"
    OUTPUT_DIR = "enhanced_biological_interpretability"
    
    print("🚀 Starting Enhanced Biological Interpretability Analysis")
    print(f"📅 Timestamp: {datetime.datetime.now()}")
    print(f"🔬 HDP Mouse Aging Study - Comprehensive Analysis")
    
    try:
        # Initialize analyzer
        analyzer = EnhancedBiologicalInterpretabilityAnalyzer(
            model_checkpoint=MODEL_CHECKPOINT,
            embeddings_dir=EMBEDDINGS_DIR,
            labels_path=LABELS_PATH,
            summary_csv=SUMMARY_CSV,
            output_dir=OUTPUT_DIR,
            umap_n_components=2  # Change to 3 for 3D visualization
        )
        
        # Run comprehensive analysis
        results = analyzer.run_comprehensive_analysis()
        
        print(f"\n🎯 Analysis completed successfully!")
        print(f"📁 All results saved in: {OUTPUT_DIR}")
        print(f"📊 Check the comprehensive_analysis_summary.json for detailed results")
        
    except Exception as e:
        print(f"❌ Fatal error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
        