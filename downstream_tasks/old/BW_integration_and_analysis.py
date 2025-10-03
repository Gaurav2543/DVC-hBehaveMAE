#!/usr/bin/env python3
"""
Bodyweight integration and enhanced analysis features for embedding analysis.
Handles bodyweight data loading, matching, and integration with embeddings.
"""
import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from pygam import LinearGAM, s
from tqdm import tqdm
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json

from downstream_tasks.embedding_analysis import EmbeddingAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

class BodyweightHandler:
    """Handles bodyweight data loading and matching to timepoints"""
    
    def __init__(self, bw_file, tissue_file, summary_metadata_file):
        self.bw_file = bw_file
        self.tissue_file = tissue_file
        self.summary_metadata_file = summary_metadata_file
        self.load_bodyweight_data()
        self.load_summary_metadata()
    
    def load_bodyweight_data(self):
        """Load and merge bodyweight data from BW.csv and TissueCollection_PP.csv"""
        print("Loading bodyweight data...")
        
        # Load BW data
        bw_df = pd.read_csv(self.bw_file)
        bw_df['date'] = pd.to_datetime(bw_df['date'])
        
        # Load tissue collection data
        tissue_df = pd.read_csv(self.tissue_file)
        tissue_df['death_date'] = pd.to_datetime(tissue_df['death_date'])
        
        # Rename columns for consistency
        tissue_df = tissue_df.rename(columns={
            'death_date': 'date',
            'weight_pre_sac': 'body_weight'
        })
        
        # Combine the datasets
        self.combined_bw = pd.concat([
            bw_df[['animal_id', 'date', 'body_weight']],
            tissue_df[['animal_id', 'date', 'body_weight']]
        ], ignore_index=True)
        
        # Remove duplicates and sort
        self.combined_bw = self.combined_bw.drop_duplicates(
            subset=['animal_id', 'date']
        ).sort_values(['animal_id', 'date'])
        
        print(f"Loaded bodyweight data for {self.combined_bw['animal_id'].nunique()} animals "
              f"with {len(self.combined_bw)} total measurements")
    
    def load_summary_metadata(self):
        """Load summary metadata"""
        self.summary_metadata = pd.read_csv(self.summary_metadata_file)
        self.summary_metadata['from_tpt'] = pd.to_datetime(self.summary_metadata['from_tpt'])
        self.summary_metadata['to_tpt'] = pd.to_datetime(self.summary_metadata['to_tpt'])
    
    def match_bodyweight_to_timepoint(self, animal_ids, target_date, strategy="gam_spline"):
        """Match bodyweight measurement to a specific timepoint for given animals"""
        if not animal_ids:
            return np.nan
        
        # Get bodyweight data for all animals in the cage
        animal_bw_data = self.combined_bw[self.combined_bw['animal_id'].isin(animal_ids)].copy()
        
        if animal_bw_data.empty:
            return np.nan
        
        if strategy == "gam_spline":
            return self._gam_spline_matching(animal_ids, target_date, animal_bw_data)
        elif strategy == "closest":
            return self._closest_matching(target_date, animal_bw_data)
        elif strategy == "interpolate":
            return self._interpolate_matching(animal_ids, target_date, animal_bw_data)
        elif strategy == "most_recent":
            return self._most_recent_matching(target_date, animal_bw_data)
        
        return np.nan
    
    def _gam_spline_matching(self, animal_ids, target_date, animal_bw_data):
        """GAM spline interpolation strategy"""
        weights = []
        for animal_id in animal_ids:
            animal_data = animal_bw_data[animal_bw_data['animal_id'] == animal_id].sort_values('date')
            if len(animal_data) < 3:
                if len(animal_data) >= 1:
                    animal_data['time_diff'] = abs((animal_data['date'] - target_date).dt.total_seconds())
                    weights.append(animal_data.loc[animal_data['time_diff'].idxmin()]['body_weight'])
                continue
            
            try:
                first_date = animal_data['date'].min()
                t = (animal_data['date'] - first_date).dt.total_seconds() / (24 * 3600)
                y = animal_data['body_weight'].values
                target_t = (target_date - first_date).total_seconds() / (24 * 3600)
                
                n_splines = min(10, len(t) - 1, max(3, len(t) // 3))
                gam = LinearGAM(s(0, n_splines=n_splines)).fit(t, y)
                
                predicted_weight = gam.predict([target_t])[0]
                
                min_weight, max_weight = y.min(), y.max()
                weight_range = max_weight - min_weight
                if min_weight - weight_range <= predicted_weight <= max_weight + weight_range:
                    weights.append(predicted_weight)
                else:
                    animal_data['time_diff'] = abs((animal_data['date'] - target_date).dt.total_seconds())
                    weights.append(animal_data.loc[animal_data['time_diff'].idxmin()]['body_weight'])
                    
            except Exception:
                animal_data['time_diff'] = abs((animal_data['date'] - target_date).dt.total_seconds())
                weights.append(animal_data.loc[animal_data['time_diff'].idxmin()]['body_weight'])
        
        return np.mean(weights) if weights else np.nan
    
    def _closest_matching(self, target_date, animal_bw_data):
        """Closest time matching strategy"""
        animal_bw_data['time_diff'] = abs((animal_bw_data['date'] - target_date).dt.total_seconds())
        closest_measurements = animal_bw_data.loc[
            animal_bw_data.groupby('animal_id')['time_diff'].idxmin()
        ]
        return closest_measurements['body_weight'].mean()
    
    def _interpolate_matching(self, animal_ids, target_date, animal_bw_data):
        """Linear interpolation strategy"""
        weights = []
        for animal_id in animal_ids:
            animal_data = animal_bw_data[animal_bw_data['animal_id'] == animal_id].sort_values('date')
            if len(animal_data) < 2:
                if len(animal_data) == 1:
                    weights.append(animal_data.iloc[0]['body_weight'])
                continue
            
            before = animal_data[animal_data['date'] <= target_date]
            after = animal_data[animal_data['date'] > target_date]
            
            if before.empty and not after.empty:
                weights.append(after.iloc[0]['body_weight'])
            elif not before.empty and after.empty:
                weights.append(before.iloc[-1]['body_weight'])
            elif not before.empty and not after.empty:
                before_row = before.iloc[-1]
                after_row = after.iloc[0]
                time_total = (after_row['date'] - before_row['date']).total_seconds()
                time_from_before = (target_date - before_row['date']).total_seconds()
                
                if time_total == 0:
                    weights.append(before_row['body_weight'])
                else:
                    interpolated = (before_row['body_weight'] + 
                                  (after_row['body_weight'] - before_row['body_weight']) * 
                                  (time_from_before / time_total))
                    weights.append(interpolated)
        
        return np.mean(weights) if weights else np.nan
    
    def _most_recent_matching(self, target_date, animal_bw_data):
        """Most recent measurement strategy"""
        valid_measurements = animal_bw_data[animal_bw_data['date'] <= target_date]
        if valid_measurements.empty:
            return np.nan
        most_recent = valid_measurements.loc[
            valid_measurements.groupby('animal_id')['date'].idxmax()
        ]
        return most_recent['body_weight'].mean()
    
    def add_bodyweight_to_metadata(self, metadata, strategy="closest"):
        """Add bodyweight information to metadata"""
        print(f"Adding bodyweight to metadata using {strategy} strategy...")
        
        bodyweights = []
        
        def process_single_row(row):
            if pd.isna(row.animal_ids) or row.animal_ids == '':
                return np.nan
            animal_ids = [aid.strip() for aid in str(row.animal_ids).split(';') if aid.strip()]
            
            target_date = row.from_tpt + (row.to_tpt - row.from_tpt) / 2
            return self.match_bodyweight_to_timepoint(animal_ids, target_date, strategy)
        
        # Process with progress bar
        with ThreadPoolExecutor(max_workers=4) as executor:
            with tqdm(total=len(self.summary_metadata), desc="Processing bodyweight") as pbar:
                future_to_idx = {
                    executor.submit(process_single_row, row): idx 
                    for idx, row in enumerate(self.summary_metadata.itertuples())
                }
                
                bodyweights = [np.nan] * len(self.summary_metadata)
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    bodyweights[idx] = future.result()
                    pbar.update(1)
        
        self.summary_metadata['bodyweight'] = bodyweights
        
        # Merge with original metadata
        metadata['from_tpt'] = pd.to_datetime(metadata['from_tpt'])
        merged = metadata.merge(
            self.summary_metadata[['cage_id', 'from_tpt', 'bodyweight']], 
            on=['cage_id', 'from_tpt'], 
            how='left'
        )
        
        valid_bw = merged['bodyweight'].notna().sum()
        print(f"Successfully matched bodyweight for {valid_bw}/{len(merged)} entries "
              f"({100*valid_bw/len(merged):.1f}%)")
        
        return merged


class EnhancedEmbeddingAnalyzer:
    """Enhanced analyzer with additional features and better visualizations"""
    
    def __init__(self, embeddings_path, metadata_path, output_dir="enhanced_analysis", 
                 bodyweight_files=None):
        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        self.output_dir = Path(output_dir)
        # ensure that the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        self.bodyweight_files = bodyweight_files
        
        # Create enhanced output structure
        self.create_output_structure()
        self.load_and_prepare_data()
    
    def create_output_structure(self):
        """Create comprehensive output directory structure"""
        dirs = [
            "activity_analysis", "regression_analysis", "classification_analysis",
            "feature_importance", "temporal_analysis", "strain_comparison",
            "interactive_plots", "statistical_reports", "model_diagnostics",
            "correlation_analysis", "dimensionality_analysis"
        ]
        
        self.output_dir.mkdir(exist_ok=True)
        for d in dirs:
            (self.output_dir / d).mkdir(exist_ok=True)
    
    def load_and_prepare_data(self):
        """Load and prepare all data with bodyweight integration"""
        print("Loading embeddings data...")
        embeddings_data = np.load(self.embeddings_path, allow_pickle=True).item()
        
        self.embeddings = embeddings_data['embeddings']
        self.frame_map = embeddings_data['frame_number_map']
        self.aggregation_info = embeddings_data['aggregation_info']
        
        print("Loading metadata...")
        self.metadata = pd.read_csv(self.metadata_path)
        
        # Add bodyweight if files provided
        if self.bodyweight_files:
            print("Integrating bodyweight data...")
            bw_handler = BodyweightHandler(
                self.bodyweight_files['bodyweight'],
                self.bodyweight_files['tissue'],
                self.bodyweight_files['summary']
            )
            self.metadata = bw_handler.add_bodyweight_to_metadata(self.metadata)
        
        # Create aligned dataset
        self.create_enhanced_aligned_dataset()
    
    def create_enhanced_aligned_dataset(self):
        """Create aligned dataset with additional features"""
        print("Creating enhanced aligned dataset...")
        
        # Prepare metadata keys
        if 'from_tpt' in self.metadata.columns:
            self.metadata['from_tpt'] = pd.to_datetime(self.metadata['from_tpt'])
            self.metadata['key'] = self.metadata.apply(
                lambda row: f"{row['cage_id']}_{row['from_tpt'].strftime('%Y-%m-%d')}", axis=1
            )
        
        aligned_embeddings = []
        aligned_metadata = []
        temporal_patterns = []  # Store temporal patterns for each sample
        
        frames_per_day = self.aggregation_info['num_frames']
        
        for idx, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc="Aligning data"):
            key = row['key'] if 'key' in row else str(idx)
            
            if key in self.frame_map:
                start, end = self.frame_map[key]
                if end > start and end <= self.embeddings.shape[0]:
                    seq_embeddings = self.embeddings[start:end]
                    
                    if seq_embeddings.shape[0] == frames_per_day and not np.isnan(seq_embeddings).any():
                        # Store both daily average and temporal patterns
                        daily_embedding = seq_embeddings.mean(axis=0)
                        
                        # Extract temporal features (hourly averages)
                        hourly_patterns = []
                        for hour in range(24):
                            hour_start = hour * 60
                            hour_end = min((hour + 1) * 60, frames_per_day)
                            if hour_end > hour_start:
                                hourly_avg = seq_embeddings[hour_start:hour_end].mean(axis=0)
                                hourly_patterns.append(hourly_avg)
                        
                        if len(hourly_patterns) == 24:
                            aligned_embeddings.append(daily_embedding)
                            aligned_metadata.append(row)
                            temporal_patterns.append(np.array(hourly_patterns))
        
        self.X = np.array(aligned_embeddings)
        self.metadata_aligned = pd.DataFrame(aligned_metadata).reset_index(drop=True)
        self.temporal_patterns = np.array(temporal_patterns)  # Shape: (n_samples, 24, n_dims)
        
        # Clean data
        valid_mask = self.metadata_aligned['avg_age_days_chunk_start'].notna()
        if 'strain' in self.metadata_aligned.columns:
            valid_mask &= self.metadata_aligned['strain'].notna()
        
        self.X = self.X[valid_mask]
        self.metadata_aligned = self.metadata_aligned[valid_mask].reset_index(drop=True)
        if len(self.temporal_patterns) > 0:
            self.temporal_patterns = self.temporal_patterns[valid_mask]
        
        print(f"Enhanced dataset: {self.X.shape[0]} samples, {self.X.shape[1]} dimensions")
        print(f"Temporal patterns: {self.temporal_patterns.shape if len(self.temporal_patterns) > 0 else 'None'}")
    
    def perform_comprehensive_temporal_analysis(self):
        """Perform comprehensive temporal pattern analysis"""
        print("Performing comprehensive temporal analysis...")
        
        if len(self.temporal_patterns) == 0:
            print("No temporal patterns available. Skipping temporal analysis.")
            return
        
        # 1. Circadian rhythm analysis
        self.analyze_circadian_rhythms()
        
        # 2. Temporal dimension importance
        self.analyze_temporal_dimension_importance()
        
        # 3. Age-related temporal changes
        self.analyze_age_temporal_relationships()
        
        # 4. Strain-specific temporal patterns
        self.analyze_strain_temporal_patterns()
    
    def analyze_circadian_rhythms(self):
        """Analyze circadian rhythms in different dimensions"""
        print("Analyzing circadian rhythms...")
        
        n_dims = min(16, self.X.shape[1])  # Analyze first 16 dimensions
        # n_dims = self.X.shape[1]  # Analyze all dimensions
        hours = np.arange(24)
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()
        
        for dim in range(n_dims):
            # Average across all samples for this dimension
            avg_pattern = self.temporal_patterns[:, :, dim].mean(axis=0)
            std_pattern = self.temporal_patterns[:, :, dim].std(axis=0)
            
            ax = axes[dim]
            ax.plot(hours, avg_pattern, 'b-', linewidth=2, label='Mean')
            ax.fill_between(hours, avg_pattern - std_pattern, avg_pattern + std_pattern,
                           alpha=0.3, color='blue', label='±1 SD')
            
            ax.set_title(f'Circadian Pattern - Dim {dim+1}')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Embedding Value')
            ax.set_xlim(0, 23)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "temporal_analysis" / "circadian_rhythms.png", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "temporal_analysis" / "circadian_rhythms.pdf", 
                   bbox_inches='tight')
        plt.close()
        
        # Save circadian data
        circadian_data = {
            'hours': hours.tolist(),
            'patterns': {
                f'dim_{dim+1}': {
                    'mean': self.temporal_patterns[:, :, dim].mean(axis=0).tolist(),
                    'std': self.temporal_patterns[:, :, dim].std(axis=0).tolist()
                }
                for dim in range(n_dims)
            }
        }
        
        with open(self.output_dir / "temporal_analysis" / "circadian_data.json", 'w') as f:
            # json.dump(circadian_data, f, indent=2)
            json.dump(convert_numpy_types(circadian_data), f, indent=2)
            
    def analyze_temporal_dimension_importance(self):
        """Analyze which dimensions show strongest temporal variation"""
        print("Analyzing temporal dimension importance...")
        
        n_dims = self.X.shape[1]
        
        # Calculate temporal variance for each dimension
        temporal_variances = []
        circadian_amplitudes = []
        
        for dim in range(n_dims):
            # Temporal variance: variance of hourly patterns within each sample
            sample_temporal_vars = [
                np.var(self.temporal_patterns[i, :, dim]) 
                for i in range(len(self.temporal_patterns))
            ]
            avg_temporal_var = np.mean(sample_temporal_vars)
            # temporal_variances.append(avg_temporal_var)
            temporal_variances.append(float(avg_temporal_var))  # Convert to Python float
            
            # Circadian amplitude: difference between max and min of average pattern
            avg_pattern = self.temporal_patterns[:, :, dim].mean(axis=0)
            circadian_amp = np.max(avg_pattern) - np.min(avg_pattern)
            # circadian_amplitudes.append(circadian_amp)
            circadian_amplitudes.append(float(circadian_amp))  # Convert to Python float
            
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Temporal variance by dimension
        dim_indices = np.arange(n_dims)
        bars1 = axes[0, 0].bar(dim_indices, temporal_variances)
        axes[0, 0].set_title('Temporal Variance by Dimension')
        axes[0, 0].set_xlabel('Dimension')
        axes[0, 0].set_ylabel('Average Temporal Variance')
        
        # Color bars by magnitude
        colors = plt.cm.viridis(np.linspace(0, 1, len(temporal_variances)))
        for bar, color in zip(bars1, colors):
            bar.set_color(color)
        
        # Circadian amplitude by dimension
        bars2 = axes[0, 1].bar(dim_indices, circadian_amplitudes)
        axes[0, 1].set_title('Circadian Amplitude by Dimension')
        axes[0, 1].set_xlabel('Dimension')
        axes[0, 1].set_ylabel('Circadian Amplitude')
        
        colors2 = plt.cm.plasma(np.linspace(0, 1, len(circadian_amplitudes)))
        for bar, color in zip(bars2, colors2):
            bar.set_color(color)
        
        # Top temporal dimensions detailed view
        top_temporal_dims = np.argsort(temporal_variances)[-6:]  # Top 6
        hours = np.arange(24)
        
        for i, dim in enumerate(top_temporal_dims[-6:]):  # Show top 6
            if i < 6:
                row, col = (1, 0) if i < 3 else (1, 1) if i == 3 else (1, 1)
                if i < 3:
                    ax = axes[1, 0]
                else:
                    ax = axes[1, 1] if 'ax2' not in locals() else plt.subplot(2, 2, 4)
                    
                if i == 3:  # First plot in bottom right
                    ax2 = axes[1, 1]
                    ax = ax2
                elif i > 3:
                    continue  # Skip for now, we'll handle differently
                
                avg_pattern = self.temporal_patterns[:, :, dim].mean(axis=0)
                ax.plot(hours, avg_pattern, label=f'Dim {dim+1}', linewidth=2)
        
        if 'ax2' in locals():
            ax2.set_title('Top Temporal Dimensions')
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Average Value')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "temporal_analysis" / "temporal_dimension_importance.png", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "temporal_analysis" / "temporal_dimension_importance.pdf", 
                   bbox_inches='tight')
        plt.close()
        
        # # Save temporal importance data
        # temporal_importance = {
        #     'temporal_variances': temporal_variances,
        #     'circadian_amplitudes': circadian_amplitudes,
        #     'top_temporal_dimensions': np.argsort(temporal_variances)[-10:].tolist(),
        #     'top_circadian_dimensions': np.argsort(circadian_amplitudes)[-10:].tolist()
        # }
        
        # Save temporal importance data - CONVERT ALL NUMPY TYPES
        temporal_importance = {
            'temporal_variances': [float(x) for x in temporal_variances],
            'circadian_amplitudes': [float(x) for x in circadian_amplitudes],
            'top_temporal_dimensions': [int(x) for x in np.argsort(temporal_variances)[-10:].tolist()],
            'top_circadian_dimensions': [int(x) for x in np.argsort(circadian_amplitudes)[-10:].tolist()]
        }
        
        with open(self.output_dir / "temporal_analysis" / "temporal_importance.json", 'w') as f:
            # json.dump(temporal_importance, f, indent=2)
            json.dump(convert_numpy_types(temporal_importance), f, indent=2)
    
    def analyze_age_temporal_relationships(self):
        """Analyze how temporal patterns change with age"""
        print("Analyzing age-temporal relationships...")
        
        ages = self.metadata_aligned['avg_age_days_chunk_start'].values
        
        # Bin ages into groups for analysis
        age_bins = np.percentile(ages, [0, 33, 66, 100])
        age_groups = ['Young', 'Middle', 'Old']
        age_labels = np.digitize(ages, age_bins[1:-1])
        
        n_dims = min(8, self.X.shape[1])
        hours = np.arange(24)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for dim in range(n_dims):
            ax = axes[dim]
            
            for age_group_idx, age_group_name in enumerate(age_groups):
                group_mask = age_labels == age_group_idx
                if np.sum(group_mask) > 5:  # Need sufficient samples
                    group_pattern = self.temporal_patterns[group_mask, :, dim].mean(axis=0)
                    ax.plot(hours, group_pattern, label=age_group_name, linewidth=2)
            
            ax.set_title(f'Age-Temporal Patterns - Dim {dim+1}')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Embedding Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 23)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "temporal_analysis" / "age_temporal_relationships.png", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "temporal_analysis" / "age_temporal_relationships.pdf", 
                   bbox_inches='tight')
        plt.close()
    
    def analyze_strain_temporal_patterns(self):
        """Analyze strain-specific temporal patterns"""
        print("Analyzing strain-specific temporal patterns...")
        
        if 'strain' not in self.metadata_aligned.columns:
            print("Strain information not available. Skipping strain temporal analysis.")
            return
        
        strains = self.metadata_aligned['strain'].values
        unique_strains = np.unique(strains)
        
        # Limit to top strains by sample count
        strain_counts = pd.Series(strains).value_counts()
        top_strains = strain_counts.head(6).index.tolist()
        
        n_dims = min(4, self.X.shape[1])
        hours = np.arange(24)
        
        fig, axes = plt.subplots(n_dims, 1, figsize=(16, 4*n_dims))
        
        for dim in range(n_dims):
            ax = axes[dim] if n_dims > 1 else axes
            
            for strain in top_strains:
                strain_mask = strains == strain
                if np.sum(strain_mask) > 5:
                    strain_pattern = self.temporal_patterns[strain_mask, :, dim].mean(axis=0)
                    ax.plot(hours, strain_pattern, label=strain, linewidth=2)
            
            ax.set_title(f'Strain Temporal Patterns - Dimension {dim+1}')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Embedding Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 23)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "temporal_analysis" / "strain_temporal_patterns.png", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "temporal_analysis" / "strain_temporal_patterns.pdf", 
                   bbox_inches='tight')
        plt.close()
    
    def create_comprehensive_correlation_analysis(self):
        """Create comprehensive correlation analysis"""
        print("Creating comprehensive correlation analysis...")
        
        # Prepare data for correlation analysis
        correlation_data = {
            'age': self.metadata_aligned['avg_age_days_chunk_start'].values
        }
        
        if 'bodyweight' in self.metadata_aligned.columns:
            correlation_data['bodyweight'] = self.metadata_aligned['bodyweight'].values
        
        if 'strain' in self.metadata_aligned.columns:
            le = LabelEncoder()
            correlation_data['strain_encoded'] = le.fit_transform(self.metadata_aligned['strain'].values)
        
        # Add embedding dimensions
        for i in range(self.X.shape[1]):
            correlation_data[f'dim_{i+1}'] = self.X[:, i]
        
        # Create correlation matrix
        corr_df = pd.DataFrame(correlation_data)
        correlation_matrix = corr_df.corr()
        
        # Create comprehensive correlation heatmap
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Full correlation matrix
        sns.heatmap(correlation_matrix, annot=False, cmap='RdBu_r', center=0, 
                   square=True, ax=axes[0, 0], cbar_kws={'label': 'Correlation'})
        axes[0, 0].set_title('Full Correlation Matrix')
        
        # Age correlations only
        age_corrs = correlation_matrix['age'].drop('age').sort_values(key=abs, ascending=False)
        top_age_corrs = age_corrs.head(20)
        
        bars = axes[0, 1].barh(range(len(top_age_corrs)), top_age_corrs.values)
        axes[0, 1].set_yticks(range(len(top_age_corrs)))
        axes[0, 1].set_yticklabels(top_age_corrs.index)
        axes[0, 1].set_title('Top Age Correlations')
        axes[0, 1].set_xlabel('Correlation with Age')
        
        # Color bars by correlation strength
        for bar, corr in zip(bars, top_age_corrs.values):
            if corr > 0:
                bar.set_color('red' if corr > 0.3 else 'orange' if corr > 0.1 else 'lightcoral')
            else:
                bar.set_color('blue' if corr < -0.3 else 'lightblue' if corr < -0.1 else 'lightsteelblue')
        
        # Bodyweight correlations (if available)
        if 'bodyweight' in correlation_data:
            bw_corrs = correlation_matrix['bodyweight'].drop(['bodyweight', 'age']).sort_values(key=abs, ascending=False)
            top_bw_corrs = bw_corrs.head(20)
            
            bars2 = axes[1, 0].barh(range(len(top_bw_corrs)), top_bw_corrs.values)
            axes[1, 0].set_yticks(range(len(top_bw_corrs)))
            axes[1, 0].set_yticklabels(top_bw_corrs.index)
            axes[1, 0].set_title('Top Bodyweight Correlations')
            axes[1, 0].set_xlabel('Correlation with Bodyweight')
            
            for bar, corr in zip(bars2, top_bw_corrs.values):
                if corr > 0:
                    bar.set_color('green' if corr > 0.3 else 'lightgreen')
                else:
                    bar.set_color('purple' if corr < -0.3 else 'plum')
        
        # Inter-dimension correlations
        dim_cols = [col for col in correlation_matrix.columns if col.startswith('dim_')]
        dim_corr_matrix = correlation_matrix.loc[dim_cols, dim_cols]
        
        sns.heatmap(dim_corr_matrix, annot=False, cmap='RdBu_r', center=0,
                   square=True, ax=axes[1, 1], cbar_kws={'label': 'Correlation'})
        axes[1, 1].set_title('Inter-Dimension Correlations')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlation_analysis" / "comprehensive_correlations.png", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "correlation_analysis" / "comprehensive_correlations.pdf", 
                   bbox_inches='tight')
        plt.close()
        
        # Save correlation data
        correlation_results = {
            'age_correlations': age_corrs.to_dict(),
            'top_age_predictive_dims': age_corrs.head(10).index.tolist(),
            'correlation_matrix_shape': correlation_matrix.shape
        }
        
        if 'bodyweight' in correlation_data:
            correlation_results['bodyweight_correlations'] = bw_corrs.to_dict()
            correlation_results['top_bodyweight_predictive_dims'] = bw_corrs.head(10).index.tolist()
        
        with open(self.output_dir / "correlation_analysis" / "correlation_results.json", 'w') as f:
            # json.dump(correlation_results, f, indent=2)
            json.dump(convert_numpy_types(correlation_results), f, indent=2)
    
    def perform_enhanced_feature_importance_analysis(self):
        """Perform enhanced feature importance analysis with multiple methods"""
        print("Performing enhanced feature importance analysis...")
        
        y_age = self.metadata_aligned['avg_age_days_chunk_start'].values
        
        # Multiple feature importance methods
        importance_methods = {}
        
        # 1. Univariate correlations
        univariate_importance = []
        for i in range(self.X.shape[1]):
            corr, pval = pearsonr(self.X[:, i], y_age)
            # univariate_importance.append({
            #     'dimension': f'dim_{i+1}',
            #     'correlation': abs(corr),
            #     'p_value': pval,
            #     'significant': pval < 0.05
            # })
            
            univariate_importance.append({
                'dimension': f'dim_{i+1}',
                'correlation': float(abs(corr)),  # Ensure Python float
                'p_value': float(pval),           # Ensure Python float
                'significant': bool(pval < 0.05)  # Ensure Python bool
            })
                    
        importance_methods['univariate'] = sorted(
            univariate_importance, key=lambda x: x['correlation'], reverse=True
        )
        
        # 2. Ridge regression coefficients
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_scaled, y_age)
        
        ridge_importance = [
            {
                'dimension': f'dim_{i+1}',
                'coefficient': abs(ridge.coef_[i]),
                'raw_coefficient': ridge.coef_[i]
            }
            for i in range(len(ridge.coef_))
        ]
        
        importance_methods['ridge'] = sorted(
            ridge_importance, key=lambda x: x['coefficient'], reverse=True
        )
        
        # 3. Random Forest feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(self.X, y_age)
        
        rf_importance = [
            {
                'dimension': f'dim_{i+1}',
                'importance': rf.feature_importances_[i]
            }
            for i in range(len(rf.feature_importances_))
        ]
        
        importance_methods['random_forest'] = sorted(
            rf_importance, key=lambda x: x['importance'], reverse=True
        )
        
        # Create comprehensive feature importance plot
        self.plot_feature_importance_comparison(importance_methods)
        
        # Save feature importance results
        with open(self.output_dir / "feature_importance" / "feature_importance_results.json", 'w') as f:
            # json.dump(importance_methods, f, indent=2)
            json.dump(convert_numpy_types(importance_methods), f, indent=2)
        
        return importance_methods
    
    def plot_feature_importance_comparison(self, importance_methods):
        """Create comprehensive feature importance comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Univariate correlations
        top_univariate = importance_methods['univariate'][:20]
        dims = [item['dimension'] for item in top_univariate]
        corrs = [item['correlation'] for item in top_univariate]
        
        bars1 = axes[0, 0].bar(range(len(dims)), corrs)
        axes[0, 0].set_title('Top Dimensions - Univariate Correlations with Age')
        axes[0, 0].set_xlabel('Dimension')
        axes[0, 0].set_ylabel('|Correlation|')
        axes[0, 0].set_xticks(range(len(dims)))
        axes[0, 0].set_xticklabels(dims, rotation=45)
        
        # Color by significance
        for i, (bar, item) in enumerate(zip(bars1, top_univariate)):
            if item['significant']:
                bar.set_color('red' if item['p_value'] < 0.01 else 'orange')
            else:
                bar.set_color('lightblue')
        
        # Ridge coefficients
        top_ridge = importance_methods['ridge'][:20]
        dims_ridge = [item['dimension'] for item in top_ridge]
        coefs = [item['coefficient'] for item in top_ridge]
        
        axes[0, 1].bar(range(len(dims_ridge)), coefs, color='purple', alpha=0.7)
        axes[0, 1].set_title('Top Dimensions - Ridge Regression Coefficients')
        axes[0, 1].set_xlabel('Dimension')
        axes[0, 1].set_ylabel('|Coefficient|')
        axes[0, 1].set_xticks(range(len(dims_ridge)))
        axes[0, 1].set_xticklabels(dims_ridge, rotation=45)
        
        # Random Forest importance
        top_rf = importance_methods['random_forest'][:20]
        dims_rf = [item['dimension'] for item in top_rf]
        importances = [item['importance'] for item in top_rf]
        
        axes[1, 0].bar(range(len(dims_rf)), importances, color='green', alpha=0.7)
        axes[1, 0].set_title('Top Dimensions - Random Forest Feature Importance')
        axes[1, 0].set_xlabel('Dimension')
        axes[1, 0].set_ylabel('Importance')
        axes[1, 0].set_xticks(range(len(dims_rf)))
        axes[1, 0].set_xticklabels(dims_rf, rotation=45)
        
        # Consensus ranking (average ranks across methods)
        all_dims = set()
        for method in importance_methods.values():
            all_dims.update([item['dimension'] for item in method])
        
        consensus_scores = {}
        for dim in all_dims:
            ranks = []
            for method_name, method_results in importance_methods.items():
                dim_ranks = {item['dimension']: i for i, item in enumerate(method_results)}
                ranks.append(dim_ranks.get(dim, len(method_results)))
            consensus_scores[dim] = np.mean(ranks)
        
        top_consensus = sorted(consensus_scores.items(), key=lambda x: x[1])[:20]
        dims_consensus = [item[0] for item in top_consensus]
        scores_consensus = [item[1] for item in top_consensus]
        
        axes[1, 1].bar(range(len(dims_consensus)), scores_consensus, color='navy', alpha=0.7)
        axes[1, 1].set_title('Consensus Ranking (Average Rank Across Methods)')
        axes[1, 1].set_xlabel('Dimension')
        axes[1, 1].set_ylabel('Average Rank')
        axes[1, 1].set_xticks(range(len(dims_consensus)))
        axes[1, 1].set_xticklabels(dims_consensus, rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_importance" / "feature_importance_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "feature_importance" / "feature_importance_comparison.pdf", 
                   bbox_inches='tight')
        plt.close()
    
    def create_interactive_exploration_dashboard(self):
        """Create interactive plots for data exploration"""
        print("Creating interactive exploration dashboard...")
        
        # Prepare data
        ages = self.metadata_aligned['avg_age_days_chunk_start'].values
        strains = self.metadata_aligned['strain'].values if 'strain' in self.metadata_aligned.columns else None
        cage_ids = self.metadata_aligned['cage_id'].values if 'cage_id' in self.metadata_aligned.columns else None
        
        # Perform PCA for visualization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        pca = PCA(n_components=min(10, self.X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        
        # Create interactive 3D PCA plot
        fig = go.Figure()
        
        if strains is not None:
            unique_strains = np.unique(strains)
            colors = px.colors.qualitative.Set3[:len(unique_strains)]
            
            for i, strain in enumerate(unique_strains):
                mask = strains == strain
                
                hover_text = []
                for j in np.where(mask)[0]:
                    text = f"Age: {ages[j]:.1f} days<br>Strain: {strain}"
                    if cage_ids is not None:
                        text += f"<br>Cage: {cage_ids[j]}"
                    hover_text.append(text)
                
                fig.add_trace(go.Scatter3d(
                    x=X_pca[mask, 0],
                    y=X_pca[mask, 1],
                    z=X_pca[mask, 2],
                    mode='markers',
                    name=strain,
                    marker=dict(
                        size=5,
                        color=colors[i % len(colors)],
                        opacity=0.7
                    ),
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>'
                ))
        else:
            # Color by age if no strain information
            fig.add_trace(go.Scatter3d(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                z=X_pca[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=ages,
                    colorscale='viridis',
                    opacity=0.7,
                    colorbar=dict(title="Age (days)")
                ),
                text=[f"Age: {age:.1f} days" for age in ages],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Interactive 3D PCA Visualization',
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)'
            ),
            width=1000,
            height=700
        )
        
        # Save interactive plot
        fig.write_html(self.output_dir / "interactive_plots" / "interactive_3d_pca.html")
        
        # Create 2D interactive plot with age slider
        fig_2d = go.Figure()
        
        # Add age slider functionality
        age_bins = np.percentile(ages, [0, 25, 50, 75, 100])
        age_labels = ['Young', 'Young-Mid', 'Mid-Old', 'Old']
        age_groups = np.digitize(ages, age_bins[1:-1])
        
        for i, label in enumerate(age_labels):
            mask = age_groups == i
            if np.sum(mask) > 0:
                fig_2d.add_trace(go.Scatter(
                    x=X_pca[mask, 0],
                    y=X_pca[mask, 1],
                    mode='markers',
                    name=label,
                    marker=dict(size=6, opacity=0.7),
                    text=[f"Age: {ages[j]:.1f}" for j in np.where(mask)[0]],
                    hovertemplate='%{text}<extra></extra>'
                ))
        
        fig_2d.update_layout(
            title='Interactive 2D PCA by Age Groups',
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
            width=800,
            height=600
        )
        
        fig_2d.write_html(self.output_dir / "interactive_plots" / "interactive_2d_pca_age.html")
        
        print("Interactive plots saved to interactive_plots directory")
    
    def run_enhanced_analysis(self):
        """Run the complete enhanced analysis pipeline"""
        print("Starting enhanced embedding analysis...")
        print("=" * 70)
        
        # 1. Comprehensive temporal analysis
        print("\n1. Comprehensive temporal analysis...")
        self.perform_comprehensive_temporal_analysis()
        
        # 2. Enhanced correlation analysis
        print("\n2. Enhanced correlation analysis...")
        self.create_comprehensive_correlation_analysis()
        
        # 3. Enhanced feature importance
        print("\n3. Enhanced feature importance analysis...")
        self.perform_enhanced_feature_importance_analysis()
        
        # 4. Interactive exploration dashboard
        print("\n4. Creating interactive exploration dashboard...")
        self.create_interactive_exploration_dashboard()
        
        print("\n" + "=" * 70)
        print("Enhanced analysis completed!")
        print(f"Results saved to: {self.output_dir}")

def main():
    # # File paths - UPDATE THESE TO YOUR ACTUAL PATHS
    # embeddings_path = "/scratch/bhole/dvc_data/smoothed/models_3days/embeddings_2850/test_1day_level1.npy"
    # metadata_path = "/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv"

    # # Optional bodyweight files
    # bodyweight_files = {
    #     'bodyweight': "/home/bhole/phenotypeFormattedSimplified_all_csv_data/BW_data.csv",
    #     'tissue': "/home/bhole/phenotypeFormattedSimplified_all_csv_data/TissueCollection_PP_data.csv",
    #     'summary': "/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv"
    # }
    
    # take the inputs as arguments
    parser = argparse.ArgumentParser(description="Run embedding analyses with optional bodyweight integration")
    parser.add_argument('--embeddings', type=str, required=True, help="Path to embeddings .npy file")
    parser.add_argument('--metadata', type=str, required=True, help="Path to metadata .csv file")
    parser.add_argument('--bodyweight', type=str, default=None, help="Path to bodyweight .csv file (optional)")
    parser.add_argument('--tissue', type=str, default=None, help="Path to tissue collection .csv file (optional)")
    parser.add_argument('--summary', type=str, default=None, help="Path to summary metadata .csv file (optional)")
    parser.add_argument('--output_dir', type=str, default="basic_analysis_results", help="Output directory for results")
    args = parser.parse_args()
    
    # # Run basic analysis
    # print("Running basic analysis...")
    # basic_analyzer = EmbeddingAnalyzer(
    #     embeddings_path=args.embeddings,
    #     metadata_path=args.metadata,
    #     output_dir=args.output_dir,
    #     max_workers=64
    # )
    # basic_analyzer.run_complete_analysis()
    
    # # Run enhanced analysis with bodyweight
    # print("\\nRunning enhanced analysis...")
    # enhanced_analyzer = EnhancedEmbeddingAnalyzer(
    #     embeddings_path=args.embeddings,
    #     metadata_path=args.metadata,
    #     output_dir=args.output_dir.replace("basic_analysis_results", "enhanced_analysis_results"),
    #     bodyweight_files={
    #         'bodyweight': args.bodyweight,
    #         'tissue': args.tissue,
    #         'summary': args.summary
    #     } if args.bodyweight and args.tissue and args.summary else None,
    # )
    # enhanced_analyzer.run_enhanced_analysis()
    
    bodyweight_files = argparse.Namespace(
        bodyweight=args.bodyweight,
        tissue=args.tissue,
        summary=args.summary
    ) if args.bodyweight and args.tissue and args.summary else None
    
    # Run basic analysis ONLY if no bodyweight files provided
    if args.analysis_type in ["basic", "both"] and not bodyweight_files:
        print("Running basic analysis...")
        basic_analyzer = EmbeddingAnalyzer(
            embeddings_path=args.embeddings,
            metadata_path=args.metadata,
            output_dir=args.output_dir,
            max_workers=64
        )
        basic_analyzer.run_complete_analysis()
    
    # Run enhanced analysis WITH bodyweight integration
    if bodyweight_files or args.analysis_type in ["enhanced", "both"]:
        print("\nRunning enhanced analysis with bodyweight integration...")
        enhanced_analyzer = EnhancedEmbeddingAnalyzer(
            embeddings_path=args.embeddings,
            metadata_path=args.metadata,
            output_dir=args.output_dir.replace("basic_analysis_results", "enhanced_analysis_results"),
            bodyweight_files=bodyweight_files
        )
        enhanced_analyzer.run_enhanced_analysis()
    
    print("\\nAll analyses completed!")
    print("Check the output directories for results:")
    print("- basic_analysis_results/")
    print("- enhanced_analysis_results/")

if __name__ == "__main__":
    main()




