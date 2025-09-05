#!/usr/bin/env python3
"""
Comprehensive embedding analysis framework for behavioral embeddings
Includes activity heatmaps, age regression, strain prediction, feature selection,
PCA analysis, clustering, and sliding window evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import os
import sys
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime, timedelta
import argparse
from pathlib import Path

warnings.filterwarnings("ignore")

class EmbeddingAnalyzer:
    def __init__(self, embeddings_path, metadata_path, output_dir="analysis_results", max_workers=8):
        """
        Initialize the embedding analyzer.
        
        Args:
            embeddings_path: Path to embeddings .npy file
            metadata_path: Path to metadata CSV file
            output_dir: Directory to save results
            max_workers: Number of threads for parallel processing
        """
        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        self.output_dir = Path(output_dir)
        # ensure that the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_workers = max_workers
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "heatmaps").mkdir(exist_ok=True)
        (self.output_dir / "regression").mkdir(exist_ok=True)
        (self.output_dir / "classification").mkdir(exist_ok=True)
        (self.output_dir / "feature_selection").mkdir(exist_ok=True)
        (self.output_dir / "pca_analysis").mkdir(exist_ok=True)
        (self.output_dir / "clustering").mkdir(exist_ok=True)
        (self.output_dir / "sliding_window").mkdir(exist_ok=True)
        (self.output_dir / "anomaly_detection").mkdir(exist_ok=True)
        
        # Load data
        self.load_data()
        self.results = {}
        
    def load_data(self):
        """Load embeddings and metadata"""
        print("Loading embeddings...")
        embeddings_data = np.load(self.embeddings_path, allow_pickle=True).item()
        
        self.embeddings = embeddings_data['embeddings']
        self.frame_map = embeddings_data['frame_number_map']
        self.aggregation_info = embeddings_data['aggregation_info']
        
        print(f"Embeddings shape: {self.embeddings.shape}")
        print(f"Frame map entries: {len(self.frame_map)}")
        
        # Load metadata
        print("Loading metadata...")
        self.metadata = pd.read_csv(self.metadata_path)
        
        # Create aligned dataset
        self.create_aligned_dataset()
        
    def create_aligned_dataset(self):
        """Create aligned embeddings and metadata using frame mapping"""
        print("Creating aligned dataset...")
        
        # Prepare metadata keys
        if 'from_tpt' in self.metadata.columns:
            self.metadata['from_tpt'] = pd.to_datetime(self.metadata['from_tpt'])
            self.metadata['key'] = self.metadata.apply(
                lambda row: f"{row['cage_id']}_{row['from_tpt'].strftime('%Y-%m-%d')}", axis=1
            )
        
        aligned_embeddings = []
        aligned_metadata = []
        
        for idx, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc="Aligning data"):
            key = row['key'] if 'key' in row else str(idx)
            
            if key in self.frame_map:
                start, end = self.frame_map[key]
                if end > start and end <= self.embeddings.shape[0]:
                    # Get embeddings for this sequence
                    seq_embeddings = self.embeddings[start:end]
                    
                    if seq_embeddings.shape[0] > 0 and not np.isnan(seq_embeddings).any():
                        # Average over time dimension to get daily representation
                        daily_embedding = seq_embeddings.mean(axis=0)
                        
                        if not np.isnan(daily_embedding).any():
                            aligned_embeddings.append(daily_embedding)
                            aligned_metadata.append(row)
        
        self.X = np.array(aligned_embeddings)
        self.metadata_aligned = pd.DataFrame(aligned_metadata).reset_index(drop=True)
        
        # Remove samples with missing target values
        valid_mask = self.metadata_aligned['avg_age_days_chunk_start'].notna()
        if 'strain' in self.metadata_aligned.columns:
            valid_mask &= self.metadata_aligned['strain'].notna()
        
        self.X = self.X[valid_mask]
        self.metadata_aligned = self.metadata_aligned[valid_mask].reset_index(drop=True)
        
        print(f"Final aligned dataset: {self.X.shape[0]} samples, {self.X.shape[1]} dimensions")
        
    # def create_activity_heatmaps(self, n_samples=1000):
    def create_activity_heatmaps(self, n_samples=100):
        """Create activity heatmaps for each dimension across 24 hours"""
        print("Creating activity heatmaps...")
        
        # Sample random sequences for visualization
        sample_keys = list(self.frame_map.keys())[:n_samples]
        # sample_keys = list(self.frame_map.keys())
        
        # Create heatmaps for each dimension
        n_dims = self.embeddings.shape[1]
        frames_per_day = self.aggregation_info['num_frames']
        
        for dim in tqdm(range(min(16, n_dims)), desc="Creating heatmaps"):  # Limit to first 16 dimensions
        # for dim in tqdm(range(n_dims), desc="Creating heatmaps"):  
            fig, ax = plt.subplots(figsize=(20, 12))
            
            heatmap_data = []
            sample_names = []
            
            for i, key in enumerate(sample_keys):
                if key in self.frame_map:
                    start, end = self.frame_map[key]
                    if end - start == frames_per_day:  # Full day
                        seq_data = self.embeddings[start:end, dim]
                        heatmap_data.append(seq_data)
                        sample_names.append(key)
            
            if heatmap_data:
                heatmap_array = np.array(heatmap_data)
                
                # Create time labels (hours)
                time_labels = [f"{i//60:02d}:{i%60:02d}" for i in range(0, frames_per_day, 60)]
                time_positions = list(range(0, frames_per_day, 60))
                
                sns.heatmap(heatmap_array, cmap='viridis', cbar_kws={'label': 'Embedding Value'},
                           xticklabels=False, yticklabels=sample_names[:20], ax=ax)
                
                ax.set_xticks(time_positions)
                ax.set_xticklabels(time_labels, rotation=45)
                ax.set_title(f'Activity Heatmap - Dimension {dim+1}\n24-hour Pattern Across Samples')
                ax.set_xlabel('Time of Day')
                ax.set_ylabel('Samples')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / "heatmaps" / f"activity_heatmap_dim_{dim+1}.png", 
                           dpi=300, bbox_inches='tight')
                plt.savefig(self.output_dir / "heatmaps" / f"activity_heatmap_dim_{dim+1}.pdf", 
                           bbox_inches='tight')
                plt.close()
        
        # Create average activity patterns across all dimensions
        self.create_average_activity_patterns(sample_keys[:n_samples])
        # self.create_average_activity_patterns(sample_keys)
        
    def create_average_activity_patterns(self, sample_keys):
        """Create average activity patterns across dimensions"""
        print("Creating average activity patterns...")
        
        frames_per_day = self.aggregation_info['num_frames']
        n_dims = min(8, self.embeddings.shape[1])  # Top 8 dimensions
        # n_dims = self.embeddings.shape[1] # Avg of all dimensions
        
        fig, axes = plt.subplots(n_dims, 1, figsize=(20, 4*n_dims))
        
        for dim in range(n_dims):
            all_sequences = []
            
            for key in sample_keys:
                if key in self.frame_map:
                    start, end = self.frame_map[key]
                    if end - start == frames_per_day:
                        seq_data = self.embeddings[start:end, dim]
                        all_sequences.append(seq_data)
            
            if all_sequences:
                # Calculate statistics
                sequences_array = np.array(all_sequences)
                mean_pattern = sequences_array.mean(axis=0)
                std_pattern = sequences_array.std(axis=0)
                
                # Time axis in hours
                time_hours = np.arange(frames_per_day) / 60.0
                
                ax = axes[dim] if n_dims > 1 else axes
                
                # Plot mean with confidence interval
                ax.plot(time_hours, mean_pattern, 'b-', linewidth=2, label='Mean')
                ax.fill_between(time_hours, mean_pattern - std_pattern, mean_pattern + std_pattern, 
                               alpha=0.3, color='blue', label='±1 SD')
                
                ax.set_title(f'Average 24-hour Pattern - Dimension {dim+1}')
                ax.set_xlabel('Time (hours)')
                ax.set_ylabel('Embedding Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 24)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "heatmaps" / "average_activity_patterns.png", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "heatmaps" / "average_activity_patterns.pdf", 
                   bbox_inches='tight')
        plt.close()
    
    def age_regression_analysis(self, include_bodyweight=True):
        """Perform age regression analysis with and without bodyweight"""
        print(f"Performing age regression analysis (bodyweight: {include_bodyweight})...")
        
        y_age = self.metadata_aligned['avg_age_days_chunk_start'].values
        groups = self.metadata_aligned['strain'].values
        
        # Prepare features
        X = self.X.copy()
        feature_names = [f'dim_{i+1}' for i in range(self.X.shape[1])]
        
        if include_bodyweight and 'bodyweight' in self.metadata_aligned.columns:
            bodyweight = self.metadata_aligned['bodyweight'].values.reshape(-1, 1)
            # Handle missing bodyweight values
            valid_bw_mask = ~np.isnan(bodyweight.flatten())
            if valid_bw_mask.sum() > len(valid_bw_mask) * 0.5:  # At least 50% valid
                X = X[valid_bw_mask]
                y_age = y_age[valid_bw_mask]
                groups = groups[valid_bw_mask]
                bodyweight = bodyweight[valid_bw_mask]
                
                # Normalize bodyweight
                bw_scaler = StandardScaler()
                bodyweight = bw_scaler.fit_transform(bodyweight)
                X = np.concatenate([X, bodyweight], axis=1)
                feature_names.append('bodyweight')
                print(f"Added bodyweight covariate. New shape: {X.shape}")
        
        suffix = "with_bodyweight" if include_bodyweight and 'bodyweight' in feature_names else "without_bodyweight"
        
        # Full model regression
        self.perform_full_model_regression(X, y_age, groups, suffix)
        
        # Dimension-wise regression
        self.perform_dimensionwise_regression(X, y_age, groups, feature_names, suffix)
        
        # Feature selection approaches
        self.perform_feature_selection_regression(X, y_age, groups, feature_names, suffix)
        
        # Sliding window analysis
        self.perform_sliding_window_regression(X, y_age, groups, suffix)
        
    def perform_full_model_regression(self, X, y, groups, suffix):
        """Perform full model regression with Leave-One-Group-Out validation"""
        print("Performing full model regression...")
        
        logo = LeaveOneGroupOut()
        # models = {
        #     'Ridge': Ridge(alpha=1.0),
        #     'Linear': LinearRegression(n_jobs=-1),
        #     'Lasso': Lasso(alpha=0.1)
        # }
        models = {
            'Linear': LinearRegression(n_jobs=-1)
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            
            rmse_scores = []
            r2_scores = []
            group_results = {}
            
            for train_idx, test_idx in tqdm(logo.split(X, y, groups), desc=f"LOGO {model_name}"):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                rmse_scores.append(rmse)
                r2_scores.append(r2)
                
                test_group = groups[test_idx][0]
                group_results[test_group] = {
                    'rmse': rmse,
                    'r2': r2,
                    'n_samples': len(y_test)
                }
            
            results[model_name] = {
                'mean_rmse': np.mean(rmse_scores),
                'std_rmse': np.std(rmse_scores),
                'mean_r2': np.mean(r2_scores),
                'std_r2': np.std(r2_scores),
                'group_results': group_results
            }
            
            print(f"{model_name} - RMSE: {np.mean(rmse_scores):.2f}±{np.std(rmse_scores):.2f}, "
                  f"R²: {np.mean(r2_scores):.3f}±{np.std(r2_scores):.3f}")
        
        # Save results
        with open(self.output_dir / "regression" / f"full_model_results_{suffix}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create visualization
        self.plot_regression_results(results, f"Full Model Age Regression ({suffix})", suffix)
        
    def perform_dimensionwise_regression(self, X, y, groups, feature_names, suffix):
        """Perform regression for each dimension individually"""
        print("Performing dimension-wise regression...")
        
        n_features = X.shape[1]
        logo = LeaveOneGroupOut()
        
        dimension_results = {}
        
        def analyze_dimension(dim):
            X_dim = X[:, [dim]]
            
            rmse_scores = []
            r2_scores = []
            
            for train_idx, test_idx in logo.split(X_dim, y, groups):
                X_train, X_test = X_dim[train_idx], X_dim[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train Ridge regression
                model = Ridge(alpha=1.0)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                rmse_scores.append(rmse)
                r2_scores.append(r2)
            
            return {
                'dimension': feature_names[dim],
                'mean_rmse': np.mean(rmse_scores),
                'std_rmse': np.std(rmse_scores),
                'mean_r2': np.mean(r2_scores),
                'std_r2': np.std(r2_scores)
            }
        
        # Parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_dim = {executor.submit(analyze_dimension, dim): dim 
                           for dim in range(n_features)}
            
            for future in tqdm(as_completed(future_to_dim), total=n_features, 
                             desc="Dimension-wise analysis"):
                dim = future_to_dim[future]
                result = future.result()
                dimension_results[dim] = result
        
        # Create comprehensive plots
        self.plot_dimensionwise_results(dimension_results, suffix, "Age Regression")
        
        # Save results
        with open(self.output_dir / "regression" / f"dimensionwise_results_{suffix}.json", 'w') as f:
            json.dump(dimension_results, f, indent=2, default=str)
    
    def perform_feature_selection_regression(self, X, y, groups, feature_names, suffix):
        """Perform regression with various feature selection methods"""
        print("Performing feature selection regression...")
        
        logo = LeaveOneGroupOut()
        
        # Define feature selection methods
        # selection_methods = {
        #     'SelectKBest_10': SelectKBest(score_func=f_regression, k=min(10, X.shape[1])),
        #     'SelectKBest_20': SelectKBest(score_func=f_regression, k=min(20, X.shape[1])),
        #     'Lasso_Selection': SelectFromModel(Lasso(alpha=0.1)),
        #     'RFE_10': RFE(Ridge(alpha=1.0), n_features_to_select=min(10, X.shape[1])),
        #     'RandomForest_Selection': SelectFromModel(RandomForestRegressor(n_estimators=50, random_state=42))
        # }
        selection_methods = {
            'Lasso_Selection': SelectFromModel(Lasso(alpha=0.1))
        }
        
        fs_results = {}
        
        for method_name, selector in selection_methods.items():
            print(f"Testing {method_name}...")
            
            rmse_scores = []
            r2_scores = []
            selected_features_counts = []
            
            for train_idx, test_idx in tqdm(logo.split(X, y, groups), desc=f"FS {method_name}"):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Feature selection
                X_train_selected = selector.fit_transform(X_train_scaled, y_train)
                X_test_selected = selector.transform(X_test_scaled)
                
                # Train model
                model = Ridge(alpha=1.0)
                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_test_selected)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                rmse_scores.append(rmse)
                r2_scores.append(r2)
                selected_features_counts.append(X_train_selected.shape[1])
            
            fs_results[method_name] = {
                'mean_rmse': np.mean(rmse_scores),
                'std_rmse': np.std(rmse_scores),
                'mean_r2': np.mean(r2_scores),
                'std_r2': np.std(r2_scores),
                'mean_features_selected': np.mean(selected_features_counts),
                'std_features_selected': np.std(selected_features_counts)
            }
        
        # Save results
        with open(self.output_dir / "feature_selection" / f"feature_selection_results_{suffix}.json", 'w') as f:
            json.dump(fs_results, f, indent=2, default=str)
        
        # Plot results
        self.plot_feature_selection_results(fs_results, suffix)
    
    def strain_classification_analysis(self):
        """Perform strain classification analysis"""
        print("Performing strain classification analysis...")
        
        if 'strain' not in self.metadata_aligned.columns:
            print("Strain information not available. Skipping strain classification.")
            return
        
        y_strain = self.metadata_aligned['strain'].values
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_strain)
        
        # Full model classification
        self.perform_full_model_classification(self.X, y_encoded, le.classes_)
        
        # Dimension-wise classification
        self.perform_dimensionwise_classification(self.X, y_encoded, le.classes_)
    
    def perform_full_model_classification(self, X, y, class_names):
        """Perform full model classification"""
        print("Performing full model classification...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # models = {
        #     'Logistic': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        #     'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        # }
        models = {
            'Logistic': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        }
        
        results = {}
        
        for model_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[model_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'n_test_samples': len(y_test)
            }
            
            print(f"{model_name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
        
        # Save results
        with open(self.output_dir / "classification" / "full_model_strain_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def perform_dimensionwise_classification(self, X, y, class_names):
        """Perform dimension-wise strain classification"""
        print("Performing dimension-wise classification...")
        
        n_features = X.shape[1]
        dimension_results = {}
        
        def analyze_dimension_classification(dim):
            X_dim = X[:, [dim]]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_dim, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            return {
                'dimension': f'dim_{dim+1}',
                'accuracy': accuracy,
                'f1_score': f1
            }
        
        # Parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_dim = {executor.submit(analyze_dimension_classification, dim): dim 
                           for dim in range(n_features)}
            
            for future in tqdm(as_completed(future_to_dim), total=n_features, 
                             desc="Dimension-wise classification"):
                dim = future_to_dim[future]
                result = future.result()
                dimension_results[dim] = result
        
        # Create plots
        self.plot_dimensionwise_results(dimension_results, "strain", "Strain Classification")
        
        # Save results
        with open(self.output_dir / "classification" / "dimensionwise_strain_results.json", 'w') as f:
            json.dump(dimension_results, f, indent=2, default=str)
    
    def perform_pca_analysis(self):
        """Perform PCA analysis to understand dimension relationships"""
        print("Performing PCA analysis...")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Save PCA results
        pca_results = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'components': pca.components_.tolist()
        }
        
        with open(self.output_dir / "pca_analysis" / "pca_results.json", 'w') as f:
            json.dump(pca_results, f, indent=2)
        
        # Create PCA plots
        self.create_pca_plots(pca, X_pca, X_scaled)
        
        # Analyze PC correlations with age and strain
        self.analyze_pc_correlations(X_pca, pca.explained_variance_ratio_)
    
    def create_pca_plots(self, pca, X_pca, X_scaled):
        """Create various PCA visualization plots"""
        n_components = min(20, len(pca.explained_variance_ratio_))
        
        # Scree plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Explained variance
        axes[0, 0].bar(range(1, n_components+1), pca.explained_variance_ratio_[:n_components])
        axes[0, 0].set_title('PCA Scree Plot')
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Explained Variance Ratio')
        
        # Cumulative variance
        axes[0, 1].plot(range(1, n_components+1), 
                       np.cumsum(pca.explained_variance_ratio_[:n_components]), 'bo-')
        axes[0, 1].axhline(y=0.8, color='r', linestyle='--', label='80% variance')
        axes[0, 1].axhline(y=0.95, color='orange', linestyle='--', label='95% variance')
        axes[0, 1].set_title('Cumulative Explained Variance')
        axes[0, 1].set_xlabel('Principal Component')
        axes[0, 1].set_ylabel('Cumulative Explained Variance')
        axes[0, 1].legend()
        
        # PC1 vs PC2 colored by age
        if len(self.metadata_aligned) > 0:
            ages = self.metadata_aligned['avg_age_days_chunk_start'].values
            scatter = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=ages, cmap='viridis', alpha=0.6)
            axes[1, 0].set_title('PC1 vs PC2 (colored by age)')
            axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.colorbar(scatter, ax=axes[1, 0], label='Age (days)')
            
            # PC1 vs PC3
            scatter2 = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 2], c=ages, cmap='plasma', alpha=0.6)
            axes[1, 1].set_title('PC1 vs PC3 (colored by age)')
            axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            axes[1, 1].set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)')
            plt.colorbar(scatter2, ax=axes[1, 1], label='Age (days)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "pca_analysis" / "pca_overview.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "pca_analysis" / "pca_overview.pdf", bbox_inches='tight')
        plt.close()
        
        # Component loadings heatmap
        n_dims_to_show = min(20, pca.components_.shape[1])
        n_pcs_to_show = min(10, pca.components_.shape[0])
        
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.heatmap(pca.components_[:n_pcs_to_show, :n_dims_to_show], 
                   cmap='RdBu_r', center=0, cbar_kws={'label': 'Loading'})
        ax.set_title('PCA Component Loadings')
        ax.set_xlabel('Original Dimensions')
        ax.set_ylabel('Principal Components')
        ax.set_xticklabels([f'Dim{i+1}' for i in range(n_dims_to_show)])
        ax.set_yticklabels([f'PC{i+1}' for i in range(n_pcs_to_show)])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "pca_analysis" / "component_loadings.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "pca_analysis" / "component_loadings.pdf", bbox_inches='tight')
        plt.close()
    
    def analyze_pc_correlations(self, X_pca, explained_variance):
        """Analyze correlations between PCs and target variables"""
        print("Analyzing PC correlations...")
        
        n_pcs = min(20, X_pca.shape[1])
        ages = self.metadata_aligned['avg_age_days_chunk_start'].values
        
        # Age correlations
        age_correlations = []
        age_pvalues = []
        
        for i in range(n_pcs):
            corr, pval = pearsonr(X_pca[:, i], ages)
            age_correlations.append(corr)
            age_pvalues.append(pval)
        
        # Strain associations (if available)
        strain_associations = []
        if 'strain' in self.metadata_aligned.columns:
            strains = self.metadata_aligned['strain'].values
            le = LabelEncoder()
            strain_encoded = le.fit_transform(strains)
            
            for i in range(n_pcs):
                corr, pval = pearsonr(X_pca[:, i], strain_encoded)
                strain_associations.append(abs(corr))  # Use absolute correlation
        
        # Create correlation plots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Age correlations
        bars1 = axes[0].bar(range(1, n_pcs+1), age_correlations)
        axes[0].set_title('PC Correlations with Age')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Correlation with Age')
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Color bars by significance
        for i, (bar, pval) in enumerate(zip(bars1, age_pvalues)):
            if pval < 0.01:
                bar.set_color('red')
            elif pval < 0.05:
                bar.set_color('orange')
            else:
                bar.set_color('lightblue')
        
        # Strain associations
        if strain_associations:
            axes[1].bar(range(1, n_pcs+1), strain_associations, color='green', alpha=0.7)
            axes[1].set_title('PC Associations with Strain')
            axes[1].set_xlabel('Principal Component')
            axes[1].set_ylabel('|Correlation| with Strain')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "pca_analysis" / "pc_correlations.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "pca_analysis" / "pc_correlations.pdf", bbox_inches='tight')
        plt.close()
        
        # Save correlation results
        correlation_results = {
            'age_correlations': age_correlations,
            'age_pvalues': age_pvalues,
            'strain_associations': strain_associations,
            'explained_variance': explained_variance[:n_pcs].tolist()
        }
        
        with open(self.output_dir / "pca_analysis" / "pc_correlations.json", 'w') as f:
            json.dump(correlation_results, f, indent=2)
    
    def perform_clustering_analysis(self):
        """Perform clustering analysis with multiple algorithms"""
        print("Performing clustering analysis...")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # Reduce dimensionality for better clustering
        pca = PCA(n_components=min(50, self.X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        
        # # Define clustering algorithms
        # clustering_algorithms = {
        #     'KMeans_5': KMeans(n_clusters=5, random_state=42),
        #     'KMeans_10': KMeans(n_clusters=10, random_state=42),
        #     'KMeans_15': KMeans(n_clusters=15, random_state=42),
        #     'Hierarchical_5': AgglomerativeClustering(n_clusters=5),
        #     'Hierarchical_10': AgglomerativeClustering(n_clusters=10),
        #     'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
        # }
        # Define clustering algorithms
        clustering_algorithms = {
            'KMeans_10': KMeans(n_clusters=10, random_state=42),
            'Hierarchical_10': AgglomerativeClustering(n_clusters=10),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
        }
        
        clustering_results = {}
        
        for alg_name, algorithm in clustering_algorithms.items():
            print(f"Running {alg_name}...")
            
            labels = algorithm.fit_predict(X_pca)
            
            # Calculate clustering metrics
            n_clusters = len(np.unique(labels))
            if n_clusters > 1:
                silhouette = silhouette_score(X_pca, labels)
            else:
                silhouette = -1
            
            clustering_results[alg_name] = {
                'labels': labels.tolist(),
                'n_clusters': n_clusters,
                'silhouette_score': silhouette
            }
            
            print(f"{alg_name}: {n_clusters} clusters, silhouette: {silhouette:.3f}")
        
        # Save clustering results
        with open(self.output_dir / "clustering" / "clustering_results.json", 'w') as f:
            json.dump(clustering_results, f, indent=2)
        
        # Create interactive clustering plots
        self.create_interactive_clustering_plots(X_pca, clustering_results)
        
        # Analyze cluster characteristics
        self.analyze_cluster_characteristics(X_scaled, clustering_results)
    
    def create_interactive_clustering_plots(self, X_pca, clustering_results):
        """Create interactive plots for clustering results"""
        print("Creating interactive clustering plots...")
        
        # Prepare data for plotting
        ages = self.metadata_aligned['avg_age_days_chunk_start'].values
        strains = self.metadata_aligned['strain'].values if 'strain' in self.metadata_aligned.columns else None
        cage_ids = self.metadata_aligned['cage_id'].values if 'cage_id' in self.metadata_aligned.columns else None
        
        # Create subplot for each clustering algorithm
        for alg_name, results in clustering_results.items():
            if results['n_clusters'] <= 1:
                continue
                
            labels = np.array(results['labels'])
            
            # Create interactive plot
            fig = go.Figure()
            
            # Add points colored by cluster
            unique_labels = np.unique(labels)
            colors = px.colors.qualitative.Set3[:len(unique_labels)]
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                
                hover_text = []
                for j in np.where(mask)[0]:
                    text = f"Age: {ages[j]:.1f} days"
                    if strains is not None:
                        text += f"<br>Strain: {strains[j]}"
                    if cage_ids is not None:
                        text += f"<br>Cage: {cage_ids[j]}"
                    text += f"<br>Cluster: {label}"
                    hover_text.append(text)
                
                fig.add_trace(go.Scatter(
                    x=X_pca[mask, 0],
                    y=X_pca[mask, 1],
                    mode='markers',
                    name=f'Cluster {label}',
                    marker=dict(color=colors[i % len(colors)], size=6),
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>'
                ))
            
            fig.update_layout(
                title=f'Interactive Clustering Plot - {alg_name}',
                xaxis_title='PC1',
                yaxis_title='PC2',
                width=800,
                height=600
            )
            
            # Save interactive plot
            fig.write_html(self.output_dir / "clustering" / f"interactive_clustering_{alg_name}.html")
        
        print("Interactive plots saved to clustering directory")
    
    def analyze_cluster_characteristics(self, X_scaled, clustering_results):
        """Analyze characteristics of clusters"""
        print("Analyzing cluster characteristics...")
        
        ages = self.metadata_aligned['avg_age_days_chunk_start'].values
        strains = self.metadata_aligned['strain'].values if 'strain' in self.metadata_aligned.columns else None
        
        cluster_analysis = {}
        
        for alg_name, results in clustering_results.items():
            if results['n_clusters'] <= 1:
                continue
                
            labels = np.array(results['labels'])
            unique_labels = np.unique(labels)
            
            cluster_stats = {}
            
            for label in unique_labels:
                mask = labels == label
                cluster_data = X_scaled[mask]
                cluster_ages = ages[mask]
                
                stats = {
                    'size': int(np.sum(mask)),
                    'age_mean': float(np.mean(cluster_ages)),
                    'age_std': float(np.std(cluster_ages)),
                    'age_range': [float(np.min(cluster_ages)), float(np.max(cluster_ages))],
                    'embedding_centroid': np.mean(cluster_data, axis=0).tolist()
                }
                
                if strains is not None:
                    cluster_strains = strains[mask]
                    strain_counts = pd.Series(cluster_strains).value_counts().to_dict()
                    stats['strain_distribution'] = strain_counts
                
                cluster_stats[f'cluster_{label}'] = stats
            
            cluster_analysis[alg_name] = cluster_stats
        
        # Save cluster analysis
        with open(self.output_dir / "clustering" / "cluster_characteristics.json", 'w') as f:
            json.dump(cluster_analysis, f, indent=2)
    
    def perform_sliding_window_regression(self, X, y, groups, suffix):
        """Perform sliding window regression analysis"""
        print(f"Performing sliding window regression ({suffix})...")
        
        ages = self.metadata_aligned['avg_age_days_chunk_start'].values
        window_sizes = [30, 60, 90]  # days
        window_step = 15  # days
        min_samples = 20
        
        sliding_results = {}
        
        for window_size in window_sizes:
            print(f"Processing window size: {window_size} days")
            
            min_age, max_age = np.min(ages), np.max(ages)
            window_starts = np.arange(min_age, max_age - window_size + 1, window_step)
            
            window_results = []
            
            for window_start in tqdm(window_starts, desc=f"Windows {window_size}d"):
                window_end = window_start + window_size
                window_mask = (ages >= window_start) & (ages < window_end)
                
                if np.sum(window_mask) < min_samples:
                    continue
                
                X_window = X[window_mask]
                y_window = y[window_mask]
                
                # Simple train-test split for window
                if len(X_window) < 10:
                    continue
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_window, y_window, test_size=0.3, random_state=42
                )
                
                # Scale and train
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model = Ridge(alpha=1.0)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                window_results.append({
                    'window_start': window_start,
                    'window_end': window_end,
                    'window_center': window_start + window_size / 2,
                    'rmse': rmse,
                    'r2': r2,
                    'n_samples': len(X_window)
                })
            
            sliding_results[f'window_{window_size}d'] = window_results
        
        # Create sliding window plots
        self.create_sliding_window_plots(sliding_results, suffix)
        
        # Save results
        with open(self.output_dir / "sliding_window" / f"sliding_window_results_{suffix}.json", 'w') as f:
            json.dump(sliding_results, f, indent=2)
    
    def create_sliding_window_plots(self, sliding_results, suffix):
        """Create sliding window visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (window_key, results) in enumerate(sliding_results.items()):
            color = colors[i % len(colors)]
            window_size = int(window_key.split('_')[1][:-1])  # Extract size from key
            
            if not results:
                continue
            
            centers = [r['window_center'] for r in results]
            rmses = [r['rmse'] for r in results]
            r2s = [r['r2'] for r in results]
            sample_counts = [r['n_samples'] for r in results]
            
            # RMSE plot
            axes[0, 0].plot(centers, rmses, 'o-', color=color, label=f'{window_size}d window', markersize=4)
            
            # R² plot
            axes[0, 1].plot(centers, r2s, 'o-', color=color, label=f'{window_size}d window', markersize=4)
            
            # Sample count plot
            axes[1, 0].plot(centers, sample_counts, 'o-', color=color, label=f'{window_size}d window', markersize=4)
        
        # Customize plots
        axes[0, 0].set_title('RMSE vs Age Window')
        axes[0, 0].set_xlabel('Window Center (days)')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('R² vs Age Window')
        axes[0, 1].set_xlabel('Window Center (days)')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Sample Count vs Age Window')
        axes[1, 0].set_xlabel('Window Center (days)')
        axes[1, 0].set_ylabel('Sample Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Empty subplot for future use
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "sliding_window" / f"sliding_window_{suffix}.png", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "sliding_window" / f"sliding_window_{suffix}.pdf", 
                   bbox_inches='tight')
        plt.close()
    
    def detect_anomalous_samples(self):
        """Detect samples that don't cluster well with their strain"""
        print("Detecting anomalous samples...")
        
        if 'strain' not in self.metadata_aligned.columns:
            print("Strain information not available. Skipping anomaly detection.")
            return
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        strains = self.metadata_aligned['strain'].values
        unique_strains = np.unique(strains)
        
        anomalous_samples = []
        
        for strain in unique_strains:
            strain_mask = strains == strain
            strain_data = X_scaled[strain_mask]
            
            if len(strain_data) < 5:  # Need sufficient samples
                continue
            
            # Calculate centroid for this strain
            strain_centroid = np.mean(strain_data, axis=0)
            
            # Calculate distances from centroid
            distances = np.linalg.norm(strain_data - strain_centroid, axis=1)
            
            # Find outliers (beyond 2.5 standard deviations)
            threshold = np.mean(distances) + 2.5 * np.std(distances)
            outlier_mask = distances > threshold
            
            if np.any(outlier_mask):
                strain_indices = np.where(strain_mask)[0]
                outlier_indices = strain_indices[outlier_mask]
                
                for idx in outlier_indices:
                    anomalous_samples.append({
                        'sample_index': int(idx),
                        'strain': strain,
                        'cage_id': self.metadata_aligned.iloc[idx]['cage_id'] if 'cage_id' in self.metadata_aligned.columns else 'unknown',
                        'age': float(self.metadata_aligned.iloc[idx]['avg_age_days_chunk_start']),
                        'distance_from_centroid': float(distances[outlier_mask][0]),  # First outlier distance
                        'date': self.metadata_aligned.iloc[idx]['from_tpt'] if 'from_tpt' in self.metadata_aligned.columns else 'unknown'
                    })
        
        # Save anomalous samples
        anomaly_results = {
            'anomalous_samples': anomalous_samples,
            'total_samples': len(self.X),
            'n_anomalous': len(anomalous_samples),
            'anomaly_rate': len(anomalous_samples) / len(self.X)
        }
        
        with open(self.output_dir / "anomaly_detection" / "anomalous_samples.json", 'w') as f:
            json.dump(anomaly_results, f, indent=2, default=str)
        
        # Save anomalous sample details to text file
        with open(self.output_dir / "anomaly_detection" / "anomalous_samples.txt", 'w') as f:
            f.write("Anomalous Samples (samples that don't cluster well with their strain)\n")
            f.write("=" * 70 + "\n\n")
            
            for sample in anomalous_samples:
                f.write(f"Sample Index: {sample['sample_index']}\n")
                f.write(f"Strain: {sample['strain']}\n")
                f.write(f"Cage ID: {sample['cage_id']}\n")
                f.write(f"Age: {sample['age']:.1f} days\n")
                f.write(f"Date: {sample['date']}\n")
                f.write(f"Distance from strain centroid: {sample['distance_from_centroid']:.3f}\n")
                f.write("-" * 50 + "\n")
        
        print(f"Found {len(anomalous_samples)} anomalous samples ({100*len(anomalous_samples)/len(self.X):.1f}%)")
    
    def plot_regression_results(self, results, title, suffix):
        """Create comprehensive regression results plots"""
        # Extract strain-wise results
        strain_data = {}
        for model_name, model_results in results.items():
            for strain, strain_results in model_results['group_results'].items():
                if strain not in strain_data:
                    strain_data[strain] = {}
                strain_data[strain][model_name] = strain_results
        
        # Create wide figure for many strains
        n_strains = len(strain_data)
        fig_width = max(16, n_strains * 0.8)
        
        fig, axes = plt.subplots(2, 1, figsize=(fig_width, 12))
        
        strains = sorted(strain_data.keys())
        models = list(results.keys())
        
        # RMSE plot
        x_pos = np.arange(len(strains))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            rmse_values = [strain_data[strain].get(model, {}).get('rmse', np.nan) for strain in strains]
            axes[0].bar(x_pos + i * width, rmse_values, width, label=model, alpha=0.8)
        
        axes[0].set_title(f'{title} - RMSE by Strain')
        axes[0].set_xlabel('Strain')
        axes[0].set_ylabel('RMSE')
        axes[0].set_xticks(x_pos + width * (len(models) - 1) / 2)
        axes[0].set_xticklabels(strains, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # R² plot
        for i, model in enumerate(models):
            r2_values = [strain_data[strain].get(model, {}).get('r2', np.nan) for strain in strains]
            axes[1].bar(x_pos + i * width, r2_values, width, label=model, alpha=0.8)
        
        axes[1].set_title(f'{title} - R² by Strain')
        axes[1].set_xlabel('Strain')
        axes[1].set_ylabel('R²')
        axes[1].set_xticks(x_pos + width * (len(models) - 1) / 2)
        axes[1].set_xticklabels(strains, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "regression" / f"strain_regression_results_{suffix}.png", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "regression" / f"strain_regression_results_{suffix}.pdf", 
                   bbox_inches='tight')
        plt.close()
    
    # def plot_dimensionwise_results(self, dimension_results, suffix, analysis_type):
    #     """Create plots for dimension-wise analysis results"""
    #     dimensions = sorted(dimension_results.keys())
        
    #     if analysis_type == "Age Regression":
    #         metric_key = 'mean_rmse'
    #         metric_name = 'RMSE'
    #     else:  # Strain Classification
    #         metric_key = 'accuracy'
    #         metric_name = 'Accuracy'
        
    #     # Sort dimensions by performance
    #     sorted_dims = sorted(dimensions, 
    #                        key=lambda x: dimension_results[x].get(metric_key, np.inf), 
    #                        reverse=(analysis_type != "Age Regression"))
        
    #     values = [dimension_results[dim][metric_key] for dim in sorted_dims]
    #     dim_names = [dimension_results[dim]['dimension'] for dim in sorted_dims]
        
    #     # Create wide figure
    #     fig, ax = plt.subplots(figsize=(max(16, len(dimensions) * 0.6), 8))
        
    #     bars = ax.bar(range(len(sorted_dims)), values)
        
    #     # Color bars by performance
    #     if analysis_type == "Age Regression":
    #         # For RMSE, lower is better
    #         colors = plt.cm.RdYlGn_r(np.linspace(0.3, 1.0, len(values)))
    #     else:
    #         # For accuracy, higher is better
    #         colors = plt.cm.RdYlGn(np.linspace(0.3, 1.0, len(values)))
        
    #     for bar, color in zip(bars, colors):
    #         bar.set_color(color)
        
    #     ax.set_title(f'{analysis_type} - {metric_name} by Dimension ({suffix})')
    #     ax.set_xlabel('Dimension')
    #     ax.set_ylabel(metric_name)
    #     ax.set_xticks(range(len(sorted_dims)))
    #     ax.set_xticklabels(dim_names, rotation=45, ha='right')
    #     ax.grid(True, alpha=0.3)
        
    #     # Add value labels on bars
    #     for i, (bar, value) in enumerate(zip(bars, values)):
    #         height = bar.get_height()
    #         ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
    #                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
    #     plt.tight_layout()
        
    #     analysis_dir = "regression" if analysis_type == "Age Regression" else "classification"
    #     plt.savefig(self.output_dir / analysis_dir / f"dimensionwise_{suffix}.png", 
    #                dpi=300, bbox_inches='tight')
    #     plt.savefig(self.output_dir / analysis_dir / f"dimensionwise_{suffix}.pdf", 
    #                bbox_inches='tight')
    #     plt.close()
    
    def plot_dimensionwise_results(self, dimension_results, suffix, analysis_type):
        """Create plots for dimension-wise analysis results with size limits"""
        dimensions = sorted(dimension_results.keys())
        
        if analysis_type == "Age Regression":
            metric_key = 'mean_rmse'
            metric_name = 'RMSE'
        else:  # Strain Classification
            metric_key = 'accuracy'
            metric_name = 'Accuracy'
        
        # Sort dimensions by performance
        sorted_dims = sorted(dimensions, 
                        key=lambda x: dimension_results[x].get(metric_key, np.inf), 
                        reverse=(analysis_type != "Age Regression"))
        
        values = [dimension_results[dim][metric_key] for dim in sorted_dims]
        dim_names = [dimension_results[dim]['dimension'] for dim in sorted_dims]
        
        # LIMIT THE NUMBER OF DIMENSIONS TO PREVENT OVERSIZED PLOTS
        MAX_DIMS_PER_PLOT = 100  # Adjust this based on your needs
        
        if len(sorted_dims) > MAX_DIMS_PER_PLOT:
            print(f"Too many dimensions ({len(sorted_dims)}). Creating plots for top {MAX_DIMS_PER_PLOT} dimensions.")
            sorted_dims = sorted_dims[:MAX_DIMS_PER_PLOT]
            values = values[:MAX_DIMS_PER_PLOT]
            dim_names = dim_names[:MAX_DIMS_PER_PLOT]
        
        # Calculate appropriate figure width (max 50 inches)
        fig_width = min(max(16, len(sorted_dims) * 0.3), 50)
        fig_height = 8
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        bars = ax.bar(range(len(sorted_dims)), values)
        
        # Color bars by performance
        if analysis_type == "Age Regression":
            colors = plt.cm.RdYlGn_r(np.linspace(0.3, 1.0, len(values)))
        else:
            colors = plt.cm.RdYlGn(np.linspace(0.3, 1.0, len(values)))
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_title(f'{analysis_type} - {metric_name} by Dimension ({suffix})')
        ax.set_xlabel('Dimension')
        ax.set_ylabel(metric_name)
        ax.set_xticks(range(len(sorted_dims)))
        ax.set_xticklabels(dim_names, rotation=45, ha='right', fontsize=max(6, 12 - len(sorted_dims)//20))
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars (only for reasonable number of bars)
        if len(sorted_dims) <= 50:
            for i, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        analysis_dir = "regression" if analysis_type == "Age Regression" else "classification"
        
        try:
            plt.savefig(self.output_dir / analysis_dir / f"dimensionwise_{suffix}.png", 
                    dpi=150, bbox_inches='tight')  # Reduced DPI to 150
            plt.savefig(self.output_dir / analysis_dir / f"dimensionwise_{suffix}.pdf", 
                    bbox_inches='tight')
        except ValueError as e:
            print(f"Warning: Could not save plot due to size constraints: {e}")
            # Save a simplified version with even fewer dimensions
            if len(sorted_dims) > 20:
                self.create_simplified_dimension_plot(dimension_results, suffix, analysis_type, top_n=20)
        
        plt.close()

    def create_simplified_dimension_plot(self, dimension_results, suffix, analysis_type, top_n=20):
        """Create a simplified plot with only top N dimensions"""
        dimensions = sorted(dimension_results.keys())
        
        if analysis_type == "Age Regression":
            metric_key = 'mean_rmse'
            metric_name = 'RMSE'
        else:
            metric_key = 'accuracy'
            metric_name = 'Accuracy'
        
        # Get top N dimensions
        sorted_dims = sorted(dimensions, 
                        key=lambda x: dimension_results[x].get(metric_key, np.inf), 
                        reverse=(analysis_type != "Age Regression"))[:top_n]
        
        values = [dimension_results[dim][metric_key] for dim in sorted_dims]
        dim_names = [dimension_results[dim]['dimension'] for dim in sorted_dims]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.bar(range(len(sorted_dims)), values)
        
        # Color bars
        if analysis_type == "Age Regression":
            colors = plt.cm.RdYlGn_r(np.linspace(0.3, 1.0, len(values)))
        else:
            colors = plt.cm.RdYlGn(np.linspace(0.3, 1.0, len(values)))
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_title(f'Top {top_n} Dimensions - {analysis_type} {metric_name} ({suffix})')
        ax.set_xlabel('Dimension')
        ax.set_ylabel(metric_name)
        ax.set_xticks(range(len(sorted_dims)))
        ax.set_xticklabels(dim_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        analysis_dir = "regression" if analysis_type == "Age Regression" else "classification"
        plt.savefig(self.output_dir / analysis_dir / f"dimensionwise_top{top_n}_{suffix}.png", 
                dpi=150, bbox_inches='tight')
        plt.savefig(self.output_dir / analysis_dir / f"dimensionwise_top{top_n}_{suffix}.pdf", 
                bbox_inches='tight')
        plt.close()
    
    def plot_feature_selection_results(self, fs_results, suffix):
        """Create plots for feature selection results"""
        methods = list(fs_results.keys())
        rmse_values = [fs_results[method]['mean_rmse'] for method in methods]
        r2_values = [fs_results[method]['mean_r2'] for method in methods]
        n_features = [fs_results[method]['mean_features_selected'] for method in methods]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # RMSE comparison
        axes[0].bar(methods, rmse_values, alpha=0.8, color='red')
        axes[0].set_title(f'Feature Selection - RMSE Comparison ({suffix})')
        axes[0].set_ylabel('RMSE')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # R² comparison
        axes[1].bar(methods, r2_values, alpha=0.8, color='blue')
        axes[1].set_title(f'Feature Selection - R² Comparison ({suffix})')
        axes[1].set_ylabel('R²')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Number of features selected
        axes[2].bar(methods, n_features, alpha=0.8, color='green')
        axes[2].set_title(f'Feature Selection - Features Selected ({suffix})')
        axes[2].set_ylabel('Number of Features')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_selection" / f"feature_selection_{suffix}.png", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "feature_selection" / f"feature_selection_{suffix}.pdf", 
                   bbox_inches='tight')
        plt.close()
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting complete embedding analysis...")
        print("=" * 60)
        
        # 1. Activity heatmaps
        print("\n1. Creating activity heatmaps...")
        self.create_activity_heatmaps()
        
        # 2. Age regression analysis
        print("\n2. Age regression analysis...")
        self.age_regression_analysis(include_bodyweight=True)
        self.age_regression_analysis(include_bodyweight=False)
        
        # 3. Strain classification analysis
        print("\n3. Strain classification analysis...")
        self.strain_classification_analysis()
        
        # 4. PCA analysis
        print("\n4. PCA analysis...")
        self.perform_pca_analysis()
        
        # 5. Clustering analysis
        print("\n5. Clustering analysis...")
        self.perform_clustering_analysis()
        
        # 6. Anomaly detection
        print("\n6. Anomaly detection...")
        self.detect_anomalous_samples()
        
        print("\n" + "=" * 60)
        print("Complete analysis finished!")
        print(f"Results saved to: {self.output_dir}")
        
        # Create summary report
        self.create_summary_report()
    
    def create_summary_report(self):
        """Create a summary report of all analyses"""
        print("Creating summary report...")
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'n_samples': int(self.X.shape[0]),
                'n_dimensions': int(self.X.shape[1]),
                'n_strains': len(np.unique(self.metadata_aligned['strain'])) if 'strain' in self.metadata_aligned.columns else 0,
                'age_range': [float(self.metadata_aligned['avg_age_days_chunk_start'].min()), 
                            float(self.metadata_aligned['avg_age_days_chunk_start'].max())],
                'embedding_info': self.aggregation_info
            },
            'analyses_completed': [
                'activity_heatmaps',
                'age_regression_with_bodyweight',
                'age_regression_without_bodyweight',
                'strain_classification',
                'pca_analysis',
                'clustering_analysis',
                'anomaly_detection'
            ]
        }
        
        with open(self.output_dir / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Summary report saved to: {self.output_dir / 'analysis_summary.json'}")


def main():
    """Main function to run the analysis"""
    parser = argparse.ArgumentParser(description='Comprehensive Embedding Analysis')
    parser.add_argument('embeddings_path', help='Path to embeddings .npy file')
    parser.add_argument('metadata_path', help='Path to metadata CSV file')
    parser.add_argument('--output_dir', default='analysis_results', 
                       help='Output directory for results')
    parser.add_argument('--max_workers', type=int, default=os.cpu_count()-4, 
                       help='Maximum number of worker threads')
    parser.add_argument('--n_heatmap_samples', type=int, default=100,
                       help='Number of samples for activity heatmaps')
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = EmbeddingAnalyzer(
        embeddings_path=args.embeddings_path,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()