#!/usr/bin/env python3
"""
Temporal Sub-Aggregation Analysis for Behavioral Embeddings
Analyzes age regression patterns across different time periods within each day:
- Day/Night split (12h each): 7am-7pm vs 7pm-7am
- 4-hour blocks: 6 periods per day
- Hourly blocks: 24 periods per day

Based on the embedding analysis framework discussed earlier.
"""

import argparse
import warnings
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
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

class TemporalSubAggregationAnalyzer:
    """Analyzes age regression across different temporal sub-aggregations"""
    
    def __init__(self, embeddings_path, metadata_path, output_dir="temporal_subaggregation_results", 
                 max_workers=8, bodyweight_files=None):
        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.bodyweight_files = bodyweight_files
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        for subdir in ["day_night", "four_hour", "hourly", "comparison", "mixed_effects", "visualizations"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Define temporal aggregations
        self.temporal_configs = {
            'day_night': {
                'blocks': [
                    {'name': 'day', 'start_hour': 7, 'end_hour': 19, 'start_min': 420, 'end_min': 1140},  # 7am-7pm
                    {'name': 'night', 'start_hour': 19, 'end_hour': 7, 'start_min': 1140, 'end_min': 1440+420}  # 7pm-7am (next day)
                ]
            },
            'four_hour': {
                'blocks': [
                    {'name': '4h_1', 'start_hour': 7, 'end_hour': 11, 'start_min': 420, 'end_min': 660},   # 7am-11am
                    {'name': '4h_2', 'start_hour': 11, 'end_hour': 15, 'start_min': 660, 'end_min': 900},   # 11am-3pm
                    {'name': '4h_3', 'start_hour': 15, 'end_hour': 19, 'start_min': 900, 'end_min': 1140},  # 3pm-7pm
                    {'name': '4h_4', 'start_hour': 19, 'end_hour': 23, 'start_min': 1140, 'end_min': 1380}, # 7pm-11pm
                    {'name': '4h_5', 'start_hour': 23, 'end_hour': 3, 'start_min': 1380, 'end_min': 1440+180}, # 11pm-3am
                    {'name': '4h_6', 'start_hour': 3, 'end_hour': 7, 'start_min': 180, 'end_min': 420}     # 3am-7am
                ]
            },
            'hourly': {
                'blocks': [
                    {'name': f'hour_{h:02d}', 'start_hour': h, 'end_hour': (h+1)%24, 
                     'start_min': h*60, 'end_min': (h+1)*60 if h < 23 else 1440}
                    for h in range(24)
                ]
            }
        }
        
        self.results = {}
        self.load_data()
    
    def load_data(self):
        """Load embeddings and metadata"""
        print("Loading embeddings...")
        embeddings_data = np.load(self.embeddings_path, allow_pickle=True).item()
        
        self.embeddings = embeddings_data['embeddings']
        self.frame_map = embeddings_data['frame_number_map']
        self.aggregation_info = embeddings_data['aggregation_info']
        
        print(f"Embeddings shape: {self.embeddings.shape}")
        print(f"Frame map entries: {len(self.frame_map)}")
        print(f"Aggregation info: {self.aggregation_info}")
        
        # Load metadata
        print("Loading metadata...")
        self.metadata = pd.read_csv(self.metadata_path)
        
        # Add bodyweight if available
        if self.bodyweight_files:
            print("Loading bodyweight data...")
            # Use the bodyweight integration logic from previous scripts
            self.metadata = self.add_bodyweight_to_metadata()
        
        # Create aligned dataset
        self.create_temporal_aligned_dataset()
    
    def add_bodyweight_to_metadata(self):
        """Add bodyweight data to metadata (simplified version)"""
        # This would use the BodyweightHandler from the previous script
        # For now, return metadata as-is
        return self.metadata
    
    def create_temporal_aligned_dataset(self):
        """Create temporally aligned dataset with sub-aggregations"""
        print("Creating temporal sub-aggregation dataset...")
        
        # Prepare metadata keys
        if 'from_tpt' in self.metadata.columns:
            self.metadata['from_tpt'] = pd.to_datetime(self.metadata['from_tpt'])
            self.metadata['key'] = self.metadata.apply(
                lambda row: f"{row['cage_id']}_{row['from_tpt'].strftime('%Y-%m-%d')}", axis=1
            )
        
        # Get expected frames per day
        expected_frames = self.aggregation_info.get('num_frames', 1440)
        print(f"Expected frames per sequence: {expected_frames}")
        
        # Create temporal sub-aggregations
        temporal_data = {}
        aligned_metadata = []
        
        for idx, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc="Processing sequences"):
            key = row['key'] if 'key' in row else str(idx)
            
            if key in self.frame_map:
                start, end = self.frame_map[key]
                if end > start and end <= self.embeddings.shape[0]:
                    seq_embeddings = self.embeddings[start:end]
                    
                    if seq_embeddings.shape[0] == expected_frames and not np.isnan(seq_embeddings).any():
                        # Extract temporal sub-blocks
                        temporal_blocks = self.extract_temporal_blocks(seq_embeddings, expected_frames)
                        
                        if temporal_blocks:
                            temporal_data[key] = temporal_blocks
                            aligned_metadata.append(row)
        
        self.temporal_data = temporal_data
        self.metadata_aligned = pd.DataFrame(aligned_metadata).reset_index(drop=True)
        
        # Remove samples with missing target values
        valid_mask = self.metadata_aligned['avg_age_days_chunk_start'].notna()
        if 'strain' in self.metadata_aligned.columns:
            valid_mask &= self.metadata_aligned['strain'].notna()
        
        # Filter temporal data and metadata
        valid_keys = set(self.metadata_aligned[valid_mask]['key'].values)
        self.temporal_data = {k: v for k, v in self.temporal_data.items() if k in valid_keys}
        self.metadata_aligned = self.metadata_aligned[valid_mask].reset_index(drop=True)
        
        print(f"Final aligned dataset: {len(self.temporal_data)} sequences")
        
        # Create aggregation-specific datasets
        self.create_aggregation_datasets()
    
    def extract_temporal_blocks(self, sequence_embeddings, expected_frames):
        """Extract temporal blocks from a single sequence"""
        if expected_frames != 1440:
            # Scale time blocks proportionally for non-1440 frame sequences
            scale_factor = expected_frames / 1440
        else:
            scale_factor = 1.0
        
        temporal_blocks = {}
        
        for agg_name, config in self.temporal_configs.items():
            temporal_blocks[agg_name] = {}
            
            for block in config['blocks']:
                start_min = int(block['start_min'] * scale_factor)
                end_min = int(block['end_min'] * scale_factor)
                
                # Handle wraparound for night periods
                if end_min > expected_frames:
                    # Split across day boundary
                    part1 = sequence_embeddings[start_min:expected_frames]
                    part2 = sequence_embeddings[0:(end_min - expected_frames)]
                    if len(part1) > 0 and len(part2) > 0:
                        block_embeddings = np.concatenate([part1, part2], axis=0)
                    elif len(part1) > 0:
                        block_embeddings = part1
                    else:
                        block_embeddings = part2
                else:
                    block_embeddings = sequence_embeddings[start_min:end_min]
                
                if len(block_embeddings) > 0:
                    # Aggregate embeddings for this time block
                    aggregated_embedding = block_embeddings.mean(axis=0)
                    temporal_blocks[agg_name][block['name']] = aggregated_embedding
        
        return temporal_blocks
    
    def create_aggregation_datasets(self):
        """Create datasets for each temporal aggregation level"""
        print("Creating aggregation-specific datasets...")
        
        self.aggregation_datasets = {}
        
        for agg_name in self.temporal_configs.keys():
            print(f"Processing {agg_name} aggregation...")
            
            # Get all block names for this aggregation
            block_names = [block['name'] for block in self.temporal_configs[agg_name]['blocks']]
            
            # Create dataset for each time block
            for block_name in block_names:
                X_block = []
                metadata_block = []
                
                for idx, row in self.metadata_aligned.iterrows():
                    key = row['key']
                    if (key in self.temporal_data and 
                        agg_name in self.temporal_data[key] and 
                        block_name in self.temporal_data[key][agg_name]):
                        
                        embedding = self.temporal_data[key][agg_name][block_name]
                        X_block.append(embedding)
                        metadata_block.append(row)
                
                if len(X_block) > 0:
                    dataset_key = f"{agg_name}_{block_name}"
                    self.aggregation_datasets[dataset_key] = {
                        'X': np.array(X_block),
                        'metadata': pd.DataFrame(metadata_block).reset_index(drop=True),
                        'agg_name': agg_name,
                        'block_name': block_name
                    }
                    
                    print(f"  {dataset_key}: {len(X_block)} samples, {len(X_block[0])} dimensions")
    
    def perform_age_regression_single_block(self, dataset_key, dataset):
        """Perform age regression for a single temporal block"""
        X = dataset['X']
        metadata = dataset['metadata']
        agg_name = dataset['agg_name']
        block_name = dataset['block_name']
        
        y_age = metadata['avg_age_days_chunk_start'].values
        groups = metadata['strain'].values if 'strain' in metadata.columns else None
        
        results = {
            'dataset_key': dataset_key,
            'agg_name': agg_name,
            'block_name': block_name,
            'n_samples': len(X),
            'n_dimensions': X.shape[1]
        }
        
        # Perform Leave-One-Group-Out cross-validation
        if groups is not None and len(np.unique(groups)) > 2:
            logo_results = self.perform_logo_regression(X, y_age, groups)
            results.update(logo_results)
        else:
            # Fall back to simple train-test split
            split_results = self.perform_split_regression(X, y_age)
            results.update(split_results)
        
        # Dimension-wise analysis
        dim_results = self.perform_dimensionwise_analysis(X, y_age, groups)
        results['dimension_analysis'] = dim_results
        
        # Feature importance
        feature_importance = self.analyze_feature_importance(X, y_age)
        results['feature_importance'] = feature_importance
        
        return results
    
    def perform_logo_regression(self, X, y, groups):
        """Perform Leave-One-Group-Out regression"""
        logo = LeaveOneGroupOut()
        
        models = {
            'Ridge': Ridge(alpha=1.0),
            'Linear': LinearRegression(),
            'Lasso': Lasso(alpha=0.1)
        }
        
        results = {}
        
        for model_name, model in models.items():
            rmse_scores = []
            r2_scores = []
            mae_scores = []
            group_results = {}
            
            for train_idx, test_idx in logo.split(X, y, groups):
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
                mae = mean_absolute_error(y_test, y_pred)
                
                rmse_scores.append(rmse)
                r2_scores.append(r2)
                mae_scores.append(mae)
                
                test_group = groups[test_idx][0]
                group_results[test_group] = {
                    'rmse': float(rmse),
                    'r2': float(r2),
                    'mae': float(mae),
                    'n_samples': len(y_test)
                }
            
            results[model_name] = {
                'mean_rmse': float(np.mean(rmse_scores)),
                'std_rmse': float(np.std(rmse_scores)),
                'mean_r2': float(np.mean(r2_scores)),
                'std_r2': float(np.std(r2_scores)),
                'mean_mae': float(np.mean(mae_scores)),
                'std_mae': float(np.std(mae_scores)),
                'group_results': group_results
            }
        
        return {'logo_results': results}
    
    def perform_split_regression(self, X, y):
        """Perform simple train-test split regression"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        models = {
            'Ridge': Ridge(alpha=1.0),
            'Linear': LinearRegression(),
            'Lasso': Lasso(alpha=0.1)
        }
        
        results = {}
        
        for model_name, model in models.items():
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
            mae = mean_absolute_error(y_test, y_pred)
            
            results[model_name] = {
                'rmse': float(rmse),
                'r2': float(r2),
                'mae': float(mae),
                'n_train': len(y_train),
                'n_test': len(y_test)
            }
        
        return {'split_results': results}
    
    def perform_dimensionwise_analysis(self, X, y, groups):
        """Analyze age prediction for each dimension individually"""
        n_dims = X.shape[1]
        dim_results = {}
        
        for dim in range(min(n_dims, 50)):  # Limit to first 50 dimensions
            X_dim = X[:, [dim]]
            
            # Simple correlation
            corr, pval = pearsonr(X_dim.flatten(), y)
            
            # Ridge regression
            if groups is not None and len(np.unique(groups)) > 2:
                logo = LeaveOneGroupOut()
                rmse_scores = []
                for train_idx, test_idx in logo.split(X_dim, y, groups):
                    X_train, X_test = X_dim[train_idx], X_dim[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    model = Ridge(alpha=1.0)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    rmse_scores.append(rmse)
                
                mean_rmse = float(np.mean(rmse_scores))
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_dim, y, test_size=0.25, random_state=42
                )
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model = Ridge(alpha=1.0)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                mean_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            
            dim_results[f'dim_{dim}'] = {
                'correlation': float(corr),
                'p_value': float(pval),
                'rmse': mean_rmse
            }
        
        return dim_results
    
    def analyze_feature_importance(self, X, y):
        """Analyze feature importance using multiple methods"""
        # Univariate selection
        selector = SelectKBest(score_func=f_regression, k=min(10, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        selected_features = selector.get_support(indices=True)
        feature_scores = selector.scores_
        
        # Random Forest importance
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_importance = rf.feature_importances_
        
        return {
            'univariate_top_features': [int(f) for f in selected_features],
            'univariate_scores': [float(s) for s in feature_scores],
            'rf_importance': [float(imp) for imp in rf_importance],
            'top_rf_features': [int(i) for i in np.argsort(rf_importance)[-10:]]
        }
    
    def run_temporal_regression_analysis(self):
        """Run age regression analysis for all temporal blocks"""
        print("Starting temporal regression analysis...")
        
        # Process all temporal blocks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_dataset = {
                executor.submit(self.perform_age_regression_single_block, dataset_key, dataset): dataset_key
                for dataset_key, dataset in self.aggregation_datasets.items()
            }
            
            results = {}
            for future in tqdm(as_completed(future_to_dataset), 
                             total=len(future_to_dataset), 
                             desc="Processing temporal blocks"):
                dataset_key = future_to_dataset[future]
                try:
                    result = future.result()
                    results[dataset_key] = result
                except Exception as e:
                    print(f"Error processing {dataset_key}: {e}")
                    results[dataset_key] = {'error': str(e)}
        
        self.results['temporal_regression'] = results
        
        # Save results
        self.save_temporal_regression_results()
        
        return results
    
    # def perform_mixed_effects_analysis(self):
    #     """Perform mixed-effects analysis treating time as a factor"""
    #     print("Performing mixed-effects analysis...")
        
    #     try:
    #         import statsmodels.formula.api as smf
    #     except ImportError:
    #         print("Warning: statsmodels not available, skipping mixed-effects analysis")
    #         return
        
    #     mixed_effects_results = {}
        
    #     for agg_name in self.temporal_configs.keys():
    #         print(f"Processing {agg_name} mixed-effects model...")
            
    #         # Create long-format data
    #         long_data = []
            
    #         block_names = [block['name'] for block in self.temporal_configs[agg_name]['blocks']]
            
    #         for idx, row in self.metadata_aligned.iterrows():
    #             key = row['key']
    #             if key in self.temporal_data and agg_name in self.temporal_data[key]:
                    
    #                 for block_name in block_names:
    #                     if block_name in self.temporal_data[key][agg_name]:
    #                         embedding = self.temporal_data[key][agg_name][block_name]
                            
    #                         # Use PCA to reduce dimensionality for mixed-effects
    #                         if not hasattr(self, f'pca_{agg_name}'):
    #                             # Collect all embeddings for PCA
    #                             all_embeddings = []
    #                             for k in self.temporal_data:
    #                                 for bn in block_names:
    #                                     if (agg_name in self.temporal_data[k] and 
    #                                         bn in self.temporal_data[k][agg_name]):
    #                                         all_embeddings.append(self.temporal_data[k][agg_name][bn])
                                
    #                             if all_embeddings:
    #                                 pca = PCA(n_components=min(10, len(all_embeddings[0])))
    #                                 pca.fit(all_embeddings)
    #                                 setattr(self, f'pca_{agg_name}', pca)
                            
    #                         pca = getattr(self, f'pca_{agg_name}')
    #                         pca_embedding = pca.transform([embedding])[0]
                            
    #                         long_data.append({
    #                             'subject_id': row['cage_id'],
    #                             'strain': row.get('strain', 'unknown'),
    #                             'age': row['avg_age_days_chunk_start'],
    #                             'time_block': block_name,
    #                             'pc1': pca_embedding[0],
    #                             'pc2': pca_embedding[1] if len(pca_embedding) > 1 else 0,
    #                             'pc3': pca_embedding[2] if len(pca_embedding) > 2 else 0
    #                         })
            
    #         if long_data:
    #             df_long = pd.DataFrame(long_data)
                
    #             # Fit mixed-effects model
    #             try:
    #                 # Model with random intercepts for subjects and fixed effects for time
    #                 formula = "age ~ C(time_block) + (1|subject_id)"
    #                 if 'strain' in df_long.columns and df_long['strain'].nunique() > 1:
    #                     formula = "age ~ C(time_block) + C(strain) + (1|subject_id)"
                    
    #                 mixed_model = smf.mixedlm(formula, df_long, groups=df_long['subject_id'])
    #                 mixed_result = mixed_model.fit()
                    
    #                 mixed_effects_results[agg_name] = {
    #                     'summary': str(mixed_result.summary()),
    #                     'aic': float(mixed_result.aic),
    #                     'bic': float(mixed_result.bic),
    #                     'log_likelihood': float(mixed_result.llf),
    #                     'n_observations': len(df_long),
    #                     'n_subjects': df_long['subject_id'].nunique()
    #                 }
                    
    #             except Exception as e:
    #                 print(f"Mixed-effects model failed for {agg_name}: {e}")
    #                 mixed_effects_results[agg_name] = {'error': str(e)}
        
    #     self.results['mixed_effects'] = mixed_effects_results
        
    #     # Save mixed-effects results
    #     with open(self.output_dir / "mixed_effects" / "mixed_effects_results.json", 'w') as f:
    #         json.dump(convert_numpy_types(mixed_effects_results), f, indent=2)
    
    def perform_mixed_effects_analysis(self):
        """Perform mixed-effects analysis treating time as a factor"""
        print("Performing mixed-effects analysis...")
        
        try:
            import statsmodels.formula.api as smf
        except ImportError:
            print("Warning: statsmodels not available, skipping mixed-effects analysis")
            return
        
        mixed_effects_results = {}
        
        for agg_name in self.temporal_configs.keys():
            print(f"Processing {agg_name} mixed-effects model...")
            
            # Create long-format data
            long_data = []
            
            block_names = [block['name'] for block in self.temporal_configs[agg_name]['blocks']]
            
            for idx, row in self.metadata_aligned.iterrows():
                key = row['key']
                if key in self.temporal_data and agg_name in self.temporal_data[key]:
                    
                    for block_name in block_names:
                        if block_name in self.temporal_data[key][agg_name]:
                            embedding = self.temporal_data[key][agg_name][block_name]
                            
                            # Use PCA to reduce dimensionality for mixed-effects
                            if not hasattr(self, f'pca_{agg_name}'):
                                # Collect all embeddings for PCA
                                all_embeddings = []
                                for k in self.temporal_data:
                                    for bn in block_names:
                                        if (agg_name in self.temporal_data[k] and 
                                            bn in self.temporal_data[k][agg_name]):
                                            all_embeddings.append(self.temporal_data[k][agg_name][bn])
                                
                                if all_embeddings:
                                    pca = PCA(n_components=min(10, len(all_embeddings[0])))
                                    pca.fit(all_embeddings)
                                    setattr(self, f'pca_{agg_name}', pca)
                            
                            pca = getattr(self, f'pca_{agg_name}')
                            pca_embedding = pca.transform([embedding])[0]
                            
                            # FIX: Ensure all data types are appropriate for statsmodels
                            long_data.append({
                                'subject_id': str(row['cage_id']),  # Convert to string
                                'strain': str(row.get('strain', 'unknown')),  # Convert to string
                                'age': float(row['avg_age_days_chunk_start']),  # Ensure float
                                'time_block': str(block_name),  # Convert to string
                                'pc1': float(pca_embedding[0]),
                                'pc2': float(pca_embedding[1]) if len(pca_embedding) > 1 else 0.0,
                                'pc3': float(pca_embedding[2]) if len(pca_embedding) > 2 else 0.0
                            })
            
            if long_data:
                df_long = pd.DataFrame(long_data)
                
                # FIX: Clean the data and ensure proper data types
                df_long = df_long.dropna()  # Remove any NaN values
                df_long['subject_id'] = df_long['subject_id'].astype(str)
                df_long['strain'] = df_long['strain'].astype(str) 
                df_long['time_block'] = df_long['time_block'].astype(str)
                df_long['age'] = pd.to_numeric(df_long['age'], errors='coerce')
                
                # Remove any remaining NaN values after conversion
                df_long = df_long.dropna()
                
                if len(df_long) == 0:
                    mixed_effects_results[agg_name] = {'error': 'No valid data after cleaning'}
                    continue
                
                # Fit mixed-effects model
                try:
                    # Start with simpler model first
                    formula = "age ~ C(time_block) + (1|subject_id)"
                    
                    # Only add strain if we have multiple strains with sufficient data
                    if df_long['strain'].nunique() > 1:
                        strain_counts = df_long['strain'].value_counts()
                        if strain_counts.min() >= 3:  # At least 3 observations per strain
                            formula = "age ~ C(time_block) + C(strain) + (1|subject_id)"
                    
                    mixed_model = smf.mixedlm(formula, df_long, groups=df_long['subject_id'])
                    mixed_result = mixed_model.fit(method='lbfgs')  # Use more robust optimizer
                    
                    mixed_effects_results[agg_name] = {
                        'formula_used': formula,
                        'summary': str(mixed_result.summary()),
                        'aic': float(mixed_result.aic),
                        'bic': float(mixed_result.bic),
                        'log_likelihood': float(mixed_result.llf),
                        'n_observations': len(df_long),
                        'n_subjects': df_long['subject_id'].nunique(),
                        'n_strains': df_long['strain'].nunique()
                    }
                    
                except Exception as e:
                    print(f"Mixed-effects model failed for {agg_name}: {e}")
                    mixed_effects_results[agg_name] = {'error': str(e), 'formula_attempted': formula}
        
        self.results['mixed_effects'] = mixed_effects_results
        
        # Save mixed-effects results
        with open(self.output_dir / "mixed_effects" / "mixed_effects_results.json", 'w') as f:
            json.dump(convert_numpy_types(mixed_effects_results), f, indent=2)
        
    def create_comparison_analysis(self):
        """Compare performance across different temporal aggregations"""
        print("Creating comparison analysis...")
        
        if 'temporal_regression' not in self.results:
            print("No temporal regression results available for comparison")
            return
        
        comparison_results = {}
        
        # Extract performance metrics for comparison
        for agg_name in self.temporal_configs.keys():
            agg_results = {}
            
            # Get all results for this aggregation
            matching_keys = [k for k in self.results['temporal_regression'].keys() 
                           if k.startswith(agg_name)]
            
            for dataset_key in matching_keys:
                result = self.results['temporal_regression'][dataset_key]
                
                if 'logo_results' in result:
                    # Use Ridge results from LOGO
                    ridge_result = result['logo_results'].get('Ridge', {})
                    agg_results[result['block_name']] = {
                        'rmse': ridge_result.get('mean_rmse', np.nan),
                        'r2': ridge_result.get('mean_r2', np.nan),
                        'mae': ridge_result.get('mean_mae', np.nan),
                        'n_samples': result.get('n_samples', 0)
                    }
                elif 'split_results' in result:
                    # Use Ridge results from split
                    ridge_result = result['split_results'].get('Ridge', {})
                    agg_results[result['block_name']] = {
                        'rmse': ridge_result.get('rmse', np.nan),
                        'r2': ridge_result.get('r2', np.nan),
                        'mae': ridge_result.get('mae', np.nan),
                        'n_samples': result.get('n_samples', 0)
                    }
            
            comparison_results[agg_name] = agg_results
        
        self.results['comparison'] = comparison_results
        
        # Create comparison visualizations
        self.create_comparison_plots(comparison_results)
        
        # Save comparison results
        with open(self.output_dir / "comparison" / "comparison_results.json", 'w') as f:
            json.dump(convert_numpy_types(comparison_results), f, indent=2)
    
    def create_comparison_plots(self, comparison_results):
        """Create comparison plots across temporal aggregations"""
        
        # 1. RMSE comparison across all temporal blocks
        fig, axes = plt.subplots(3, 1, figsize=(16, 20))
        
        for i, (agg_name, agg_data) in enumerate(comparison_results.items()):
            if not agg_data:
                continue
                
            block_names = list(agg_data.keys())
            rmse_values = [agg_data[block]['rmse'] for block in block_names]
            
            # Sort by RMSE for better visualization
            sorted_indices = np.argsort(rmse_values)
            block_names_sorted = [block_names[idx] for idx in sorted_indices]
            rmse_values_sorted = [rmse_values[idx] for idx in sorted_indices]
            
            axes[i].barh(range(len(block_names_sorted)), rmse_values_sorted)
            axes[i].set_yticks(range(len(block_names_sorted)))
            axes[i].set_yticklabels(block_names_sorted)
            axes[i].set_xlabel('RMSE (days)')
            axes[i].set_title(f'Age Prediction RMSE - {agg_name.replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for j, value in enumerate(rmse_values_sorted):
                axes[i].text(value + 0.01, j, f'{value:.2f}', 
                           va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "comparison" / "rmse_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "comparison" / "rmse_comparison.pdf", bbox_inches='tight')
        plt.close()
        
        # 2. R² comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        
        all_blocks = []
        all_r2_values = []
        all_agg_names = []
        
        for agg_name, agg_data in comparison_results.items():
            for block_name, metrics in agg_data.items():
                if not np.isnan(metrics['r2']):
                    all_blocks.append(f"{agg_name}_{block_name}")
                    all_r2_values.append(metrics['r2'])
                    all_agg_names.append(agg_name)
        
        # Sort by R²
        sorted_indices = np.argsort(all_r2_values)[::-1]  # Descending
        sorted_blocks = [all_blocks[i] for i in sorted_indices]
        sorted_r2 = [all_r2_values[i] for i in sorted_indices]
        sorted_agg_names = [all_agg_names[i] for i in sorted_indices]
        
        # Color by aggregation type
        colors = {'day_night': 'red', 'four_hour': 'blue', 'hourly': 'green'}
        bar_colors = [colors.get(agg, 'gray') for agg in sorted_agg_names]
        
        bars = ax.bar(range(len(sorted_blocks)), sorted_r2, color=bar_colors, alpha=0.7)
        ax.set_xticks(range(len(sorted_blocks)))
        ax.set_xticklabels(sorted_blocks, rotation=45, ha='right')
        ax.set_ylabel('R² Score')
        ax.set_title('Age Prediction R² Across All Temporal Blocks')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        handles = [plt.Rectangle((0,0),1,1, color=color, alpha=0.7) for color in colors.values()]
        labels = [name.replace('_', ' ').title() for name in colors.keys()]
        ax.legend(handles, labels)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "comparison" / "r2_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "comparison" / "r2_comparison.pdf", bbox_inches='tight')
        plt.close()
        
        # 3. Heatmap of performance across time
        self.create_temporal_heatmap(comparison_results)
    
    def create_temporal_heatmap(self, comparison_results):
        """Create heatmap showing performance across different times of day"""
        
        # Create hourly heatmap
        if 'hourly' in comparison_results:
            hourly_data = comparison_results['hourly']
            
            hours = []
            rmse_values = []
            r2_values = []
            
            for hour in range(24):
                hour_key = f'hour_{hour:02d}'
                if hour_key in hourly_data and not np.isnan(hourly_data[hour_key]['rmse']):
                    hours.append(hour)
                    rmse_values.append(hourly_data[hour_key]['rmse'])
                    r2_values.append(hourly_data[hour_key]['r2'])
            
            if hours:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
                
                # RMSE heatmap
                rmse_matrix = np.array(rmse_values).reshape(1, -1)
                im1 = ax1.imshow(rmse_matrix, cmap='Reds', aspect='auto')
                ax1.set_xticks(range(len(hours)))
                ax1.set_xticklabels([f'{h:02d}:00' for h in hours])
                ax1.set_yticks([0])
                ax1.set_yticklabels(['RMSE'])
                ax1.set_title('Hourly Age Prediction RMSE Across 24-Hour Period')
                
                # Add colorbar
                plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.1, label='RMSE (days)')
                
                # R² heatmap
                r2_matrix = np.array(r2_values).reshape(1, -1)
                im2 = ax2.imshow(r2_matrix, cmap='Blues', aspect='auto')
                ax2.set_xticks(range(len(hours)))
                ax2.set_xticklabels([f'{h:02d}:00' for h in hours])
                ax2.set_yticks([0])
                ax2.set_yticklabels(['R²'])
                ax2.set_title('Hourly Age Prediction R² Across 24-Hour Period')
                ax2.set_xlabel('Hour of Day')
                
                plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.1, label='R² Score')
                
                # Add day/night shading
                for ax in [ax1, ax2]:
                    # Day period (7am-7pm) - light yellow background
                    ax.axvspan(6.5, 18.5, alpha=0.2, color='yellow', zorder=0)
                    # Night period - light blue background
                    ax.axvspan(-0.5, 6.5, alpha=0.2, color='blue', zorder=0)
                    ax.axvspan(18.5, len(hours)-0.5, alpha=0.2, color='blue', zorder=0)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / "visualizations" / "hourly_performance_heatmap.png", 
                           dpi=300, bbox_inches='tight')
                plt.savefig(self.output_dir / "visualizations" / "hourly_performance_heatmap.pdf", 
                           bbox_inches='tight')
                plt.close()
    
    def create_circadian_analysis_plots(self):
        """Create specific plots for circadian analysis"""
        print("Creating circadian analysis plots...")
        
        if 'temporal_regression' not in self.results:
            return
        
        # Day vs Night comparison
        day_night_results = {}
        for key, result in self.results['temporal_regression'].items():
            if key.startswith('day_night_'):
                day_night_results[result['block_name']] = result
        
        if 'day' in day_night_results and 'night' in day_night_results:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Compare RMSE
            day_rmse = day_night_results['day']['logo_results']['Ridge']['mean_rmse']
            night_rmse = day_night_results['night']['logo_results']['Ridge']['mean_rmse']
            
            axes[0,0].bar(['Day (7am-7pm)', 'Night (7pm-7am)'], [day_rmse, night_rmse], 
                         color=['gold', 'navy'], alpha=0.7)
            axes[0,0].set_ylabel('RMSE (days)')
            axes[0,0].set_title('Day vs Night Age Prediction RMSE')
            axes[0,0].grid(True, alpha=0.3)
            
            # Compare R²
            day_r2 = day_night_results['day']['logo_results']['Ridge']['mean_r2']
            night_r2 = day_night_results['night']['logo_results']['Ridge']['mean_r2']
            
            axes[0,1].bar(['Day (7am-7pm)', 'Night (7pm-7am)'], [day_r2, night_r2], 
                         color=['gold', 'navy'], alpha=0.7)
            axes[0,1].set_ylabel('R² Score')
            axes[0,1].set_title('Day vs Night Age Prediction R²')
            axes[0,1].grid(True, alpha=0.3)
            
            # Dimension-wise correlations for day vs night
            if 'dimension_analysis' in day_night_results['day']:
                day_dims = day_night_results['day']['dimension_analysis']
                night_dims = day_night_results['night']['dimension_analysis']
                
                # Get top 20 dimensions from each
                day_corrs = [(k, v['correlation']) for k, v in day_dims.items()]
                night_corrs = [(k, v['correlation']) for k, v in night_dims.items()]
                
                day_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
                night_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
                
                # Plot top correlations
                top_n = min(15, len(day_corrs))
                
                day_dims_plot = [x[0] for x in day_corrs[:top_n]]
                day_vals_plot = [x[1] for x in day_corrs[:top_n]]
                
                axes[1,0].barh(range(len(day_dims_plot)), day_vals_plot, color='gold', alpha=0.7)
                axes[1,0].set_yticks(range(len(day_dims_plot)))
                axes[1,0].set_yticklabels(day_dims_plot, fontsize=8)
                axes[1,0].set_xlabel('Correlation with Age')
                axes[1,0].set_title('Top Day-Active Dimensions')
                axes[1,0].grid(True, alpha=0.3)
                
                night_dims_plot = [x[0] for x in night_corrs[:top_n]]
                night_vals_plot = [x[1] for x in night_corrs[:top_n]]
                
                axes[1,1].barh(range(len(night_dims_plot)), night_vals_plot, color='navy', alpha=0.7)
                axes[1,1].set_yticks(range(len(night_dims_plot)))
                axes[1,1].set_yticklabels(night_dims_plot, fontsize=8)
                axes[1,1].set_xlabel('Correlation with Age')
                axes[1,1].set_title('Top Night-Active Dimensions')
                axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "visualizations" / "circadian_analysis.png", 
                       dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / "visualizations" / "circadian_analysis.pdf", 
                       bbox_inches='tight')
            plt.close()
    
    def save_temporal_regression_results(self):
        """Save temporal regression results to JSON files"""
        print("Saving temporal regression results...")
        
        # Save by aggregation type
        for agg_name in self.temporal_configs.keys():
            agg_results = {k: v for k, v in self.results['temporal_regression'].items() 
                          if k.startswith(agg_name)}
            
            if agg_results:
                output_file = self.output_dir / agg_name / f"{agg_name}_results.json"
                with open(output_file, 'w') as f:
                    json.dump(convert_numpy_types(agg_results), f, indent=2)
        
        # Save complete results
        with open(self.output_dir / "temporal_regression_complete.json", 'w') as f:
            json.dump(convert_numpy_types(self.results['temporal_regression']), f, indent=2)
    
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        print("Creating summary report...")
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'n_sequences': len(self.temporal_data),
                'n_total_samples': len(self.metadata_aligned),
                'embedding_dimensions': self.embeddings.shape[1] if hasattr(self, 'embeddings') else 'unknown',
                'temporal_resolution': self.aggregation_info.get('num_frames', 'unknown'),
                'age_range': [
                    float(self.metadata_aligned['avg_age_days_chunk_start'].min()),
                    float(self.metadata_aligned['avg_age_days_chunk_start'].max())
                ] if 'avg_age_days_chunk_start' in self.metadata_aligned.columns else 'unknown'
            },
            'temporal_configs': self.temporal_configs,
            'analysis_completed': []
        }
        
        if 'temporal_regression' in self.results:
            summary['analysis_completed'].append('temporal_regression')
            
            # Best performing blocks
            best_blocks = {}
            for agg_name in self.temporal_configs.keys():
                agg_blocks = {k: v for k, v in self.results['temporal_regression'].items() 
                             if k.startswith(agg_name)}
                
                if agg_blocks:
                    # Find best block by lowest RMSE
                    best_block_key = min(agg_blocks.keys(), 
                                       key=lambda k: agg_blocks[k].get('logo_results', {}).get('Ridge', {}).get('mean_rmse', float('inf')))
                    best_block = agg_blocks[best_block_key]
                    
                    best_blocks[agg_name] = {
                        'block_name': best_block['block_name'],
                        'rmse': best_block.get('logo_results', {}).get('Ridge', {}).get('mean_rmse', 'unknown'),
                        'r2': best_block.get('logo_results', {}).get('Ridge', {}).get('mean_r2', 'unknown'),
                        'n_samples': best_block.get('n_samples', 'unknown')
                    }
            
            summary['best_performing_blocks'] = best_blocks
        
        if 'comparison' in self.results:
            summary['analysis_completed'].append('comparison_analysis')
        
        if 'mixed_effects' in self.results:
            summary['analysis_completed'].append('mixed_effects_analysis')
        
        # Save summary
        with open(self.output_dir / "analysis_summary.json", 'w') as f:
            json.dump(convert_numpy_types(summary), f, indent=2)
        
        # Create text report
        with open(self.output_dir / "analysis_summary.txt", 'w') as f:
            f.write("TEMPORAL SUB-AGGREGATION ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis completed: {summary['analysis_timestamp']}\n")
            f.write(f"Dataset: {summary['dataset_info']['n_sequences']} sequences\n")
            f.write(f"Age range: {summary['dataset_info']['age_range']} days\n\n")
            
            if 'best_performing_blocks' in summary:
                f.write("BEST PERFORMING TIME BLOCKS:\n")
                f.write("-" * 30 + "\n")
                for agg_name, best_block in summary['best_performing_blocks'].items():
                    f.write(f"\n{agg_name.upper()}:\n")
                    f.write(f"  Best block: {best_block['block_name']}\n")
                    f.write(f"  RMSE: {best_block['rmse']:.3f} days\n")
                    f.write(f"  R²: {best_block['r2']:.3f}\n")
                    f.write(f"  Samples: {best_block['n_samples']}\n")
            
            f.write(f"\nAnalysis types completed: {', '.join(summary['analysis_completed'])}\n")
            f.write(f"\nResults saved to: {self.output_dir}\n")
    
    def run_complete_analysis(self):
        """Run the complete temporal sub-aggregation analysis"""
        print("Starting complete temporal sub-aggregation analysis...")
        print("=" * 60)
        
        # 1. Temporal regression analysis
        print("\n1. Running temporal regression analysis...")
        self.run_temporal_regression_analysis()
        
        # 2. Mixed-effects analysis
        print("\n2. Running mixed-effects analysis...")
        self.perform_mixed_effects_analysis()
        
        # 3. Comparison analysis
        print("\n3. Creating comparison analysis...")
        self.create_comparison_analysis()
        
        # 4. Visualization
        print("\n4. Creating visualizations...")
        self.create_circadian_analysis_plots()
        
        # 5. Summary report
        print("\n5. Creating summary report...")
        self.create_summary_report()
        
        print("\n" + "=" * 60)
        print("Complete temporal sub-aggregation analysis finished!")
        print(f"Results saved to: {self.output_dir}")
        
        # Print key findings
        if 'temporal_regression' in self.results:
            print("\nKEY FINDINGS:")
            
            # Day vs Night comparison
            day_night_keys = [k for k in self.results['temporal_regression'].keys() 
                             if k.startswith('day_night_')]
            if len(day_night_keys) >= 2:
                day_result = next((v for k, v in self.results['temporal_regression'].items() 
                                 if k.startswith('day_night_day')), None)
                night_result = next((v for k, v in self.results['temporal_regression'].items() 
                                   if k.startswith('day_night_night')), None)
                
                if day_result and night_result:
                    day_rmse = day_result.get('logo_results', {}).get('Ridge', {}).get('mean_rmse')
                    night_rmse = night_result.get('logo_results', {}).get('Ridge', {}).get('mean_rmse')
                    
                    if day_rmse and night_rmse:
                        better_period = "day" if day_rmse < night_rmse else "night"
                        print(f"• Better age prediction during {better_period} period")
                        print(f"  Day RMSE: {day_rmse:.2f} days")
                        print(f"  Night RMSE: {night_rmse:.2f} days")


def main():
    """Main function to run temporal sub-aggregation analysis"""
    parser = argparse.ArgumentParser(description='Temporal Sub-Aggregation Analysis')
    parser.add_argument('embeddings_path', help='Path to embeddings .npy file')
    parser.add_argument('metadata_path', help='Path to metadata CSV file')
    parser.add_argument('--output_dir', default='temporal_subaggregation_results', 
                       help='Output directory for results')
    parser.add_argument('--max_workers', type=int, default=8, 
                       help='Maximum number of worker threads')
    parser.add_argument('--bodyweight_file', help='Path to BW.csv file (optional)')
    parser.add_argument('--tissue_file', help='Path to TissueCollection_PP.csv file (optional)')
    parser.add_argument('--summary_file', help='Path to summary_metadata.csv file (optional)')
    
    args = parser.parse_args()
    
    # Prepare bodyweight files if provided
    bodyweight_files = None
    if args.bodyweight_file and args.tissue_file and args.summary_file:
        bodyweight_files = {
            'bodyweight': args.bodyweight_file,
            'tissue': args.tissue_file,
            'summary': args.summary_file
        }
    
    # Create and run analyzer
    analyzer = TemporalSubAggregationAnalyzer(
        embeddings_path=args.embeddings_path,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        bodyweight_files=bodyweight_files
    )
    
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()