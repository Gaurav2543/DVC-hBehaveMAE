#!/usr/bin/env python3
"""
Complete embedding analysis pipeline that starts from embeddings and performs:
1. Age regression analysis (with/without bodyweight)
2. Strain classification analysis  
3. Saves all results and predictions for downstream analysis
4. Creates strain misclassification plots
ADAPTED for Zarr-based embedding format.
"""

import argparse
import warnings
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                           accuracy_score, f1_score, classification_report,
                           confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from scipy.stats import pearsonr
import pickle
import matplotlib.colors as mcolors
import zarr
from pygam import LinearGAM, s

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
    else:
        return obj

class CompleteEmbeddingAnalyzer:
    """Complete analysis pipeline for behavioral embeddings"""
    
    def __init__(self, embeddings_path, metadata_path, output_dir="complete_analysis_results", 
                 max_workers=8, bodyweight_files=None, aggregation='4weeks', embedding_level='combined'):
        self.embeddings_path = Path(embeddings_path)
        self.metadata_path = metadata_path
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.bodyweight_files = bodyweight_files
        self.aggregation = aggregation
        self.embedding_level = embedding_level
        
        # Create comprehensive output structure
        subdirs = ["age_regression", "strain_classification", "raw_results", 
                  "visualizations", "model_objects", "predictions"]
        self.output_dir.mkdir(exist_ok=True)
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Storage for results
        self.results = {
            'age_regression': {},
            'strain_classification': {},
            'predictions': {},
            'metadata': {}
        }
    
    def load_bodyweight_data(self):
        """Load and merge bodyweight data from BW.csv and TissueCollection_PP.csv"""
        if not self.bodyweight_files:
            return None
        
        print("Loading bodyweight data...")
        
        # Load BW data
        bw_df = pd.read_csv(self.bodyweight_files['bodyweight'])
        bw_df['date'] = pd.to_datetime(bw_df['date'])
        
        # Load tissue collection data
        tissue_df = pd.read_csv(self.bodyweight_files['tissue'])
        tissue_df['death_date'] = pd.to_datetime(tissue_df['death_date'])
        
        # Rename columns for consistency
        tissue_df = tissue_df.rename(columns={
            'death_date': 'date',
            'weight_pre_sac': 'body_weight'
        })
        
        # Combine the datasets
        combined_bw = pd.concat([
            bw_df[['animal_id', 'date', 'body_weight']],
            tissue_df[['animal_id', 'date', 'body_weight']]
        ], ignore_index=True)
        
        # Remove duplicates and sort
        combined_bw = combined_bw.drop_duplicates(
            subset=['animal_id', 'date']
        ).sort_values(['animal_id', 'date'])
        
        print(f"Loaded bodyweight data for {combined_bw['animal_id'].nunique()} animals "
              f"with {len(combined_bw)} total measurements")
        
        return combined_bw
    
    def match_bodyweight_to_timepoint(self, animal_ids, target_date, bw_data, strategy="gam_spline"):
        """Match bodyweight measurement to a specific timepoint for given animals"""
        if not animal_ids or bw_data is None:
            return np.nan
        
        # Get bodyweight data for all animals in the cage
        animal_bw_data = bw_data[bw_data['animal_id'].isin(animal_ids)].copy()
        
        if animal_bw_data.empty:
            return np.nan
        
        if strategy == "gam_spline":
            return self._gam_spline_matching(animal_ids, target_date, animal_bw_data)
        elif strategy == "closest":
            # Find the measurement closest in time to target_date
            animal_bw_data['time_diff'] = abs((animal_bw_data['date'] - target_date).dt.total_seconds())
            closest_measurements = animal_bw_data.loc[
                animal_bw_data.groupby('animal_id')['time_diff'].idxmin()
            ]
            return closest_measurements['body_weight'].mean()
        
        elif strategy == "interpolate":
            # Linear interpolation strategy
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
    
    def add_bodyweight_to_metadata(self):
        """Add bodyweight information to metadata"""
        if not self.bodyweight_files:
            return self.metadata
        
        print("Processing bodyweight data...")
        
        # Load bodyweight data
        combined_bw = self.load_bodyweight_data()
        if combined_bw is None:
            return self.metadata
        
        # Load summary metadata
        summary_metadata = pd.read_csv(self.bodyweight_files['summary'])
        summary_metadata['from_tpt'] = pd.to_datetime(summary_metadata['from_tpt'])
        summary_metadata['to_tpt'] = pd.to_datetime(summary_metadata['to_tpt'])
        
        # Process bodyweight matching
        bodyweights = []
        
        def process_single_row(row):
            if pd.isna(row.animal_ids) or row.animal_ids == '':
                return np.nan
            animal_ids = [aid.strip() for aid in str(row.animal_ids).split(';') if aid.strip()]
            
            # Use midpoint of time range as target
            target_date = row.from_tpt + (row.to_tpt - row.from_tpt) / 2
            
            return self.match_bodyweight_to_timepoint(animal_ids, target_date, combined_bw, "gam_spline")
        
        # Process with progress bar
        with ThreadPoolExecutor(max_workers=4) as executor:
            with tqdm(total=len(summary_metadata), desc="Processing bodyweight") as pbar:
                future_to_idx = {
                    executor.submit(process_single_row, row): idx 
                    for idx, row in enumerate(summary_metadata.itertuples())
                }
                
                bodyweights = [np.nan] * len(summary_metadata)
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    bodyweights[idx] = future.result()
                    pbar.update(1)
        
        summary_metadata['bodyweight'] = bodyweights
        
        # Merge with original metadata
        original_metadata = self.metadata.copy()
        original_metadata['from_tpt'] = pd.to_datetime(original_metadata['from_tpt'])
        merged = original_metadata.merge(
            summary_metadata[['cage_id', 'from_tpt', 'bodyweight']], 
            on=['cage_id', 'from_tpt'], 
            how='left'
        )
        
        valid_bw = merged['bodyweight'].notna().sum()
        print(f"Successfully matched bodyweight for {valid_bw}/{len(merged)} entries "
              f"({100*valid_bw/len(merged):.1f}%)")
        
        return merged
    
    def load_and_prepare_data(self):
        """Load embeddings from Zarr stores and metadata, create aligned dataset"""
        print("Loading embeddings from Zarr stores...")
        
        core_emb_dir = self.embeddings_path / self.aggregation / 'core_embeddings'
        
        if not core_emb_dir.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {core_emb_dir}")
        
        self.core_emb_dir = core_emb_dir
        
        # Get aggregation info from a sample Zarr file
        sample_zarr = next(core_emb_dir.glob('*.zarr'), None)
        if sample_zarr:
            sample_store = zarr.open(str(sample_zarr), mode='r')
            if 'timestamps' in sample_store:
                self.aggregation_info = {
                    'num_frames': len(sample_store['timestamps']),
                    'aggregation_name': self.aggregation
                }
            else:
                self.aggregation_info = {
                    'num_frames': 40320,  # 4 weeks default
                    'aggregation_name': self.aggregation
                }
        else:
            raise FileNotFoundError(f"No Zarr files found in {core_emb_dir}")
        
        print(f"Aggregation info: {self.aggregation_info}")
        
        # Load metadata
        print("Loading metadata...")
        self.metadata = pd.read_csv(self.metadata_path)
        
        # Add bodyweight data if files provided
        if self.bodyweight_files:
            self.metadata = self.add_bodyweight_to_metadata()
        
        # Create aligned dataset
        self.create_aligned_dataset_zarr()
    
    def create_aligned_dataset_zarr(self):
        """Create aligned embeddings and metadata using Zarr stores"""
        print("Creating aligned dataset from Zarr stores...")
        
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
            zarr_path = self.core_emb_dir / f"{key}.zarr"
            
            if zarr_path.exists():
                try:
                    store = zarr.open(str(zarr_path), mode='r')
                    
                    # Get the specified embedding level
                    if self.embedding_level not in store:
                        print(f"[WARN] Level '{self.embedding_level}' not in {zarr_path}, using 'combined'")
                        level_to_use = 'combined'
                    else:
                        level_to_use = self.embedding_level
                    
                    seq_embeddings = store[level_to_use][:]
                    
                    # Handle both 1D (already aggregated) and 2D (temporal) embeddings
                    if seq_embeddings.ndim == 1:
                        daily_embedding = seq_embeddings
                    else:
                        daily_embedding = seq_embeddings.mean(axis=0)
                    
                    if not np.isnan(daily_embedding).any():
                        aligned_embeddings.append(daily_embedding)
                        aligned_metadata.append(row)
                        
                except Exception as e:
                    print(f"[WARN] Failed to load {zarr_path}: {e}")
        
        self.X = np.array(aligned_embeddings)
        self.metadata_aligned = pd.DataFrame(aligned_metadata).reset_index(drop=True)
        
        # Clean data
        required_cols = ['avg_age_days_chunk_start', 'strain']
        valid_mask = True
        
        for col in required_cols:
            if col in self.metadata_aligned.columns:
                valid_mask &= self.metadata_aligned[col].notna()
        
        self.X = self.X[valid_mask]
        self.metadata_aligned = self.metadata_aligned[valid_mask].reset_index(drop=True)
        
        print(f"Final aligned dataset: {self.X.shape[0]} samples, {self.X.shape[1]} dimensions")
        print(f"Age range: {self.metadata_aligned['avg_age_days_chunk_start'].min():.1f} - "
              f"{self.metadata_aligned['avg_age_days_chunk_start'].max():.1f} days")
        print(f"Number of strains: {self.metadata_aligned['strain'].nunique()}")
        
        # Save aligned data
        np.save(self.output_dir / "raw_results" / "aligned_embeddings.npy", self.X)
        self.metadata_aligned.to_csv(self.output_dir / "raw_results" / "aligned_metadata.csv", index=False)
    
    def perform_age_regression_analysis(self, include_bodyweight=False):
        """Perform comprehensive age regression analysis"""
        print(f"\nPerforming age regression analysis (bodyweight: {include_bodyweight})...")
        
        y_age = self.metadata_aligned['avg_age_days_chunk_start'].values
        groups = self.metadata_aligned['strain'].values
        
        # Prepare features
        X = self.X.copy()
        feature_names = [f'dim_{i+1}' for i in range(self.X.shape[1])]
        
        # Handle bodyweight if requested and available
        if include_bodyweight and 'bodyweight' in self.metadata_aligned.columns:
            bodyweight = self.metadata_aligned['bodyweight'].values.reshape(-1, 1)
            valid_bw_mask = ~np.isnan(bodyweight.flatten())
            
            if valid_bw_mask.sum() > len(valid_bw_mask) * 0.5:
                X = X[valid_bw_mask]
                y_age = y_age[valid_bw_mask]
                groups = groups[valid_bw_mask]
                bodyweight = bodyweight[valid_bw_mask]
                
                bw_scaler = StandardScaler()
                bodyweight = bw_scaler.fit_transform(bodyweight)
                X = np.concatenate([X, bodyweight], axis=1)
                feature_names.append('bodyweight')
                print(f"Added bodyweight covariate. New shape: {X.shape}")
        
        suffix = "with_bodyweight" if include_bodyweight and 'bodyweight' in feature_names else "without_bodyweight"
        
        # Perform LOGO regression
        regression_results = self.perform_logo_age_regression(X, y_age, groups, suffix)
        
        # Dimension-wise analysis
        dimension_results = self.perform_dimensionwise_age_regression(X, y_age, groups, feature_names, suffix)
        
        # Store results
        self.results['age_regression'][suffix] = {
            'logo_results': regression_results,
            'dimension_results': dimension_results,
            'feature_names': feature_names,
            'n_samples': len(X),
            'n_features': X.shape[1]
        }
        
        # Save results
        with open(self.output_dir / "age_regression" / f"age_regression_{suffix}.json", 'w') as f:
            json.dump(convert_numpy_types(self.results['age_regression'][suffix]), f, indent=2)
        
        return regression_results, dimension_results
    
    def perform_logo_age_regression(self, X, y, groups, suffix):
        """Perform Leave-One-Group-Out age regression"""
        print("Performing LOGO age regression...")
        
        logo = LeaveOneGroupOut()
        
        # Use LinearRegression as requested
        model = LinearRegression()
        
        rmse_scores = []
        r2_scores = []
        mae_scores = []
        group_results = {}
        all_predictions = {}
        all_true_values = {}
        
        for train_idx, test_idx in tqdm(logo.split(X, y, groups), desc="LOGO CV"):
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
            
            # Store group-specific results
            test_group = groups[test_idx][0]
            group_results[test_group] = {
                'rmse': float(rmse),
                'r2': float(r2),
                'mae': float(mae),
                'n_samples': len(y_test)
            }
            
            # Store predictions for this group
            all_predictions[test_group] = y_pred.tolist()
            all_true_values[test_group] = y_test.tolist()
        
        # Overall results
        results = {
            'mean_rmse': float(np.mean(rmse_scores)),
            'std_rmse': float(np.std(rmse_scores)),
            'mean_r2': float(np.mean(r2_scores)),
            'std_r2': float(np.std(r2_scores)),
            'mean_mae': float(np.mean(mae_scores)),
            'std_mae': float(np.std(mae_scores)),
            'group_results': group_results,
            'all_predictions': all_predictions,
            'all_true_values': all_true_values
        }
        
        # Save predictions for later analysis
        predictions_data = {
            'predictions': all_predictions,
            'true_values': all_true_values,
            'group_results': group_results
        }
        
        with open(self.output_dir / "predictions" / f"age_regression_predictions_{suffix}.json", 'w') as f:
            json.dump(convert_numpy_types(predictions_data), f, indent=2)
        
        print(f"Age regression ({suffix}) - RMSE: {results['mean_rmse']:.2f}±{results['std_rmse']:.2f}, "
              f"R²: {results['mean_r2']:.3f}±{results['std_r2']:.3f}")
        
        return results
    
    def perform_dimensionwise_age_regression(self, X, y, groups, feature_names, suffix):
        """Perform dimension-wise age regression analysis"""
        print("Performing dimension-wise age regression...")
        
        logo = LeaveOneGroupOut()
        dimension_results = {}
        
        def analyze_single_dimension(dim):
            X_dim = X[:, [dim]]
            
            rmse_scores = []
            r2_scores = []
            
            for train_idx, test_idx in logo.split(X_dim, y, groups):
                X_train, X_test = X_dim[train_idx], X_dim[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model = LinearRegression()
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                rmse_scores.append(rmse)
                r2_scores.append(r2)
            
            # Also calculate simple correlation
            corr, pval = pearsonr(X_dim.flatten(), y)
            
            return {
                'dimension': feature_names[dim],
                'mean_rmse': float(np.mean(rmse_scores)),
                'std_rmse': float(np.std(rmse_scores)),
                'mean_r2': float(np.mean(r2_scores)),
                'std_r2': float(np.std(r2_scores)),
                'correlation': float(corr),
                'p_value': float(pval)
            }
        
        # Process dimensions in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_dim = {executor.submit(analyze_single_dimension, dim): dim 
                           for dim in range(X.shape[1])}
            
            for future in tqdm(as_completed(future_to_dim), total=X.shape[1], 
                             desc="Dimension analysis"):
                dim = future_to_dim[future]
                result = future.result()
                dimension_results[dim] = result
        
        return dimension_results
    
    def perform_strain_classification_analysis(self):
        """Perform comprehensive strain classification analysis"""
        print("\nPerforming strain classification analysis...")
        
        y_strain = self.metadata_aligned['strain'].values
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_strain)
        
        print(f"Number of strains: {len(le.classes_)}")
        print(f"Strain distribution: {np.bincount(y_encoded)}")
        
        # Perform train-test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Linear classifier (Logistic Regression as requested)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Get predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Get classification report
        report = classification_report(y_test, y_pred, target_names=le.classes_, 
                                     output_dict=True, zero_division=0)
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Convert back to strain names for storage
        y_test_strains = le.inverse_transform(y_test)
        y_pred_strains = le.inverse_transform(y_pred)
        
        classification_results = {
            'accuracy': float(accuracy),
            'f1_weighted': float(f1),
            'classification_report': convert_numpy_types(report),
            'confusion_matrix': convert_numpy_types(cm),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'strain_names': le.classes_.tolist(),
            'predictions': {
                'true_strains': y_test_strains.tolist(),
                'predicted_strains': y_pred_strains.tolist(),
                'prediction_probabilities': convert_numpy_types(y_pred_proba)
            }
        }
        
        # Store results
        self.results['strain_classification'] = classification_results
        
        # Save results
        with open(self.output_dir / "strain_classification" / "strain_classification_results.json", 'w') as f:
            json.dump(convert_numpy_types(classification_results), f, indent=2)
        
        # Save predictions separately for easy access
        predictions_data = {
            'true_strains': y_test_strains.tolist(),
            'predicted_strains': y_pred_strains.tolist(),
            'strain_names': le.classes_.tolist(),
            'confusion_matrix': convert_numpy_types(cm),
            'accuracy': float(accuracy),
            'f1_score': float(f1)
        }
        
        with open(self.output_dir / "predictions" / "strain_classification_predictions.json", 'w') as f:
            json.dump(predictions_data, f, indent=2)
        
        # Save model objects
        with open(self.output_dir / "model_objects" / "strain_classifier.pkl", 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler, 'label_encoder': le}, f)
        
        print(f"Strain classification - Accuracy: {accuracy:.3f}, F1-weighted: {f1:.3f}")
        
        # Create misclassification plot
        self.create_strain_misclassification_plot(y_test_strains, y_pred_strains, le.classes_)
        
        return classification_results
    
    def create_strain_color_mapping(self, strain_list):
        """Create color mapping based on strain families"""
        family_base_colors = {
            'CC00': '#FF6B6B',      # Red shades
            'BXD': '#4ECDC4',       # Teal shades  
            'C3H': '#45B7D1',       # Blue shades
            'C57': '#96CEB4',       # Green shades
            'DBA': '#FFEAA7',       # Yellow shades
            'other': '#DDA0DD'      # Purple shades for others
        }
        
        strain_families = {family: [] for family in family_base_colors.keys()}
        
        for strain in strain_list:
            strain_upper = str(strain).upper()
            assigned = False
            
            for family in ['CC00', 'BXD', 'C3H', 'C57', 'DBA']:
                if strain_upper.startswith(family):
                    strain_families[family].append(strain)
                    assigned = True
                    break
            
            if not assigned:
                strain_families['other'].append(strain)
        
        color_mapping = {}
        
        for family, base_color in family_base_colors.items():
            family_strains = strain_families[family]
            if len(family_strains) == 0:
                continue
            
            if len(family_strains) == 1:
                color_mapping[family_strains[0]] = base_color
            else:
                base_rgb = mcolors.to_rgb(base_color)
                
                for i, strain in enumerate(family_strains):
                    lightness_factor = 0.3 + (0.7 * i / max(1, len(family_strains) - 1))
                    adjusted_rgb = tuple(min(1.0, c + (1 - c) * (1 - lightness_factor)) for c in base_rgb)
                    color_mapping[strain] = mcolors.to_hex(adjusted_rgb)
        
        return color_mapping
    
    def create_strain_misclassification_plot(self, y_true, y_pred, strain_names):
        """Create strain misclassification stacked bar plot with individual strain details"""
        print("Creating detailed strain misclassification plots...")
        
        # Create confusion matrix
        le_temp = LabelEncoder()
        all_strains = list(set(y_true) | set(y_pred) | set(strain_names))
        le_temp.fit(all_strains)
        
        y_true_encoded = le_temp.transform(y_true)
        y_pred_encoded = le_temp.transform(y_pred)
        
        cm = confusion_matrix(y_true_encoded, y_pred_encoded, labels=range(len(all_strains)))
        cm_df = pd.DataFrame(cm, index=all_strains, columns=all_strains)
        
        # Calculate incorrect predictions (remove diagonal)
        incorrect_predictions = cm_df.copy()
        np.fill_diagonal(incorrect_predictions.values, 0)
        
        # Normalize by row to get proportions
        row_sums = cm_df.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        normalized_incorrect = incorrect_predictions.div(row_sums, axis=0)
        
        # Sort strains for better visualization
        strain_order = sorted(all_strains, key=lambda x: (
            0 if str(x).upper().startswith('CC00') else
            1 if str(x).upper().startswith('BXD') else  
            2 if str(x).upper().startswith('C3H') else
            3 if str(x).upper().startswith('C57') else
            4 if str(x).upper().startswith('DBA') else 5,
            str(x)
        ))
        
        # Reorder the matrix
        normalized_incorrect = normalized_incorrect.loc[strain_order, strain_order]
        
        # # Create color mapping
        # color_mapping = self.create_strain_color_mapping(strain_order)
        
        # # Create the plot
        # fig_width = max(20, len(strain_order) * 0.5)
        # fig, ax = plt.subplots(figsize=(fig_width, 10))
        
        # x_positions = np.arange(len(strain_order))
        # bottoms = np.zeros(len(strain_order))
        
        # # Create stacked bars
        # for predicted_strain in strain_order:
        #     values = normalized_incorrect.loc[:, predicted_strain].values
            
        #     nonzero_mask = values > 0
        #     if np.any(nonzero_mask):
        #         ax.bar(x_positions[nonzero_mask], 
        #               values[nonzero_mask],
        #               bottom=bottoms[nonzero_mask],
        #               color=color_mapping.get(predicted_strain, '#808080'),
        #               label=predicted_strain,
        #               alpha=0.8,
        #               edgecolor='white',
        #               linewidth=0.5)
                
        #         bottoms += values
        
        # # Customize the plot
        # ax.set_xlabel('True Strain', fontsize=12, fontweight='bold')
        # ax.set_ylabel('Proportion of Incorrect Predictions', fontsize=12, fontweight='bold')
        # ax.set_title('Normalized Stacked Incorrect Predictions per Strain', 
        #             fontsize=14, fontweight='bold')
        
        # ax.set_xticks(x_positions)
        # ax.set_xticklabels(strain_order, rotation=45, ha='right', fontsize=8)
        # ax.set_ylim(0, 1)
        # ax.grid(True, alpha=0.3, axis='y')
        # ax.set_axisbelow(True)
        
        # # Create simplified legend by strain families
        # handles, labels = ax.get_legend_handles_labels()
        # family_groups = {'CC00': [], 'BXD': [], 'C3H': [], 'C57': [], 'DBA': [], 'Other': []}
        
        # for handle, label in zip(handles, labels):
        #     label_upper = str(label).upper()
        #     if label_upper.startswith('CC00'):
        #         family_groups['CC00'].append((handle, label))
        #     elif label_upper.startswith('BXD'):
        #         family_groups['BXD'].append((handle, label))
        #     elif label_upper.startswith('C3H'):
        #         family_groups['C3H'].append((handle, label))
        #     elif label_upper.startswith('C57'):
        #         family_groups['C57'].append((handle, label))
        #     elif label_upper.startswith('DBA'):
        #         family_groups['DBA'].append((handle, label))
        #     else:
        #         family_groups['Other'].append((handle, label))
        
        # legend_handles = []
        # legend_labels = []
        
        # for family, items in family_groups.items():
        #     if items:
        #         handle, _ = items[0]
        #         legend_handles.append(handle)
        #         legend_labels.append(f"{family} strains ({len(items)})")
        
        # ax.legend(legend_handles, legend_labels, 
        #           bbox_to_anchor=(1.05, 1), loc='upper left',
        #           title='Predicted as')
        
        # # Add statistics
        # total_samples = len(y_true)
        # accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        
        # stats_text = f'Total samples: {total_samples}\nOverall accuracy: {accuracy:.3f}'
        # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
        #         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # plt.tight_layout()
        
        # # Save plot
        # plt.savefig(self.output_dir / "visualizations" / "strain_misclassification.png", 
        #            dpi=300, bbox_inches='tight')
        # plt.savefig(self.output_dir / "visualizations" / "strain_misclassification.pdf", 
        #            bbox_inches='tight')
        # plt.close()
        
        # print(f"Strain misclassification plot saved to {self.output_dir / 'visualizations'}")
        # Create color mapping with individual strain names
        color_mapping = self.create_detailed_strain_color_mapping(strain_order)
        
        # Create INTERACTIVE plot first
        self.create_interactive_misclassification_plot(normalized_incorrect, strain_order, color_mapping, y_true, y_pred)
        
        # Create STATIC plot with detailed legend
        self.create_static_detailed_plot(normalized_incorrect, strain_order, color_mapping, y_true, y_pred)

    def create_detailed_strain_color_mapping(self, strain_list):
        """Create color mapping with similar shades for strain families but distinct colors"""
        import colorsys
        
        # Group strains by family
        strain_families = {'CC00': [], 'BXD': [], 'C3H': [], 'C57': [], 'DBA': [], 'other': []}
        
        for strain in strain_list:
            strain_upper = str(strain).upper()
            assigned = False
            for family in ['CC00', 'BXD', 'C3H', 'C57', 'DBA']:
                if strain_upper.startswith(family):
                    strain_families[family].append(strain)
                    assigned = True
                    break
            if not assigned:
                strain_families['other'].append(strain)
        
        # Base hue values for each family (HSV color space)
        family_hues = {
            'CC00': 0.0,      # Red family
            'BXD': 0.55,      # Cyan family  
            'C3H': 0.67,      # Blue family
            'C57': 0.33,      # Green family
            'DBA': 0.17,      # Yellow family
            'other': 0.83     # Purple family
        }
        
        color_mapping = {}
        
        for family, base_hue in family_hues.items():
            family_strains = strain_families[family]
            if not family_strains:
                continue
            
            # Create distinct colors within the family hue range
            for i, strain in enumerate(family_strains):
                # Vary saturation and value to create distinct but similar colors
                hue = base_hue + (i * 0.05) % 0.1 - 0.05  # Small hue variations
                saturation = 0.4 + (i % 3) * 0.2  # 0.4, 0.6, 0.8
                value = 0.6 + (i % 4) * 0.1       # 0.6, 0.7, 0.8, 0.9
                
                rgb = colorsys.hsv_to_rgb(hue % 1.0, saturation, value)
                hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                color_mapping[strain] = hex_color
        
        return color_mapping

    def create_interactive_misclassification_plot(self, normalized_incorrect, strain_order, color_mapping, y_true, y_pred):
        """Create interactive plotly version with hover information"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig = go.Figure()
        
        # Create stacked bars with hover info
        for predicted_strain in strain_order:
            values = normalized_incorrect.loc[:, predicted_strain].values
            nonzero_mask = values > 0
            
            if np.any(nonzero_mask):
                # Create hover text with detailed information
                hover_text = []
                for i, (true_strain, value) in enumerate(zip(strain_order, values)):
                    if value > 0:
                        # Count actual misclassifications for this pair
                        actual_misclass = sum(1 for t, p in zip(y_true, y_pred) 
                                            if t == true_strain and p == predicted_strain)
                        total_true = sum(1 for t in y_true if t == true_strain)
                        
                        hover_text.append(
                            f"True: {true_strain}<br>"
                            f"Predicted as: {predicted_strain}<br>"
                            f"Misclassifications: {actual_misclass}/{total_true}<br>"
                            f"Proportion: {value:.3f}<br>"
                            f"Family: {self.get_strain_family(predicted_strain)}"
                        )
                    else:
                        hover_text.append("")
                
                # Only add traces for strains with nonzero values
                x_positions = np.arange(len(strain_order))[nonzero_mask]
                y_values = values[nonzero_mask]
                hover_subset = [hover_text[i] for i in range(len(hover_text)) if nonzero_mask[i]]
                
                fig.add_trace(go.Bar(
                    x=x_positions,
                    y=y_values,
                    name=predicted_strain,
                    marker_color=color_mapping.get(predicted_strain, '#808080'),
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=hover_subset,
                    showlegend=False  # Too many for legend
                ))
        
        fig.update_layout(
            title='Interactive Strain Misclassification Analysis<br><sub>Hover over bars for detailed information</sub>',
            xaxis_title='True Strain',
            yaxis_title='Proportion of Incorrect Predictions',
            barmode='stack',
            width=max(1200, len(strain_order) * 20),
            height=600,
            xaxis=dict(
                tickangle=45,
                tickvals=list(range(len(strain_order))),
                ticktext=strain_order
            )
        )
        
        # Save interactive plot
        fig.write_html(self.output_dir / "visualizations" / "interactive_strain_misclassification.html")
        print("Interactive plot saved as HTML")

    def create_static_detailed_plot(self, normalized_incorrect, strain_order, color_mapping, y_true, y_pred):
        """Create static plot with comprehensive legend showing all strain names"""
        
        fig_width = max(25, len(strain_order) * 0.6)
        fig, ax = plt.subplots(figsize=(fig_width, 12))
        
        x_positions = np.arange(len(strain_order))
        bottoms = np.zeros(len(strain_order))
        
        # Track which strains appear in the plot
        strains_in_plot = []
        
        # Create stacked bars
        for predicted_strain in strain_order:
            values = normalized_incorrect.loc[:, predicted_strain].values
            nonzero_mask = values > 0
            
            if np.any(nonzero_mask):
                ax.bar(x_positions[nonzero_mask], 
                    values[nonzero_mask],
                    bottom=bottoms[nonzero_mask],
                    color=color_mapping.get(predicted_strain, '#808080'),
                    label=predicted_strain,
                    alpha=0.9,
                    edgecolor='white',
                    linewidth=0.3)
                
                bottoms += values
                strains_in_plot.append(predicted_strain)
        
        # Customize the plot
        ax.set_xlabel('True Strain', fontsize=12, fontweight='bold')
        ax.set_ylabel('Proportion of Incorrect Predictions', fontsize=12, fontweight='bold')
        ax.set_title('Strain Misclassification Analysis - All Strains with Individual Colors', 
                    fontsize=14, fontweight='bold')
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(strain_order, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Create comprehensive legend organized by families
        self.create_organized_legend(ax, strains_in_plot, color_mapping)
        
        # Add statistics box
        total_samples = len(y_true)
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        
        stats_text = f'Total samples: {total_samples}\nOverall accuracy: {accuracy:.3f}\nTotal strains: {len(strain_order)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save static plot
        plt.savefig(self.output_dir / "visualizations" / "detailed_strain_misclassification.png", 
                dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "visualizations" / "detailed_strain_misclassification.pdf", 
                bbox_inches='tight')
        plt.close()

    def create_organized_legend(self, ax, strains_in_plot, color_mapping):
        """Create organized legend by strain families"""
        # Group strains by family for legend
        family_groups = {'CC00': [], 'BXD': [], 'C3H': [], 'C57': [], 'DBA': [], 'Other': []}
        
        for strain in strains_in_plot:
            strain_upper = str(strain).upper()
            assigned = False
            for family in ['CC00', 'BXD', 'C3H', 'C57', 'DBA']:
                if strain_upper.startswith(family):
                    family_groups[family].append(strain)
                    assigned = True
                    break
            if not assigned:
                family_groups['Other'].append(strain)
        
        # Create legend with family sections
        legend_elements = []
        
        for family, strains in family_groups.items():
            if strains:
                # Add family header (invisible element for spacing)
                legend_elements.append(plt.Line2D([0], [0], color='none', label=f'--- {family} ---'))
                
                # Sort strains within family
                strains.sort()
                for strain in strains:
                    legend_elements.append(plt.Rectangle((0,0),1,1, 
                                                    facecolor=color_mapping.get(strain, '#808080'),
                                                    alpha=0.9, label=strain))
        
        # Create legend outside plot area
        ax.legend(handles=legend_elements, 
                bbox_to_anchor=(1.02, 1), loc='upper left',
                title='Predicted as (by family)', 
                ncol=max(1, len(strains_in_plot) // 25),  # Multiple columns for many strains
                fontsize=8)

    def get_strain_family(self, strain):
        """Helper function to get strain family for hover info"""
        strain_upper = str(strain).upper()
        for family in ['CC00', 'BXD', 'C3H', 'C57', 'DBA']:
            if strain_upper.startswith(family):
                return family
        return 'Other'
    
    def create_summary_visualizations(self):
        """Create summary visualizations of all results"""
        print("Creating summary visualizations...")
        
        # Age regression performance by strain
        if 'age_regression' in self.results:
            self.plot_age_regression_by_strain()
        
        # Overall summary plot
        self.create_overall_summary_plot()
    
    def plot_age_regression_by_strain(self):
        """Plot age regression performance by strain"""
        for suffix, results in self.results['age_regression'].items():
            if 'logo_results' in results:
                group_results = results['logo_results']['group_results']
                
                strains = list(group_results.keys())
                rmse_values = [group_results[strain]['rmse'] for strain in strains]
                
                # Sort by RMSE
                sorted_data = sorted(zip(strains, rmse_values), key=lambda x: x[1])
                strains_sorted, rmse_sorted = zip(*sorted_data)
                
                fig, ax = plt.subplots(figsize=(max(12, len(strains) * 0.3), 8))
                
                bars = ax.bar(range(len(strains_sorted)), rmse_sorted)
                ax.set_xticks(range(len(strains_sorted)))
                ax.set_xticklabels(strains_sorted, rotation=45, ha='right')
                ax.set_ylabel('RMSE (days)')
                ax.set_title(f'Age Regression RMSE by Strain ({suffix})')
                ax.grid(True, alpha=0.3)
                
                # Color bars
                colors = plt.cm.RdYlGn_r(np.linspace(0.3, 1.0, len(rmse_sorted)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / "visualizations" / f"age_regression_by_strain_{suffix}.png", 
                           dpi=300, bbox_inches='tight')
                plt.savefig(self.output_dir / "visualizations" / f"age_regression_by_strain_{suffix}.pdf", 
                           bbox_inches='tight')
                plt.close()
    
    def create_overall_summary_plot(self):
        """Create an overall summary plot of all analyses"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Age regression summary
        if 'age_regression' in self.results:
            methods = list(self.results['age_regression'].keys())
            rmse_values = [self.results['age_regression'][method]['logo_results']['mean_rmse'] 
                          for method in methods]
            r2_values = [self.results['age_regression'][method]['logo_results']['mean_r2'] 
                        for method in methods]
            
            axes[0, 0].bar(methods, rmse_values, alpha=0.7, color='red')
            axes[0, 0].set_title('Age Regression RMSE')
            axes[0, 0].set_ylabel('RMSE (days)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            axes[0, 1].bar(methods, r2_values, alpha=0.7, color='blue')
            axes[0, 1].set_title('Age Regression R²')
            axes[0, 1].set_ylabel('R² Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Strain classification summary
        if 'strain_classification' in self.results:
            accuracy = self.results['strain_classification']['accuracy']
            f1_score = self.results['strain_classification']['f1_weighted']
            
            axes[1, 0].bar(['Accuracy', 'F1-Score'], [accuracy, f1_score], 
                          alpha=0.7, color='green')
            axes[1, 0].set_title('Strain Classification Performance')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_ylim(0, 1)
        
        # Dataset summary
        dataset_info = [
            f"Samples: {self.X.shape[0]}",
            f"Dimensions: {self.X.shape[1]}",
            f"Strains: {self.metadata_aligned['strain'].nunique()}",
            f"Age range: {self.metadata_aligned['avg_age_days_chunk_start'].min():.0f}-{self.metadata_aligned['avg_age_days_chunk_start'].max():.0f} days"
        ]
        
        axes[1, 1].text(0.1, 0.5, '\n'.join(dataset_info), transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_title('Dataset Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "visualizations" / "overall_summary.png", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "visualizations" / "overall_summary.pdf", 
                   bbox_inches='tight')
        plt.close()
    
    def save_complete_results(self):
        """Save all results to a comprehensive JSON file"""
        print("Saving complete results...")
        
        # Add metadata
        self.results['metadata'] = {
            'analysis_timestamp': datetime.now().isoformat(),
            'embeddings_path': str(self.embeddings_path),
            'metadata_path': str(self.metadata_path),
            'output_dir': str(self.output_dir),
            'dataset_info': {
                'n_samples': int(self.X.shape[0]),
                'n_dimensions': int(self.X.shape[1]),
                'n_strains': int(self.metadata_aligned['strain'].nunique()),
                'age_range': [
                    float(self.metadata_aligned['avg_age_days_chunk_start'].min()),
                    float(self.metadata_aligned['avg_age_days_chunk_start'].max())
                ],
                'strain_list': sorted(self.metadata_aligned['strain'].unique().tolist())
            },
            'aggregation_info': self.aggregation_info
        }
        
        # Save complete results
        with open(self.output_dir / "complete_analysis_results.json", 'w') as f:
            json.dump(convert_numpy_types(self.results), f, indent=2)
        
        # Create summary text file
        with open(self.output_dir / "analysis_summary.txt", 'w') as f:
            f.write("COMPLETE EMBEDDING ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis completed: {self.results['metadata']['analysis_timestamp']}\n")
            f.write(f"Dataset: {self.X.shape[0]} samples, {self.X.shape[1]} dimensions\n")
            f.write(f"Strains: {self.metadata_aligned['strain'].nunique()}\n")
            f.write(f"Age range: {self.metadata_aligned['avg_age_days_chunk_start'].min():.1f} - "
                   f"{self.metadata_aligned['avg_age_days_chunk_start'].max():.1f} days\n\n")
            
            # Age regression results
            if 'age_regression' in self.results:
                f.write("AGE REGRESSION RESULTS:\n")
                f.write("-" * 30 + "\n")
                for method, results in self.results['age_regression'].items():
                    f.write(f"\n{method.upper()}:\n")
                    logo_results = results['logo_results']
                    f.write(f"  RMSE: {logo_results['mean_rmse']:.2f} ± {logo_results['std_rmse']:.2f} days\n")
                    f.write(f"  R²: {logo_results['mean_r2']:.3f} ± {logo_results['std_r2']:.3f}\n")
                    f.write(f"  MAE: {logo_results['mean_mae']:.2f} ± {logo_results['std_mae']:.2f} days\n")
            
            # Strain classification results
            if 'strain_classification' in self.results:
                f.write(f"\nSTRAIN CLASSIFICATION RESULTS:\n")
                f.write("-" * 30 + "\n")
                sc_results = self.results['strain_classification']
                f.write(f"  Accuracy: {sc_results['accuracy']:.3f}\n")
                f.write(f"  F1-weighted: {sc_results['f1_weighted']:.3f}\n")
                f.write(f"  Test samples: {sc_results['n_test_samples']}\n")
            
            f.write(f"\nFiles saved to: {self.output_dir}\n")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting complete embedding analysis pipeline...")
        print("=" * 70)
        
        # 1. Age regression analysis (without bodyweight)
        print("\n1. Age regression analysis (without bodyweight)...")
        self.perform_age_regression_analysis(include_bodyweight=False)
        
        # 2. Age regression analysis (with bodyweight if available)
        print("\n2. Age regression analysis (with bodyweight)...")
        if 'bodyweight' in self.metadata_aligned.columns:
            self.perform_age_regression_analysis(include_bodyweight=True)
        else:
            print("Bodyweight data not available, skipping bodyweight analysis")
        
        # 3. Strain classification analysis
        print("\n3. Strain classification analysis...")
        self.perform_strain_classification_analysis()
        
        # 4. Create visualizations
        print("\n4. Creating visualizations...")
        self.create_summary_visualizations()
        
        # 5. Save all results
        print("\n5. Saving complete results...")
        self.save_complete_results()
        
        print("\n" + "=" * 70)
        print("COMPLETE ANALYSIS FINISHED!")
        print(f"Results saved to: {self.output_dir}")
        
        # Print key findings
        print("\nKEY FINDINGS:")
        if 'age_regression' in self.results:
            for method, results in self.results['age_regression'].items():
                logo_results = results['logo_results']
                print(f"• Age regression ({method}): RMSE = {logo_results['mean_rmse']:.2f} days, "
                      f"R² = {logo_results['mean_r2']:.3f}")
        
        if 'strain_classification' in self.results:
            sc_results = self.results['strain_classification']
            print(f"• Strain classification: Accuracy = {sc_results['accuracy']:.3f}, "
                  f"F1 = {sc_results['f1_weighted']:.3f}")
        
        print(f"\nDetailed results available in:")
        print(f"  - {self.output_dir}/complete_analysis_results.json")
        print(f"  - {self.output_dir}/predictions/ (for raw predictions)")
        print(f"  - {self.output_dir}/visualizations/ (for all plots)")
        print(f"  - {self.output_dir}/analysis_summary.txt")


def main():
    """Main function to run complete embedding analysis"""
    parser = argparse.ArgumentParser(description='Complete Embedding Analysis Pipeline')
    parser.add_argument('embeddings_path', help='Path to embeddings .npy file')
    parser.add_argument('metadata_path', help='Path to metadata CSV file')
    parser.add_argument('--output_dir', default='complete_analysis_results', 
                       help='Output directory for results')
    parser.add_argument('--max_workers', type=int, default=8, 
                       help='Maximum number of worker threads')
    
    # Bodyweight integration arguments
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
        print("Bodyweight integration enabled")
    else:
        print("Bodyweight integration disabled (files not provided)")
    
    # Create and run analyzer
    analyzer = CompleteEmbeddingAnalyzer(
        embeddings_path=args.embeddings_path,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        bodyweight_files=bodyweight_files
    )
    
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main() 