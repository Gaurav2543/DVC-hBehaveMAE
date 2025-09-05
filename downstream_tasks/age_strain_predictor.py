#!/usr/bin/env python3
# unified_evaluate_embeddings.py

import argparse
import warnings
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (mean_absolute_error, r2_score, classification_report, 
                           ConfusionMatrixDisplay, root_mean_squared_error, mean_squared_error,
                           accuracy_score, f1_score)
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial
from pygam import LinearGAM, s

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def get_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(
        description="Unified evaluation of learned embeddings with linear models and multiple algorithms."
    )
    p.add_argument("--embeddings", required=True, help="Path to embeddings file (*.npy or *.npz)")
    p.add_argument("--labels", required=True, help="Path to labels CSV file")
    p.add_argument("--task", required=True, choices=["regression", "classification", "both"],
                   help="The evaluation task to perform.")
    p.add_argument("--target-label", required=True,
                   help="The column name in the CSV file to predict (e.g., 'avg_age_days_chunk_start').")
    p.add_argument("--group-label", default="strain",
                   help="The column name to group by for Leave-One-Out validation (e.g., 'strain').")
    p.add_argument("--classification-target", default=None,
                   help="Column name for classification target (defaults to group-label if not specified).")
    p.add_argument("--test-size", type=float, default=0.25,
                   help="Proportion of the dataset for the test split (default: 0.25 for a 75-25 split).")
    p.add_argument("--plot-confusion-matrix", action="store_true",
                   help="If set, saves the confusion matrix as a PNG file.")
    p.add_argument("--output-dir", default="evaluation_outputs",
                   help="Directory to save output plots and results.")
    p.add_argument("--results-json", default="evaluation_results.json",
                   help="Name of JSON file to save detailed results.")
    p.add_argument("--aggregation-method", choices=["mean", "median", "percentiles"], default="mean",
                   help="Method for aggregating frame-level embeddings.")
    p.add_argument("--enable-multi-algorithm", action="store_true",
                   help="Enable evaluation with multiple algorithms (RF, SVM) in addition to linear models.")
    
    # Covariate arguments
    p.add_argument("--use-covariate", action="store_true",
                   help="Include bodyweight as a covariate in regression tasks.")

    p.add_argument("--covariate-name", default="bodyweight",
                   help="Name of the covariate to use (default: bodyweight).")
    p.add_argument("--summary-metadata", default=None,
                   help="Path to summary_metadata.csv file (required if using covariates).")
    p.add_argument("--bodyweight-file", default=None,
                   help="Path to BW.csv file (required if using covariates).")
    p.add_argument("--tissue-collection-file", default=None,
                   help="Path to TissueCollection_PP.csv file (required if using covariates).")
    p.add_argument("--covariate-strategy", choices=["closest", "interpolate", "most_recent", "gam_spline"], 
                    default="closest", help="Strategy for matching bodyweight to timepoints.")
    p.add_argument("--cage-aggregation", choices=["mean", "median", "first"], default="mean",
                   help="How to aggregate bodyweight across multiple animals in a cage.")
    
    p.add_argument("--sliding-window", action="store_true",
               help="Perform sliding window evaluation in addition to standard evaluation.")
    p.add_argument("--window-sizes", nargs='+', type=int, default=[30],
                help="Window sizes in days for sliding window evaluation (default: [30]).")
    p.add_argument("--window-step", type=int, default=15,
                help="Step size in days for sliding window evaluation (default: 15).")
    p.add_argument("--min-samples-per-window", type=int, default=10,
                help="Minimum number of samples required per window (default: 10).")
    
    # Performance arguments
    p.add_argument("--max-workers", type=int, default=4,
                   help="Maximum number of worker threads for parallel processing.")
    
    return p.parse_args()

def load_bodyweight_data(bw_file: str, tissue_file: str) -> pd.DataFrame:
    """Load and merge bodyweight data from BW.csv and TissueCollection_PP.csv."""
    print("[INFO] Loading bodyweight data...")
    
    # Load BW data
    bw_df = pd.read_csv(bw_file)
    bw_df['date'] = pd.to_datetime(bw_df['date'])
    
    # Load tissue collection data
    tissue_df = pd.read_csv(tissue_file)
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
    combined_bw = combined_bw.drop_duplicates(subset=['animal_id', 'date']).sort_values(['animal_id', 'date'])
    
    print(f"[INFO] Loaded bodyweight data for {combined_bw['animal_id'].nunique()} animals "
          f"with {len(combined_bw)} total measurements")
    
    return combined_bw

def match_bodyweight_to_timepoint(animal_ids: list, target_date: datetime, bw_data: pd.DataFrame,
                                strategy: str = "closest") -> float:
    """Match bodyweight measurement to a specific timepoint for given animals."""
    if not animal_ids:
        return np.nan
    
    # Get bodyweight data for all animals in the cage
    animal_bw_data = bw_data[bw_data['animal_id'].isin(animal_ids)].copy()
    
    if animal_bw_data.empty:
        return np.nan
    
    if strategy == "gam_spline":
        # Use GAM spline interpolation per animal, then average
        weights = []
        for animal_id in animal_ids:
            animal_data = animal_bw_data[animal_bw_data['animal_id'] == animal_id].sort_values('date')
            if len(animal_data) < 3:  # Need at least 3 points for GAM
                if len(animal_data) >= 1:
                    # Fall back to closest for insufficient data
                    animal_data['time_diff'] = abs((animal_data['date'] - target_date).dt.total_seconds())
                    weights.append(animal_data.loc[animal_data['time_diff'].idxmin()]['body_weight'])
                continue
            
            try:
                # Convert dates to days from first measurement for GAM
                first_date = animal_data['date'].min()
                t = (animal_data['date'] - first_date).dt.total_seconds() / (24 * 3600)  # Convert to days
                y = animal_data['body_weight'].values
                target_t = (target_date - first_date).total_seconds() / (24 * 3600)
                
                # Fit GAM with adaptive number of splines
                n_splines = min(10, len(t) - 1, max(3, len(t) // 3))
                gam = LinearGAM(s(0, n_splines=n_splines)).fit(t, y)
                
                # Predict at target time (extrapolation allowed)
                predicted_weight = gam.predict([target_t])[0]
                
                # Sanity check: ensure prediction is reasonable
                min_weight, max_weight = y.min(), y.max()
                weight_range = max_weight - min_weight
                if min_weight - weight_range <= predicted_weight <= max_weight + weight_range:
                    weights.append(predicted_weight)
                else:
                    # Fall back to closest if prediction is unreasonable
                    animal_data['time_diff'] = abs((animal_data['date'] - target_date).dt.total_seconds())
                    weights.append(animal_data.loc[animal_data['time_diff'].idxmin()]['body_weight'])
                    
            except Exception:
                # Fall back to closest if GAM fails
                animal_data['time_diff'] = abs((animal_data['date'] - target_date).dt.total_seconds())
                weights.append(animal_data.loc[animal_data['time_diff'].idxmin()]['body_weight'])
        
        return np.mean(weights) if weights else np.nan
    
    elif strategy == "closest":
        # Find the measurement closest in time to target_date
        animal_bw_data['time_diff'] = abs((animal_bw_data['date'] - target_date).dt.total_seconds())
        closest_measurements = animal_bw_data.loc[animal_bw_data.groupby('animal_id')['time_diff'].idxmin()]
        return closest_measurements['body_weight'].mean()
    
    elif strategy == "most_recent":
        # Use the most recent measurement before or on target_date
        valid_measurements = animal_bw_data[animal_bw_data['date'] <= target_date]
        if valid_measurements.empty:
            return np.nan
        most_recent = valid_measurements.loc[valid_measurements.groupby('animal_id')['date'].idxmax()]
        return most_recent['body_weight'].mean()
    
    elif strategy == "interpolate":
        # Simple linear interpolation (existing code)
        weights = []
        for animal_id in animal_ids:
            animal_data = animal_bw_data[animal_bw_data['animal_id'] == animal_id].sort_values('date')
            if len(animal_data) < 2:
                if len(animal_data) == 1:
                    weights.append(animal_data.iloc[0]['body_weight'])
                continue
            
            # Find surrounding measurements
            before = animal_data[animal_data['date'] <= target_date]
            after = animal_data[animal_data['date'] > target_date]
            
            if before.empty and not after.empty:
                weights.append(after.iloc[0]['body_weight'])
            elif not before.empty and after.empty:
                weights.append(before.iloc[-1]['body_weight'])
            elif not before.empty and not after.empty:
                # Linear interpolation
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

def process_metadata_with_bodyweight(metadata: pd.DataFrame, summary_metadata: pd.DataFrame,
                                   bw_data: pd.DataFrame, strategy: str, cage_agg: str) -> pd.DataFrame:
    """Process metadata to include bodyweight covariates."""
    print("[INFO] Processing metadata with bodyweight covariates...")
    
    # Prepare summary metadata
    summary_metadata['from_tpt'] = pd.to_datetime(summary_metadata['from_tpt'])
    summary_metadata['to_tpt'] = pd.to_datetime(summary_metadata['to_tpt'])
    
    # Validate 72-hour difference
    time_diffs = (summary_metadata['to_tpt'] - summary_metadata['from_tpt']).dt.total_seconds() / 3600
    non_72h = summary_metadata[abs(time_diffs - 72) > 1]  # Allow 1-hour tolerance
    if len(non_72h) > 0:
        print(f"[WARNING] Found {len(non_72h)} entries with non-72h time differences")
    
    # Create bodyweight column
    bodyweights = []
    
    def process_single_row(row):
        # Parse animal IDs
        if pd.isna(row.animal_ids) or row.animal_ids == '':
            return np.nan
        animal_ids = [aid.strip() for aid in str(row.animal_ids).split(';') if aid.strip()]
        
        # Use midpoint of time range as target
        target_date = row.from_tpt + (row.to_tpt - row.from_tpt) / 2
        
        return match_bodyweight_to_timepoint(animal_ids, target_date, bw_data, strategy)
    
    # Process with progress bar and threading
    with ThreadPoolExecutor(max_workers=8) as executor:
        with tqdm(total=len(summary_metadata), desc="Processing bodyweight data") as pbar:
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
    
    # Merge with original metadata based on cage_id and timepoint
    metadata['from_tpt'] = pd.to_datetime(metadata['from_tpt'])
    merged = metadata.merge(
        summary_metadata[['cage_id', 'from_tpt', 'bodyweight']], 
        on=['cage_id', 'from_tpt'], 
        how='left'
    )
    
    valid_bw = merged['bodyweight'].notna().sum()
    print(f"[INFO] Successfully matched bodyweight for {valid_bw}/{len(merged)} entries "
          f"({100*valid_bw/len(merged):.1f}%)")
    
    return merged

def load_data(emb_path: str, labels_path: str, target_label: str, task: str, 
              group_label: str, classification_target: str = None, 
              aggregation_method: str = "mean", use_covariate: bool = False,
              covariate_files: dict = None, covariate_strategy: str = "closest",
              cage_aggregation: str = "mean") -> tuple:
    """Loads embeddings and labels, aligns them, and removes missing values."""
    print(f"[INFO] Loading embeddings from {emb_path}...")
    
    # Load the file, which could be a standard array or a wrapped object
    loaded_obj = np.load(emb_path, allow_pickle=True)

    # Check if the loaded object is a 0-d array containing a dictionary
    if loaded_obj.shape == () and isinstance(loaded_obj.item(), dict):
        embeddings_data = loaded_obj.item()
        embeddings = embeddings_data['embeddings']
        frame_map = embeddings_data.get('frame_number_map', None)
    # Check if it's an NPZ file with an 'embeddings' key
    elif isinstance(loaded_obj, np.lib.npyio.NpzFile):
        embeddings = loaded_obj['embeddings']
        frame_map = loaded_obj.get('frame_number_map', None)
    # Otherwise, assume it's a standard numpy array
    else:
        embeddings = loaded_obj
        frame_map = None

    print(f"[INFO] Loading labels from {labels_path}...")
    labels_df = pd.read_csv(labels_path)
    
    # Load and process bodyweight data if requested
    if use_covariate and covariate_files:
        print("[INFO] Loading covariate data...")
        bw_data = load_bodyweight_data(covariate_files['bodyweight'], covariate_files['tissue'])
        summary_metadata = pd.read_csv(covariate_files['summary'])
        labels_df = process_metadata_with_bodyweight(
            labels_df, summary_metadata, bw_data, covariate_strategy, cage_aggregation
        )
    
    # Handle frame mapping if available
    if frame_map is not None:
        print(f"[INFO] Using frame mapping for aggregation method: {aggregation_method}")
        return load_data_with_frame_mapping(embeddings, frame_map, labels_df, target_label, 
                                          task, group_label, classification_target, 
                                          aggregation_method, use_covariate)
    
    # Define required columns based on the task
    required_cols = [target_label]
    if task in ['regression', 'both']:
        required_cols.append(group_label)
    if task in ['classification', 'both'] and classification_target:
        if classification_target not in required_cols:
            required_cols.append(classification_target)
    if use_covariate:
        required_cols.append('bodyweight')

    for col in required_cols:
        if col not in labels_df.columns:
            raise ValueError(f"Required label '{col}' not found in the CSV file.")

    n_samps = min(len(embeddings), len(labels_df))
    embeddings = embeddings[:n_samps]
    labels_df = labels_df.iloc[:n_samps].copy()

    # Create a working copy and handle missing values
    work_df = labels_df[required_cols].copy()
    work_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    work_df.dropna(inplace=True)

    if work_df.empty:
        raise ValueError(f"No valid data remains for targets after removing NaNs.")

    aligned_embeddings = embeddings[work_df.index]
    
    # Add covariate to embeddings if requested
    if use_covariate:
        bodyweight_values = work_df['bodyweight'].values.reshape(-1, 1)
        aligned_embeddings = np.concatenate([aligned_embeddings, bodyweight_values], axis=1)
        print(f"[INFO] Added bodyweight covariate. New embedding dimension: {aligned_embeddings.shape[1]}")
    
    y_regression = work_df[target_label].values if task in ['regression', 'both'] else None
    groups = work_df[group_label].values if task in ['regression', 'both'] else None
    
    if task in ['classification', 'both']:
        class_target = classification_target if classification_target else group_label
        y_classification = work_df[class_target].values
        class_names = sorted(list(np.unique(y_classification)))
    else:
        y_classification = None
        class_names = None

    print(f"[INFO] Data loaded. Found {len(work_df)} valid samples for evaluation.")
    return aligned_embeddings, y_regression, y_classification, groups, class_names

def load_data_with_frame_mapping(embeddings, frame_map, metadata, target_label, task, 
                               group_label, classification_target, aggregation_method, use_covariate):
    """Handle data loading when frame mapping is available."""
    # Prepare metadata
    if 'from_tpt' in metadata.columns:
        metadata["from_tpt"] = pd.to_datetime(metadata["from_tpt"])
        metadata["key"] = metadata.apply(lambda row: f"{row['cage_id']}_{row['from_tpt'].strftime('%Y-%m-%d')}", axis=1)
    else:
        # Assume there's already a key column or create one
        if 'key' not in metadata.columns:
            metadata['key'] = metadata.index.astype(str)
    
    X, y_reg, y_class, groups_list, bodyweights, skipped = [], [], [], [], [], 0
    
    # Process with progress bar
    with tqdm(total=len(metadata), desc="Processing frame mappings") as pbar:
        for row in metadata.itertuples():
            key = getattr(row, 'key', str(row.Index))
            if not key or key not in frame_map:
                skipped += 1
                pbar.update(1)
                continue
                
            start, end = frame_map[key]
            if end <= start or end > embeddings.shape[0]:
                skipped += 1
                pbar.update(1)
                continue
                
            emb = embeddings[start:end]
            if emb.shape[0] == 0 or np.isnan(emb).any():
                skipped += 1
                pbar.update(1)
                continue

            # Apply aggregation method
            if aggregation_method == "mean":
                flat = emb.mean(axis=0)
            elif aggregation_method == "median":
                flat = np.median(emb, axis=0)
            elif aggregation_method == "percentiles":
                flat = np.percentile(emb, [25, 50, 75], axis=0).flatten()
            else:
                flat = emb.mean(axis=0)  # Default to mean

            if np.isnan(flat).any():
                skipped += 1
                pbar.update(1)
                continue

            X.append(flat)
            if task in ['regression', 'both']:
                y_reg.append(getattr(row, target_label))
                groups_list.append(getattr(row, group_label))
            
            if task in ['classification', 'both']:
                class_target = classification_target if classification_target else group_label
                y_class.append(getattr(row, class_target))
            
            if use_covariate:
                bodyweights.append(getattr(row, 'bodyweight', np.nan))
            
            pbar.update(1)

    print(f"[INFO] Processed {len(X)} samples, skipped {skipped} due to missing data or frame mapping issues.")
    
    X = np.array(X)
    
    # Add covariate if requested
    if use_covariate and bodyweights:
        bodyweight_array = np.array(bodyweights).reshape(-1, 1)
        # Remove samples with missing bodyweight
        valid_bw_mask = ~np.isnan(bodyweight_array.flatten())
        X = X[valid_bw_mask]
        bodyweight_array = bodyweight_array[valid_bw_mask]
        y_reg = np.array(y_reg)[valid_bw_mask] if y_reg else None
        y_class = np.array(y_class)[valid_bw_mask] if y_class else None
        groups_list = np.array(groups_list)[valid_bw_mask] if groups_list else None
        
        X = np.concatenate([X, bodyweight_array], axis=1)
        print(f"[INFO] Added bodyweight covariate. New embedding dimension: {X.shape[1]}")
        print(f"[INFO] Valid samples after bodyweight filtering: {len(X)}")
    else:
        y_reg = np.array(y_reg) if y_reg else None
        y_class = np.array(y_class) if y_class else None
        groups_list = np.array(groups_list) if groups_list else None
    
    y_regression = y_reg
    y_classification = y_class
    groups = groups_list
    class_names = sorted(list(np.unique(y_classification))) if y_classification is not None else None
    
    return X, y_regression, y_classification, groups, class_names

def train_single_model(args):
    """Train a single model for one fold - used for parallel processing."""
    model_class, X_train, X_test, y_train, y_test, fold_info = args
    
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = model_class.fit(X_train_scaled, y_train) if hasattr(model_class, 'fit') else model_class.__class__().fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics for this fold
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    
    baseline_pred = np.full_like(y_test, fill_value=y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    
    return {
        'fold_info': fold_info,
        'mae': mae,
        'r2': r2,
        'rmse': rmse,
        'baseline_mae': baseline_mae,
        'sample_count': len(y_test)
    }

def evaluate_regression_logo(X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
                           output_dir: str, enable_multi: bool = False, max_workers: int = 4):
    """Performs Leave-One-Group-Out regression evaluation with multiple algorithms."""
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(groups=groups)
    unique_groups = np.unique(groups)
    print(f"\n[INFO] Starting Leave-One-Group-Out validation across {n_splits} groups...")

    # Define models
    models = {
        "LinearRegression": LinearRegression(n_jobs=-1),
        
    }
    if enable_multi:
        models.update({
            "Ridge": Ridge(alpha=1.0),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "SVR": SVR()
        })

    # Store results for each model
    all_results = {}
    
    for model_name, model_class in models.items():
        print(f"\n--- Evaluating {model_name} ---")
        
        # Prepare arguments for parallel processing
        fold_args = []
        for i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            held_out_group = unique_groups[i]
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            fold_args.append((model_class, X_train, X_test, y_train, y_test, held_out_group))
        
        # Process folds in parallel
        fold_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=len(fold_args), desc=f"Training {model_name}") as pbar:
                future_to_fold = {executor.submit(train_single_model, args): args for args in fold_args}
                
                for future in as_completed(future_to_fold):
                    result = future.result()
                    fold_results.append(result)
                    pbar.update(1)
        
        # Aggregate results
        maes = [r['mae'] for r in fold_results]
        r2s = [r['r2'] for r in fold_results]
        rmses = [r['rmse'] for r in fold_results]
        baseline_maes = [r['baseline_mae'] for r in fold_results]
        
        group_results = {r['fold_info']: {
            'mae': r['mae'], 'r2': r['r2'], 'rmse': r['rmse'], 'sample_count': r['sample_count']
        } for r in fold_results}

        # Calculate final mean and std dev of scores
        mean_mae, std_mae = np.mean(maes), np.std(maes)
        mean_r2, std_r2 = np.mean(r2s), np.std(r2s)
        mean_rmse, std_rmse = np.mean(rmses), np.std(rmses)
        mean_baseline_mae = np.mean(baseline_maes)

        all_results[model_name] = {
            'mean_mae': mean_mae,
            'std_mae': std_mae,
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'mean_baseline_mae': mean_baseline_mae,
            'per_group_results': group_results
        }

        print(f"\n{model_name} Results:")
        print(f"  Mean Absolute Error (MAE): {mean_mae:.2f} (± {std_mae:.2f})")
        print(f"  Mean RMSE:                 {mean_rmse:.2f} (± {std_rmse:.2f})")
        print(f"  Mean R-squared (R²):       {mean_r2:.3f} (± {std_r2:.3f})")
        print(f"  Mean Baseline MAE:         {mean_baseline_mae:.2f}")

        # Create visualization for this model
        create_regression_plot(group_results, model_name, output_dir)

    return all_results

def create_regression_plot(group_results, model_name, output_dir):
    """Create per-group RMSE visualization."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by RMSE
    sorted_results = sorted(group_results.items(), key=lambda x: x[1]['rmse'])
    group_names = [x[0] for x in sorted_results]
    rmse_values = [x[1]['rmse'] for x in sorted_results]
    counts = [x[1]['sample_count'] for x in sorted_results]
    avg_rmse = np.mean(rmse_values)

    fig, ax1 = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(group_names))
    ax1.barh(y_pos, rmse_values, color="black", label="RMSE")
    ax1.set_xlabel("RMSE", fontsize=12)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(group_names, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_title(f"Per-Group RMSE ({model_name}) | Avg RMSE: {avg_rmse:.2f}", fontsize=14)

    ax2 = ax1.twiny()
    ax2.barh(y_pos, counts, color="gray", alpha=0.5, label="Sample Count")
    ax2.set_xlabel("Sample Count", fontsize=12)

    fig.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_group_rmse.pdf"))
    plt.savefig(os.path.join(output_dir, f"{model_name}_group_rmse.png"))
    plt.close()

def train_single_classifier(args):
    """Train a single classifier - used for parallel processing."""
    model_class, X_train, X_test, y_train, y_test = args
    
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = model_class.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    
    return model, y_pred, accuracy, f1_macro

def evaluate_classification(X: np.ndarray, y: np.ndarray, test_size: float, class_names: list, 
                          plot_cm: bool, output_dir: str, enable_multi: bool = False, max_workers: int = 4):
    """Performs classification evaluation using a train-test split with multiple algorithms."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )

    # Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    }
    if enable_multi:
        models.update({
            "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "SVC": SVC(random_state=42)
        })

    all_results = {}
    os.makedirs(output_dir, exist_ok=True)

    # Process models with progress bar
    with tqdm(total=len(models), desc="Training classifiers") as pbar:
        for model_name, model in models.items():
            print(f"\n--- Training {model_name} ---")
            
            # Train model (could be parallelized for cross-validation, but single train-test split is fast)
            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Overall metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average="macro")
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

            print(f"  Accuracy:  {accuracy:.3f}")
            print(f"  F1 Macro:  {f1_macro:.3f}")

            # Per-class accuracy
            class_accuracies = {}
            class_counts = {}
            for i, class_name in enumerate(le.classes_):
                true_idx = y_test == i
                if np.sum(true_idx) > 0:
                    acc_i = (y_pred[true_idx] == i).mean()
                    class_accuracies[class_name] = acc_i
                    class_counts[class_name] = np.sum(true_idx)
                else:
                    class_accuracies[class_name] = 0
                    class_counts[class_name] = 0

            all_results[model_name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'classification_report': report,
                'per_class_accuracy': class_accuracies,
                'per_class_counts': class_counts
            }

            # Create per-class accuracy plot
            create_classification_plot(class_accuracies, class_counts, model_name, output_dir)

            # Create confusion matrix for first model or if specifically requested
            if plot_cm and (model_name == "LogisticRegression" or len(models) == 1):
                fig, ax = plt.subplots(figsize=(10, 10))
                ConfusionMatrixDisplay.from_predictions(
                    y_test, y_pred, ax=ax, display_labels=le.classes_,
                    xticks_rotation='vertical', values_format='d', cmap='Blues'
                )
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"confusion_matrix_{model_name}.png"))
                plt.close()
            
            pbar.update(1)

    return all_results

def create_classification_plot(class_accuracies, class_counts, model_name, output_dir):
    """Create per-class accuracy visualization."""
    # Sort by accuracy (descending)
    sorted_items = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
    class_names = [x[0] for x in sorted_items]
    accuracies = [x[1] for x in sorted_items]
    counts = [class_counts[x] for x in class_names]
    avg_acc = np.mean(accuracies)

    fig, ax1 = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(class_names))
    ax1.barh(y_pos, accuracies, color="black", label="Accuracy")
    ax1.set_xlabel("Accuracy", fontsize=12)
    ax1.set_xlim(0, 1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(class_names, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_title(f"{model_name} Per-Class Accuracy | Avg Acc: {avg_acc:.3f}", fontsize=14)

    ax2 = ax1.twiny()
    ax2.barh(y_pos, counts, color="gray", alpha=0.5, label="Sample Count")
    ax2.set_xlabel("Sample Count", fontsize=12)

    fig.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_class_accuracy.pdf"))
    plt.savefig(os.path.join(output_dir, f"{model_name}_class_accuracy.png"))
    plt.close()
    
def perform_sliding_window_evaluation(X: np.ndarray, y_regression: np.ndarray, y_classification: np.ndarray,
                                    groups: np.ndarray, metadata: pd.DataFrame, window_sizes: list,
                                    window_step: int, min_samples: int, output_dir: str, 
                                    target_label: str, enable_multi: bool = False):
    """Perform sliding window evaluation for both regression and classification."""
    
    # Ensure we have age data for windowing
    if target_label not in metadata.columns:
        raise ValueError(f"Target label '{target_label}' not found in metadata for sliding window evaluation")
    
    ages = metadata[target_label].values
    
    results = {}
    
    for window_size in tqdm(window_sizes, desc="Processing window sizes"):
        print(f"\n[INFO] Processing window size: {window_size} days")
        
        # Calculate window boundaries
        min_age, max_age = np.nanmin(ages), np.nanmax(ages)
        window_starts = np.arange(min_age, max_age - window_size + 1, window_step)
        
        window_results = {
            'window_size': window_size,
            'window_step': window_step,
            'regression_results': [],
            'classification_results': [],
            'window_info': []
        }
        
        for window_start in tqdm(window_starts, desc=f"Windows (size={window_size})", leave=False):
            window_end = window_start + window_size
            window_center = window_start + window_size / 2
            
            # Find samples in this window
            window_mask = (ages >= window_start) & (ages < window_end)
            window_indices = np.where(window_mask)[0]
            
            if len(window_indices) < min_samples:
                continue
            
            X_window = X[window_indices]
            
            window_info = {
                'window_start': window_start,
                'window_end': window_end,
                'window_center': window_center,
                'n_samples': len(window_indices),
                'age_range': [ages[window_indices].min(), ages[window_indices].max()],
                'age_mean': ages[window_indices].mean(),
                'age_std': ages[window_indices].std()
            }
            
            # Regression evaluation
            if y_regression is not None:
                y_reg_window = y_regression[window_indices]
                groups_window = groups[window_indices] if groups is not None else None
                
                reg_result = evaluate_window_regression(X_window, y_reg_window, groups_window)
                reg_result['window_center'] = window_center
                window_results['regression_results'].append(reg_result)
            
            # Classification evaluation
            if y_classification is not None:
                y_class_window = y_classification[window_indices]
                
                class_result = evaluate_window_classification(X_window, y_class_window)
                class_result['window_center'] = window_center
                window_results['classification_results'].append(class_result)
            
            window_results['window_info'].append(window_info)
        
        results[f'window_{window_size}'] = window_results
    
    # Create plots and save data
    create_sliding_window_plots(results, output_dir, target_label)
    save_sliding_window_data(results, output_dir)
    
    return results

def evaluate_window_regression(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Evaluate regression performance for a single window."""
    if len(X) < 5:  # Need minimum samples
        return {'rmse': np.nan, 'r2': np.nan, 'mae': np.nan, 'n_samples': len(X)}
    
    try:
        # Use simple train-test split for windows (faster than LOGO for many windows)
        if len(np.unique(y)) < 2:  # No variance in target
            return {'rmse': np.nan, 'r2': np.nan, 'mae': np.nan, 'n_samples': len(X)}
        
        # Use stratified split if possible
        test_size = min(0.3, max(0.1, 5.0 / len(X)))  # Adaptive test size
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Ridge regression
        model = Ridge(alpha=1.0).fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        return {
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'n_samples': len(X),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
    except Exception as e:
        return {'rmse': np.nan, 'r2': np.nan, 'mae': np.nan, 'n_samples': len(X), 'error': str(e)}

def evaluate_window_classification(X: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate classification performance for a single window."""
    if len(X) < 5 or len(np.unique(y)) < 2:  # Need minimum samples and classes
        return {'accuracy': np.nan, 'f1_macro': np.nan, 'n_samples': len(X), 'n_classes': len(np.unique(y))}
    
    try:
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Use stratified split
        test_size = min(0.3, max(0.1, 5.0 / len(X)))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train logistic regression
        model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'n_samples': len(X),
            'n_classes': len(np.unique(y)),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
    except Exception as e:
        return {'accuracy': np.nan, 'f1_macro': np.nan, 'n_samples': len(X), 
                'n_classes': len(np.unique(y)), 'error': str(e)}

def create_sliding_window_plots(results: dict, output_dir: str, target_label: str):
    """Create comprehensive sliding window plots."""
    os.makedirs(os.path.join(output_dir, 'sliding_window'), exist_ok=True)
    
    # 1. Main sliding window plot (like your attached image)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, (window_key, window_data) in enumerate(results.items()):
        color = colors[i]
        window_size = window_data['window_size']
        
        # Regression RMSE plot
        if window_data['regression_results']:
            reg_data = window_data['regression_results']
            centers = [r['window_center'] for r in reg_data]
            rmses = [r['rmse'] for r in reg_data if not np.isnan(r['rmse'])]
            valid_centers = [centers[j] for j, r in enumerate(reg_data) if not np.isnan(r['rmse'])]
            
            if valid_centers:
                axes[0, 0].plot(valid_centers, rmses, 'o-', color=color, 
                               label=f'Window {window_size}d', markersize=4, linewidth=1.5)
        
        # Regression R² plot
        if window_data['regression_results']:
            r2s = [r['r2'] for r in reg_data if not np.isnan(r['r2'])]
            valid_centers_r2 = [centers[j] for j, r in enumerate(reg_data) if not np.isnan(r['r2'])]
            
            if valid_centers_r2:
                axes[0, 1].plot(valid_centers_r2, r2s, 'o-', color=color, 
                               label=f'Window {window_size}d', markersize=4, linewidth=1.5)
        
        # Classification accuracy plot
        if window_data['classification_results']:
            class_data = window_data['classification_results']
            centers_class = [r['window_center'] for r in class_data]
            accuracies = [r['accuracy'] for r in class_data if not np.isnan(r['accuracy'])]
            valid_centers_acc = [centers_class[j] for j, r in enumerate(class_data) if not np.isnan(r['accuracy'])]
            
            if valid_centers_acc:
                axes[1, 0].plot(valid_centers_acc, accuracies, 'o-', color=color, 
                               label=f'Window {window_size}d', markersize=4, linewidth=1.5)
        
        # Sample count plot
        if window_data['window_info']:
            info_centers = [w['window_center'] for w in window_data['window_info']]
            sample_counts = [w['n_samples'] for w in window_data['window_info']]
            
            axes[1, 1].plot(info_centers, sample_counts, 'o-', color=color, 
                           label=f'Window {window_size}d', markersize=4, linewidth=1.5)
    
    # Customize plots
    axes[0, 0].set_title('RMSE vs Age Window', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Window Midpoint (days)')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('R² vs Age Window', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Window Midpoint (days)')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Classification Accuracy vs Age Window', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Window Midpoint (days)')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Sample Count vs Age Window', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Window Midpoint (days)')
    axes[1, 1].set_ylabel('Sample Count')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sliding_window', 'sliding_window_overview.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'sliding_window', 'sliding_window_overview.pdf'), bbox_inches='tight')
    plt.close()
    
    # 2. Individual detailed plots for each window size
    for window_key, window_data in results.items():
        create_individual_window_plots(window_data, output_dir, window_key)
    
    # 3. Comparative heatmaps
    create_sliding_window_heatmaps(results, output_dir)

def create_individual_window_plots(window_data: dict, output_dir: str, window_key: str):
    """Create detailed plots for individual window sizes."""
    window_size = window_data['window_size']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    if window_data['regression_results']:
        reg_data = window_data['regression_results']
        centers = [r['window_center'] for r in reg_data]
        
        # RMSE with error bars (using sample size as weight indicator)
        rmses = [r['rmse'] for r in reg_data]
        sample_sizes = [r['n_samples'] for r in reg_data]
        
        valid_mask = ~np.isnan(rmses)
        if np.any(valid_mask):
            scatter = axes[0, 0].scatter([centers[i] for i in range(len(centers)) if valid_mask[i]], 
                                       [rmses[i] for i in range(len(rmses)) if valid_mask[i]], 
                                       s=[sample_sizes[i] for i in range(len(sample_sizes)) if valid_mask[i]], 
                                       c=[sample_sizes[i] for i in range(len(sample_sizes)) if valid_mask[i]], 
                                       alpha=0.7, cmap='viridis')
            plt.colorbar(scatter, ax=axes[0, 0], label='Sample Size')
        
        axes[0, 0].set_title(f'RMSE vs Age (Window {window_size}d)')
        axes[0, 0].set_xlabel('Window Midpoint (days)')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].grid(True, alpha=0.3)
        
        # R² plot
        r2s = [r['r2'] for r in reg_data]
        valid_r2_mask = ~np.isnan(r2s)
        if np.any(valid_r2_mask):
            axes[0, 1].scatter([centers[i] for i in range(len(centers)) if valid_r2_mask[i]], 
                              [r2s[i] for i in range(len(r2s)) if valid_r2_mask[i]], 
                              s=[sample_sizes[i] for i in range(len(sample_sizes)) if valid_r2_mask[i]], 
                              alpha=0.7, color='orange')
        
        axes[0, 1].set_title(f'R² vs Age (Window {window_size}d)')
        axes[0, 1].set_xlabel('Window Midpoint (days)')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].grid(True, alpha=0.3)
        
        # MAE plot
        maes = [r['mae'] for r in reg_data]
        valid_mae_mask = ~np.isnan(maes)
        if np.any(valid_mae_mask):
            axes[0, 2].scatter([centers[i] for i in range(len(centers)) if valid_mae_mask[i]], 
                              [maes[i] for i in range(len(maes)) if valid_mae_mask[i]], 
                              s=[sample_sizes[i] for i in range(len(sample_sizes)) if valid_mae_mask[i]], 
                              alpha=0.7, color='red')
        
        axes[0, 2].set_title(f'MAE vs Age (Window {window_size}d)')
        axes[0, 2].set_xlabel('Window Midpoint (days)')
        axes[0, 2].set_ylabel('MAE')
        axes[0, 2].grid(True, alpha=0.3)
    
    if window_data['classification_results']:
        class_data = window_data['classification_results']
        centers_class = [r['window_center'] for r in class_data]
        
        # Accuracy plot
        accuracies = [r['accuracy'] for r in class_data]
        sample_sizes_class = [r['n_samples'] for r in class_data]
        
        valid_acc_mask = ~np.isnan(accuracies)
        if np.any(valid_acc_mask):
            axes[1, 0].scatter([centers_class[i] for i in range(len(centers_class)) if valid_acc_mask[i]], 
                              [accuracies[i] for i in range(len(accuracies)) if valid_acc_mask[i]], 
                              s=[sample_sizes_class[i] for i in range(len(sample_sizes_class)) if valid_acc_mask[i]], 
                              alpha=0.7, color='green')
        
        axes[1, 0].set_title(f'Classification Accuracy vs Age (Window {window_size}d)')
        axes[1, 0].set_xlabel('Window Midpoint (days)')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1 score plot
        f1_scores = [r['f1_macro'] for r in class_data]
        valid_f1_mask = ~np.isnan(f1_scores)
        if np.any(valid_f1_mask):
            axes[1, 1].scatter([centers_class[i] for i in range(len(centers_class)) if valid_f1_mask[i]], 
                              [f1_scores[i] for i in range(len(f1_scores)) if valid_f1_mask[i]], 
                              s=[sample_sizes_class[i] for i in range(len(sample_sizes_class)) if valid_f1_mask[i]], 
                              alpha=0.7, color='purple')
        
        axes[1, 1].set_title(f'F1 Macro vs Age (Window {window_size}d)')
        axes[1, 1].set_xlabel('Window Midpoint (days)')
        axes[1, 1].set_ylabel('F1 Macro')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Sample distribution
    if window_data['window_info']:
        info_data = window_data['window_info']
        info_centers = [w['window_center'] for w in info_data]
        sample_counts = [w['n_samples'] for w in info_data]
        
        axes[1, 2].bar(info_centers, sample_counts, width=window_data['window_step']*0.8, 
                       alpha=0.7, color='skyblue', edgecolor='navy', linewidth=0.5)
        axes[1, 2].set_title(f'Sample Distribution (Window {window_size}d)')
        axes[1, 2].set_xlabel('Window Midpoint (days)')
        axes[1, 2].set_ylabel('Sample Count')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sliding_window', f'{window_key}_detailed.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'sliding_window', f'{window_key}_detailed.pdf'), bbox_inches='tight')
    plt.close()

def create_sliding_window_heatmaps(results: dict, output_dir: str):
    """Create heatmap visualizations for sliding window results."""
    
    # Collect all window centers and sizes
    all_centers = set()
    window_sizes = []
    
    for window_key, window_data in results.items():
        window_sizes.append(window_data['window_size'])
        if window_data['regression_results']:
            all_centers.update([r['window_center'] for r in window_data['regression_results']])
        if window_data['classification_results']:
            all_centers.update([r['window_center'] for r in window_data['classification_results']])
    
    all_centers = sorted(list(all_centers))
    window_sizes = sorted(window_sizes)
    
    if not all_centers or not window_sizes:
        return
    
    # Create heatmap data
    rmse_matrix = np.full((len(window_sizes), len(all_centers)), np.nan)
    r2_matrix = np.full((len(window_sizes), len(all_centers)), np.nan)
    acc_matrix = np.full((len(window_sizes), len(all_centers)), np.nan)
    
    for i, window_size in enumerate(window_sizes):
        window_key = f'window_{window_size}'
        window_data = results[window_key]
        
        # Regression heatmaps
        if window_data['regression_results']:
            for result in window_data['regression_results']:
                center = result['window_center']
                if center in all_centers:
                    j = all_centers.index(center)
                    if not np.isnan(result['rmse']):
                        rmse_matrix[i, j] = result['rmse']
                    if not np.isnan(result['r2']):
                        r2_matrix[i, j] = result['r2']
        
        # Classification heatmaps
        if window_data['classification_results']:
            for result in window_data['classification_results']:
                center = result['window_center']
                if center in all_centers:
                    j = all_centers.index(center)
                    if not np.isnan(result['accuracy']):
                        acc_matrix[i, j] = result['accuracy']
    
    # Plot heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # RMSE heatmap
    im1 = axes[0].imshow(rmse_matrix, aspect='auto', cmap='viridis_r', interpolation='nearest')
    axes[0].set_title('RMSE Heatmap')
    axes[0].set_xlabel('Age Window Midpoint')
    axes[0].set_ylabel('Window Size (days)')
    axes[0].set_xticks(np.linspace(0, len(all_centers)-1, min(10, len(all_centers))))
    axes[0].set_xticklabels([f'{int(all_centers[int(i)])}' for i in np.linspace(0, len(all_centers)-1, min(10, len(all_centers)))])
    axes[0].set_yticks(range(len(window_sizes)))
    axes[0].set_yticklabels(window_sizes)
    plt.colorbar(im1, ax=axes[0], label='RMSE')
    
    # R² heatmap
    im2 = axes[1].imshow(r2_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[1].set_title('R² Heatmap')
    axes[1].set_xlabel('Age Window Midpoint')
    axes[1].set_ylabel('Window Size (days)')
    axes[1].set_xticks(np.linspace(0, len(all_centers)-1, min(10, len(all_centers))))
    axes[1].set_xticklabels([f'{int(all_centers[int(i)])}' for i in np.linspace(0, len(all_centers)-1, min(10, len(all_centers)))])
    axes[1].set_yticks(range(len(window_sizes)))
    axes[1].set_yticklabels(window_sizes)
    plt.colorbar(im2, ax=axes[1], label='R²')
    
    # Accuracy heatmap
    im3 = axes[2].imshow(acc_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[2].set_title('Classification Accuracy Heatmap')
    axes[2].set_xlabel('Age Window Midpoint')
    axes[2].set_ylabel('Window Size (days)')
    axes[2].set_xticks(np.linspace(0, len(all_centers)-1, min(10, len(all_centers))))
    axes[2].set_xticklabels([f'{int(all_centers[int(i)])}' for i in np.linspace(0, len(all_centers)-1, min(10, len(all_centers)))])
    axes[2].set_yticks(range(len(window_sizes)))
    axes[2].set_yticklabels(window_sizes)
    plt.colorbar(im3, ax=axes[2], label='Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sliding_window', 'performance_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'sliding_window', 'performance_heatmaps.pdf'), bbox_inches='tight')
    plt.close()

def save_sliding_window_data(results: dict, output_dir: str):
    """Save numerical data for all sliding window plots."""
    data_dir = os.path.join(output_dir, 'sliding_window', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    for window_key, window_data in results.items():
        window_size = window_data['window_size']
        
        # Save regression results
        if window_data['regression_results']:
            reg_df = pd.DataFrame(window_data['regression_results'])
            reg_df.to_csv(os.path.join(data_dir, f'{window_key}_regression_results.csv'), index=False)
            
            # Save plot-specific data
            valid_reg = reg_df.dropna(subset=['rmse'])
            if not valid_reg.empty:
                np.savetxt(os.path.join(data_dir, f'{window_key}_rmse_plot_data.txt'), 
                          np.column_stack([valid_reg['window_center'], valid_reg['rmse']]),
                          header='window_center rmse', fmt='%.6f')
            
            valid_r2 = reg_df.dropna(subset=['r2'])
            if not valid_r2.empty:
                np.savetxt(os.path.join(data_dir, f'{window_key}_r2_plot_data.txt'), 
                          np.column_stack([valid_r2['window_center'], valid_r2['r2']]),
                          header='window_center r2', fmt='%.6f')
        
        # Save classification results
        if window_data['classification_results']:
            class_df = pd.DataFrame(window_data['classification_results'])
            class_df.to_csv(os.path.join(data_dir, f'{window_key}_classification_results.csv'), index=False)
            
            # Save plot-specific data
            valid_acc = class_df.dropna(subset=['accuracy'])
            if not valid_acc.empty:
                np.savetxt(os.path.join(data_dir, f'{window_key}_accuracy_plot_data.txt'), 
                          np.column_stack([valid_acc['window_center'], valid_acc['accuracy']]),
                          header='window_center accuracy', fmt='%.6f')
        
        # Save window info
        if window_data['window_info']:
            info_df = pd.DataFrame(window_data['window_info'])
            info_df.to_csv(os.path.join(data_dir, f'{window_key}_window_info.csv'), index=False)
            
            # Save sample count plot data
            np.savetxt(os.path.join(data_dir, f'{window_key}_sample_count_plot_data.txt'), 
                      np.column_stack([info_df['window_center'], info_df['n_samples']]),
                      header='window_center n_samples', fmt='%.6f')
    
    # Save combined overview data
    overview_data = {
        'rmse': [],
        'r2': [],
        'accuracy': [],
        'window_center': [],
        'window_size': []
    }
    
    for window_key, window_data in results.items():
        window_size = window_data['window_size']
        
        if window_data['regression_results']:
            for result in window_data['regression_results']:
                overview_data['window_center'].append(result['window_center'])
                overview_data['window_size'].append(window_size)
                overview_data['rmse'].append(result.get('rmse', np.nan))
                overview_data['r2'].append(result.get('r2', np.nan))
                overview_data['accuracy'].append(np.nan)  # No classification data for this row
        
        if window_data['classification_results']:
            for result in window_data['classification_results']:
                # Check if we already have this window_center from regression
                existing_idx = None
                for i, (center, size) in enumerate(zip(overview_data['window_center'], overview_data['window_size'])):
                    if center == result['window_center'] and size == window_size:
                        existing_idx = i
                        break
                
                if existing_idx is not None:
                    overview_data['accuracy'][existing_idx] = result.get('accuracy', np.nan)
                else:
                    overview_data['window_center'].append(result['window_center'])
                    overview_data['window_size'].append(window_size)
                    overview_data['rmse'].append(np.nan)
                    overview_data['r2'].append(np.nan)
                    overview_data['accuracy'].append(result.get('accuracy', np.nan))
                    overview_df = pd.DataFrame(overview_data)
                    overview_df.to_csv(os.path.join(data_dir, 'sliding_window_overview_data.csv'), index=False)
                    
                    # Save data for the main overview plot (like your attached image)
                    for window_key, window_data in results.items():
                        window_size = window_data['window_size']
                        
                        # Regression data for overview plot
                        if window_data['regression_results']:
                            reg_data = window_data['regression_results']
                            centers = [r['window_center'] for r in reg_data if not np.isnan(r.get('rmse', np.nan))]
                            rmses = [r['rmse'] for r in reg_data if not np.isnan(r.get('rmse', np.nan))]
                            
                            if centers and rmses:
                                np.savetxt(os.path.join(data_dir, f'overview_rmse_window_{window_size}d.txt'), 
                                            np.column_stack([centers, rmses]),
                                            header=f'window_center rmse_window_{window_size}d', fmt='%.6f')
                        
                        # Classification data for overview plot
                        if window_data['classification_results']:
                            class_data = window_data['classification_results']
                            centers_class = [r['window_center'] for r in class_data if not np.isnan(r.get('accuracy', np.nan))]
                            accuracies = [r['accuracy'] for r in class_data if not np.isnan(r.get('accuracy', np.nan))]
                            
                            if centers_class and accuracies:
                                np.savetxt(os.path.join(data_dir, f'overview_accuracy_window_{window_size}d.txt'), 
                                            np.column_stack([centers_class, accuracies]),
                                            header=f'window_center accuracy_window_{window_size}d', fmt='%.6f')
                    
                    print(f"[INFO] Sliding window data saved to {data_dir}")

def main():
    """Main function to run the unified evaluation."""
    args = get_args()
    
    try:
        # Validate covariate arguments
        if args.use_covariate:
            if not all([args.summary_metadata, args.bodyweight_file, args.tissue_collection_file]):
                raise ValueError("When using covariates, you must provide --summary-metadata, "
                               "--bodyweight-file, and --tissue-collection-file")
            
            covariate_files = {
                'summary': args.summary_metadata,
                'bodyweight': args.bodyweight_file,
                'tissue': args.tissue_collection_file
            }
        else:
            covariate_files = None
        
        # Load embeddings first to check for frame mapping
        print(f"[INFO] Loading embeddings from {args.embeddings}...")
        loaded_obj = np.load(args.embeddings, allow_pickle=True)
        
        # Check if the loaded object is a 0-d array containing a dictionary
        if loaded_obj.shape == () and isinstance(loaded_obj.item(), dict):
            embeddings_data = loaded_obj.item()
            embeddings = embeddings_data['embeddings']
            frame_map = embeddings_data.get('frame_number_map', None)
        # Check if it's an NPZ file with an 'embeddings' key
        elif isinstance(loaded_obj, np.lib.npyio.NpzFile):
            embeddings = loaded_obj['embeddings']
            frame_map = loaded_obj.get('frame_number_map', None)
        # Otherwise, assume it's a standard numpy array
        else:
            embeddings = loaded_obj
            frame_map = None
        
        # Load data
        print("[INFO] Starting data loading...")
        X, y_regression, y_classification, groups, class_names = load_data(
            args.embeddings, args.labels, args.target_label, args.task, 
            args.group_label, args.classification_target, args.aggregation_method,
            args.use_covariate, covariate_files, args.covariate_strategy, args.cage_aggregation
        )
        
        os.makedirs(args.output_dir, exist_ok=True)
        all_results = {
            'evaluation_settings': {
                'aggregation_method': args.aggregation_method,
                'use_covariate': args.use_covariate,
                'covariate_strategy': args.covariate_strategy if args.use_covariate else None,
                'cage_aggregation': args.cage_aggregation if args.use_covariate else None,
                'embedding_shape': X.shape,
                'max_workers': args.max_workers,
                'sliding_window': args.sliding_window,
                'window_sizes': args.window_sizes if args.sliding_window else None,
                'window_step': args.window_step if args.sliding_window else None
            }
        }
        
        # Run regression evaluation
        if args.task in ["regression", "both"]:
            if groups is None:
                raise ValueError("Group labels are required for Leave-One-Group-Out regression.")
            print("\n" + "="*60)
            print("      LEAVE-ONE-GROUP-OUT REGRESSION EVALUATION")
            print("="*60)
            regression_results = evaluate_regression_logo(
                X, y_regression, groups, args.output_dir, 
                args.enable_multi_algorithm, args.max_workers
            )
            all_results['regression'] = regression_results
            
        # Run classification evaluation
        if args.task in ["classification", "both"]:
            if y_classification is None:
                raise ValueError("Classification target is required for classification evaluation.")
            print("\n" + "="*60)
            print("      CLASSIFICATION EVALUATION")
            print("="*60)
            classification_results = evaluate_classification(
                X, y_classification, args.test_size, class_names, 
                args.plot_confusion_matrix, args.output_dir, 
                args.enable_multi_algorithm, args.max_workers
            )
            all_results['classification'] = classification_results

        # Run sliding window evaluation if requested
        if args.sliding_window:
            if args.task in ["regression", "both"] or args.task in ["classification", "both"]:
                print("\n" + "="*60)
                print("      SLIDING WINDOW EVALUATION")
                print("="*60)
                
                # Load original metadata with age information for sliding window
                print("[INFO] Loading metadata for sliding window evaluation...")
                original_metadata = pd.read_csv(args.labels)
                
                # Process bodyweight data if using covariates
                if args.use_covariate and covariate_files:
                    print("[INFO] Processing bodyweight data for sliding window...")
                    bw_data = load_bodyweight_data(covariate_files['bodyweight'], covariate_files['tissue'])
                    summary_metadata = pd.read_csv(covariate_files['summary'])
                    original_metadata = process_metadata_with_bodyweight(
                        original_metadata, summary_metadata, bw_data, args.covariate_strategy, args.cage_aggregation
                    )
                
                # Align metadata with processed data
                print("[INFO] Aligning metadata with processed embeddings...")
                if frame_map is not None:
                    # For frame mapping case, we need to reconstruct the alignment
                    print("[INFO] Using frame mapping for metadata alignment...")
                    if 'from_tpt' in original_metadata.columns:
                        original_metadata["from_tpt"] = pd.to_datetime(original_metadata["from_tpt"])
                        original_metadata["key"] = original_metadata.apply(
                            lambda row: f"{row['cage_id']}_{row['from_tpt'].strftime('%Y-%m-%d')}", axis=1
                        )
                    else:
                        if 'key' not in original_metadata.columns:
                            original_metadata['key'] = original_metadata.index.astype(str)
                    
                    # Filter to only include rows that were successfully processed
                    valid_keys = []
                    valid_indices = []
                    for idx, row in original_metadata.iterrows():
                        key = getattr(row, 'key', str(idx))
                        if key in frame_map:
                            start, end = frame_map[key]
                            if end > start and end <= embeddings.shape[0]:
                                # Check if this sample would have valid embeddings
                                emb = embeddings[start:end]
                                if emb.shape[0] > 0 and not np.isnan(emb).any():
                                    # Apply aggregation to check validity
                                    if args.aggregation_method == "mean":
                                        flat = emb.mean(axis=0)
                                    elif args.aggregation_method == "median":
                                        flat = np.median(emb, axis=0)
                                    elif args.aggregation_method == "percentiles":
                                        flat = np.percentile(emb, [25, 50, 75], axis=0).flatten()
                                    else:
                                        flat = emb.mean(axis=0)
                                    
                                    if not np.isnan(flat).any():
                                        valid_keys.append(key)
                                        valid_indices.append(idx)
                    
                    # Filter metadata to valid samples
                    aligned_metadata = original_metadata.loc[valid_indices].copy()
                    
                    # Additional filtering for missing values in required columns
                    required_cols = [args.target_label]
                    if args.task in ['regression', 'both']:
                        required_cols.append(args.group_label)
                    if args.task in ['classification', 'both'] and args.classification_target:
                        if args.classification_target not in required_cols:
                            required_cols.append(args.classification_target)
                    if args.use_covariate:
                        required_cols.append('bodyweight')
                    
                    # Remove rows with missing values in required columns
                    work_df = aligned_metadata[required_cols].copy()
                    work_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    work_df.dropna(inplace=True)
                    aligned_metadata = aligned_metadata.loc[work_df.index]
                    
                else:
                    # Simple case - use the same filtering as in load_data
                    print("[INFO] Using simple alignment for metadata...")
                    n_samps = min(len(embeddings), len(original_metadata))
                    original_metadata = original_metadata.iloc[:n_samps].copy()
                    
                    # Define required columns based on the task
                    required_cols = [args.target_label]
                    if args.task in ['regression', 'both']:
                        required_cols.append(args.group_label)
                    if args.task in ['classification', 'both'] and args.classification_target:
                        if args.classification_target not in required_cols:
                            required_cols.append(args.classification_target)
                    if args.use_covariate:
                        required_cols.append('bodyweight')
                    
                    # Create a working copy and handle missing values
                    work_df = original_metadata[required_cols].copy()
                    work_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    work_df.dropna(inplace=True)
                    aligned_metadata = original_metadata.iloc[work_df.index]
                
                print(f"[INFO] Aligned metadata: {len(aligned_metadata)} samples for sliding window evaluation")
                
                if len(aligned_metadata) < args.min_samples_per_window:
                    print(f"[WARNING] Not enough samples ({len(aligned_metadata)}) for sliding window evaluation "
                          f"(minimum required: {args.min_samples_per_window})")
                else:
                    sliding_results = perform_sliding_window_evaluation(
                        X, y_regression, y_classification, groups, aligned_metadata,
                        args.window_sizes, args.window_step, args.min_samples_per_window,
                        args.output_dir, args.target_label, args.enable_multi_algorithm
                    )
                    all_results['sliding_window'] = sliding_results

        # Save comprehensive results
        results_path = os.path.join(args.output_dir, args.results_json)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=4, default=str)
        
        print(f"\n[INFO] Complete results saved to {results_path}")
        print(f"[INFO] Visualizations saved to {args.output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("      EVALUATION SUMMARY")
        print("="*60)
        
        if args.use_covariate:
            print(f"Bodyweight covariate: ENABLED (strategy: {args.covariate_strategy})")
        else:
            print("Bodyweight covariate: DISABLED")
        
        if 'regression' in all_results:
            print("\nRegression Results:")
            for model_name, results in all_results['regression'].items():
                print(f"  {model_name}:")
                print(f"    - RMSE: {results['mean_rmse']:.2f} (±{results['std_rmse']:.2f})")
                print(f"    - R²: {results['mean_r2']:.3f} (±{results['std_r2']:.3f})")
        
        if 'classification' in all_results:
            print("\nClassification Results:")
            for model_name, results in all_results['classification'].items():
                print(f"  {model_name}:")
                print(f"    - Accuracy: {results['accuracy']:.3f}")
                print(f"    - F1 Macro: {results['f1_macro']:.3f}")
        
        # Sliding window summary
        if args.sliding_window and 'sliding_window' in all_results:
            print(f"\nSliding Window Results:")
            print(f"  Window sizes evaluated: {args.window_sizes}")
            print(f"  Window step: {args.window_step} days")
            
            for window_key, window_data in all_results['sliding_window'].items():
                window_size = window_data['window_size']
                n_windows = len(window_data.get('window_info', []))
                print(f"  Window {window_size}d: {n_windows} windows evaluated")
                
                if window_data.get('regression_results'):
                    valid_rmse = [r['rmse'] for r in window_data['regression_results'] 
                                 if not np.isnan(r.get('rmse', np.nan))]
                    if valid_rmse:
                        print(f"    - RMSE range: {min(valid_rmse):.2f} - {max(valid_rmse):.2f} "
                              f"(avg: {np.mean(valid_rmse):.2f})")
                
                if window_data.get('classification_results'):
                    valid_acc = [r['accuracy'] for r in window_data['classification_results'] 
                                if not np.isnan(r.get('accuracy', np.nan))]
                    if valid_acc:
                        print(f"    - Accuracy range: {min(valid_acc):.3f} - {max(valid_acc):.3f} "
                              f"(avg: {np.mean(valid_acc):.3f})")
        
        print("\n[COMPLETE] Evaluation finished successfully!")

    except (ValueError, FileNotFoundError) as e:
        print(f"\n[ERROR] {e}")
    except KeyboardInterrupt:
        print(f"\n[INFO] Evaluation interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
if __name__ == "__main__":
    main()