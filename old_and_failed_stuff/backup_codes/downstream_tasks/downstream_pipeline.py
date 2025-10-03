#!/usr/bin/env python3
"""
Unified downstream analysis pipeline with sliding window analysis
Automatically detects embedding levels and performs comprehensive analysis
"""

import os
import argparse
import numpy as np
import pandas as pd
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import itertools

# ML imports
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap

warnings.filterwarnings("ignore")

# =============================
# Configuration and Arguments
# =============================

def get_args():
    parser = argparse.ArgumentParser(description="Unified downstream analysis pipeline")
    
    # Input paths
    parser.add_argument("--metadata_path", required=True, help="Path to metadata CSV")
    parser.add_argument("--embeddings_dir", required=True, help="Directory containing embedding files")
    parser.add_argument("--output_dir", required=True, help="Base output directory")
    
    # Analysis parameters
    parser.add_argument("--aggregation_methods", nargs='+', default=["mean", "median", "percentiles"],
                       help="Aggregation methods to use")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    
    # Sliding window parameters
    parser.add_argument("--window_size", type=int, default=30, help="Age window size in days")
    parser.add_argument("--window_step", type=int, default=15, help="Age window step size in days")
    parser.add_argument("--min_samples_per_window", type=int, default=10, 
                       help="Minimum samples required per window")
    
    # Analysis toggles
    parser.add_argument("--skip_umaps", action="store_true", help="Skip UMAP generation")
    parser.add_argument("--skip_clustering", action="store_true", help="Skip strain clustering")
    
    return parser.parse_args()

# =============================
# Utility Functions
# =============================

def load_embeddings_and_metadata(path):
    """Load embeddings file and extract metadata - Fixed version"""
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.size == 1:
        data = data.item()
    
    # Handle both old and new embedding file formats
    if "embeddings" in data:
        embeddings = data["embeddings"]
    else:
        embeddings = data
    
    if "frame_number_map" in data:
        frame_map = data["frame_number_map"]
    elif "frame_map" in data:
        frame_map = data["frame_map"]  # Alternative key name
    else:
        raise KeyError("No frame map found in embedding file")
    
    # Extract metadata if available
    metadata = data.get("aggregation_info", {})
    
    print(f"[DEBUG] Loaded embeddings shape: {embeddings.shape}")
    print(f"[DEBUG] Frame map has {len(frame_map)} keys")
    print(f"[DEBUG] Sample frame map keys: {list(frame_map.keys())[:5]}")
    
    return embeddings, frame_map, metadata

def detect_embedding_levels(embeddings_dir):
    """Automatically detect available embedding levels from directory"""
    embedding_files = {}
    aggregation_names = set()
    
    for file in os.listdir(embeddings_dir):
        if file.endswith('.npy') and file.startswith('test_'):
            # Parse filename: test_{aggregation}_{level}.npy
            parts = file.replace('test_', '').replace('.npy', '').split('_')
            if len(parts) >= 2:
                level = parts[-1]  # last part is level
                aggregation = '_'.join(parts[:-1])  # everything before level
                
                if aggregation not in embedding_files:
                    embedding_files[aggregation] = {}
                embedding_files[aggregation][level] = os.path.join(embeddings_dir, file)
                aggregation_names.add(aggregation)
    
    print(f"[INFO] Detected aggregations: {sorted(aggregation_names)}")
    for agg in sorted(aggregation_names):
        levels = sorted(embedding_files[agg].keys())
        print(f"  {agg}: {levels}")
    
    return embedding_files, sorted(aggregation_names)

def create_age_windows(ages, window_size=30, step_size=15):
    """Create sliding age windows"""
    min_age, max_age = int(np.min(ages)), int(np.max(ages))
    windows = []
    
    start_age = min_age
    while start_age + window_size <= max_age:
        end_age = start_age + window_size
        windows.append((start_age, end_age))
        start_age += step_size
    
    print(f"[INFO] Created {len(windows)} age windows from {min_age} to {max_age} days")
    return windows

def filter_data_by_age_window(X, y_age, y_strain, cage_ids, age_window):
    """Filter data to specific age window"""
    start_age, end_age = age_window
    mask = (y_age >= start_age) & (y_age < end_age)
    
    return X[mask], y_age[mask], y_strain[mask], cage_ids[mask], np.sum(mask)

# =============================
# Data Processing
# =============================

def process_embeddings_for_aggregation(embeddings_path, metadata, method="mean"):
    """Process embeddings with specified aggregation method - Fixed version"""
    embeddings, frame_map, emb_metadata = load_embeddings_and_metadata(embeddings_path)
    
    X, y_age, y_strain, cage_ids, skipped = [], [], [], [], 0
    
    print(f"[DEBUG] Processing {len(metadata)} metadata rows")
    print(f"[DEBUG] Sample metadata keys: {metadata['key'].head().tolist()}")
    
    # Check for key mismatches
    metadata_keys = set(metadata['key'].tolist())
    frame_map_keys = set(frame_map.keys())
    
    intersection = metadata_keys.intersection(frame_map_keys)
    print(f"[DEBUG] Metadata keys: {len(metadata_keys)}")
    print(f"[DEBUG] Frame map keys: {len(frame_map_keys)}")
    print(f"[DEBUG] Matching keys: {len(intersection)}")
    
    if len(intersection) < 0.1 * len(metadata_keys):  # Less than 10% match
        print("[WARNING] Very few keys match between metadata and frame map!")
        print(f"[DEBUG] Sample metadata keys: {list(metadata_keys)[:5]}")
        print(f"[DEBUG] Sample frame map keys: {list(frame_map_keys)[:5]}")
    
    for row in metadata.itertuples():
        key = row.key
        if not key or pd.isna(key):
            skipped += 1
            continue
            
        if key not in frame_map:
            skipped += 1
            continue
            
        start, end = frame_map[key]
        if end <= start or end > embeddings.shape[0]:
            skipped += 1
            continue
            
        emb = embeddings[start:end]
        if emb.shape[0] == 0:
            skipped += 1
            continue
        
        # Check for NaN values more carefully
        if np.any(np.isnan(emb)):
            skipped += 1
            continue
        
        # Apply aggregation method
        try:
            if method == "mean":
                flat = emb.mean(axis=0)
            elif method == "median":
                flat = np.median(emb, axis=0)
            elif method == "percentiles":
                flat = np.percentile(emb, [25, 50, 75], axis=0).flatten()
            else:
                skipped += 1
                continue
                
            if np.any(np.isnan(flat)):
                skipped += 1
                continue
        except Exception as e:
            print(f"[WARNING] Aggregation failed for {key}: {e}")
            skipped += 1
            continue
        
        X.append(flat)
        y_age.append(row.avg_age_days_chunk_start)
        y_strain.append(str(row.strain).strip())
        cage_ids.append(row.cage_id)
    
    X = np.array(X)
    y_age = np.array(y_age)
    y_strain = np.array(y_strain)
    cage_ids = np.array(cage_ids)
    
    print(f"[INFO] Processed {len(X)} samples, skipped {skipped}")
    
    # If we're still skipping too many, let's investigate further
    if skipped > len(X):
        print(f"[WARNING] Skipped more samples ({skipped}) than processed ({len(X)})")
        print("[DEBUG] This suggests a data alignment issue. Please check:")
        print("  1. Key construction in metadata vs embedding extraction")
        print("  2. Frame map structure in embedding files")
        print("  3. Date formatting consistency")
    
    return X, y_age, y_strain, cage_ids

# =============================
# Analysis Functions
# =============================

def train_and_evaluate_regressors(X_train, X_test, y_train, y_test):
    """Train and evaluate regression models"""
    regressors = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR": SVR()
    }
    
    results = {}
    for name, model in regressors.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            if not np.isnan(preds).any():
                rmse = mean_squared_error(y_test, preds, squared=False)
                r2 = r2_score(y_test, preds)
                results[name] = {"rmse": rmse, "r2": r2, "model": model}
        except Exception as e:
            print(f"[WARNING] {name} failed: {e}")
            
    return results

def train_and_evaluate_classifiers(X_train, X_test, y_train, y_test, le_strain):
    """Train and evaluate classification models"""
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=5000),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVC": SVC()
    }
    
    results = {}
    for name, model in classifiers.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, preds)
            f1_macro = f1_score(y_test, preds, average="macro")
            
            # Per-strain accuracies
            strain_accuracies = {}
            for i, class_name in enumerate(le_strain.classes_):
                true_idx = y_test == i
                if np.sum(true_idx) > 0:
                    acc_i = (preds[true_idx] == i).mean()
                    strain_accuracies[class_name] = acc_i
            
            results[name] = {
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "strain_accuracies": strain_accuracies,
                "model": model
            }
        except Exception as e:
            print(f"[WARNING] {name} failed: {e}")
            
    return results

def analyze_single_configuration(X, y_age, y_strain, cage_ids, config_name, output_dir, 
                                test_size=0.2, random_state=42):
    """Analyze a single embedding configuration"""
    print(f"\n=== Analyzing {config_name} ===")
    
    # Encode strains
    le_strain = LabelEncoder()
    y_strain_encoded = le_strain.fit_transform(y_strain)
    
    # Split data
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_age, test_size=test_size, random_state=random_state)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_strain_encoded, test_size=test_size, random_state=random_state)
    
    # Regression analysis
    regression_results = train_and_evaluate_regressors(X_train_r, X_test_r, y_train_r, y_test_r)
    
    # Classification analysis
    classification_results = train_and_evaluate_classifiers(
        X_train_c, X_test_c, y_train_c, y_test_c, le_strain)
    
    # Per-strain RMSE analysis (using Random Forest)
    strain_rmse = {}
    strain_counts = {}
    if "RandomForestRegressor" in regression_results:
        rf_model = regression_results["RandomForestRegressor"]["model"]
        for strain in np.unique(y_test_c):
            strain_name = le_strain.inverse_transform([strain])[0]
            indices = np.where(y_test_c == strain)[0]
            if len(indices) > 0:
                preds_sub = rf_model.predict(X_test_r[indices])
                true_sub = y_test_r[indices]
                rmse = mean_squared_error(true_sub, preds_sub, squared=False)
                strain_rmse[strain_name] = rmse
                strain_counts[strain_name] = len(indices)
    
    # Save results
    config_output_dir = os.path.join(output_dir, config_name)
    os.makedirs(config_output_dir, exist_ok=True)
    
    # Compile results for JSON
    results_json = {}
    for strain in le_strain.classes_:
        results_json[strain] = {
            "rmse_age_prediction": strain_rmse.get(strain),
            "sample_count": strain_counts.get(strain, 0)
        }
        
        # Add classification accuracies
        for clf_name, clf_results in classification_results.items():
            acc_key = f"accuracy_classification_{clf_name}"
            results_json[strain][acc_key] = clf_results["strain_accuracies"].get(strain)
    
    # Save JSON results
    json_path = os.path.join(config_output_dir, f"results_{config_name}.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=4)
    
    return {
        "regression": regression_results,
        "classification": classification_results,
        "strain_rmse": strain_rmse,
        "strain_counts": strain_counts,
        "results_json": results_json,
        "le_strain": le_strain
    }

def sliding_window_analysis(X, y_age, y_strain, cage_ids, config_name, output_dir,
                           window_size=30, step_size=15, min_samples=10, 
                           test_size=0.2, random_state=42):
    """Perform sliding window analysis across age ranges"""
    print(f"\n=== Sliding Window Analysis for {config_name} ===")
    
    # Create age windows
    age_windows = create_age_windows(y_age, window_size, step_size)
    
    window_results = []
    window_details = []
    
    for i, (start_age, end_age) in enumerate(tqdm(age_windows, desc="Processing windows")):
        # Filter data for this window
        X_win, y_age_win, y_strain_win, cage_ids_win, n_samples = filter_data_by_age_window(
            X, y_age, y_strain, cage_ids, (start_age, end_age))
        
        if n_samples < min_samples:
            continue
        
        window_midpoint = (start_age + end_age) / 2
        
        try:
            # Analyze this window
            window_analysis = analyze_single_configuration(
                X_win, y_age_win, y_strain_win, cage_ids_win, 
                f"{config_name}_window_{i:03d}_{start_age}_{end_age}",
                os.path.join(output_dir, "windows"),
                test_size=test_size, random_state=random_state
            )
            
            # Extract key metrics
            window_result = {
                "window_id": i,
                "start_age": start_age,
                "end_age": end_age,
                "midpoint": window_midpoint,
                "n_samples": n_samples,
                "n_strains": len(np.unique(y_strain_win))
            }
            
            # Add regression metrics
            for reg_name, reg_result in window_analysis["regression"].items():
                window_result[f"{reg_name}_rmse"] = reg_result["rmse"]
                window_result[f"{reg_name}_r2"] = reg_result["r2"]
            
            # Add classification metrics
            for clf_name, clf_result in window_analysis["classification"].items():
                window_result[f"{clf_name}_accuracy"] = clf_result["accuracy"]
                window_result[f"{clf_name}_f1_macro"] = clf_result["f1_macro"]
            
            window_results.append(window_result)
            window_details.append({
                "window_info": window_result,
                "analysis": window_analysis
            })
            
        except Exception as e:
            print(f"[WARNING] Window {start_age}-{end_age} failed: {e}")
            continue
    
    if not window_results:
        print(f"[WARNING] No valid windows for {config_name}")
        return None
    
    # Convert to DataFrame
    window_df = pd.DataFrame(window_results)
    
    # Save window results
    window_output_dir = os.path.join(output_dir, "sliding_windows", config_name)
    os.makedirs(window_output_dir, exist_ok=True)
    
    window_df.to_csv(os.path.join(window_output_dir, "window_results.csv"), index=False)
    
    # Save detailed results
    with open(os.path.join(window_output_dir, "window_details.json"), "w") as f:
        json.dump(window_details, f, indent=4, default=str)
    
    return window_df, window_details

def plot_sliding_window_trends(window_results_dict, output_dir):
    """Create combined plots showing trends across windows for all configurations"""
    print("\n=== Creating sliding window trend plots ===")
    
    plots_dir = os.path.join(output_dir, "sliding_window_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Combine all window results
    all_data = []
    for config_name, (window_df, _) in window_results_dict.items():
        if window_df is not None:
            window_df_copy = window_df.copy()
            window_df_copy['config'] = config_name
            all_data.append(window_df_copy)
    
    if not all_data:
        print("[WARNING] No window results to plot")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Plot RMSE trends
    plt.figure(figsize=(15, 10))
    
    rmse_columns = [col for col in combined_df.columns if 'rmse' in col.lower()]
    n_rmse = len(rmse_columns)
    
    if n_rmse > 0:
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, rmse_col in enumerate(rmse_columns[:4]):  # Limit to first 4
            ax = axes[i]
            for config in combined_df['config'].unique():
                config_data = combined_df[combined_df['config'] == config]
                ax.plot(config_data['midpoint'], config_data[rmse_col], 
                       marker='o', label=config, linewidth=2, markersize=4)
            
            ax.set_xlabel('Window midpoint (days)')
            ax.set_ylabel('RMSE')
            ax.set_title(f'{rmse_col} vs Age Window')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(n_rmse, 4):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "rmse_trends_comparison.pdf"))
        plt.close()
    
    # Plot accuracy trends
    accuracy_columns = [col for col in combined_df.columns if 'accuracy' in col.lower()]
    n_acc = len(accuracy_columns)
    
    if n_acc > 0:
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, acc_col in enumerate(accuracy_columns[:4]):  # Limit to first 4
            ax = axes[i]
            for config in combined_df['config'].unique():
                config_data = combined_df[combined_df['config'] == config]
                ax.plot(config_data['midpoint'], config_data[acc_col], 
                       marker='s', label=config, linewidth=2, markersize=4)
            
            ax.set_xlabel('Window midpoint (days)')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{acc_col} vs Age Window')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(n_acc, 4):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "accuracy_trends_comparison.pdf"))
        plt.close()
    
    # Create summary statistics
    summary_stats = []
    for config in combined_df['config'].unique():
        config_data = combined_df[combined_df['config'] == config]
        
        config_summary = {"config": config}
        for col in combined_df.columns:
            if col in ['midpoint', 'n_samples', 'n_strains'] or col.endswith(('rmse', 'accuracy', 'f1_macro', 'r2')):
                if pd.api.types.is_numeric_dtype(config_data[col]):
                    config_summary[f"{col}_mean"] = config_data[col].mean()
                    config_summary[f"{col}_std"] = config_data[col].std()
        
        summary_stats.append(config_summary)
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(plots_dir, "window_summary_statistics.csv"), index=False)
    
    print(f"[SAVED] Sliding window plots and statistics to {plots_dir}")

# =============================
# UMAP Functions (Optional)
# =============================

def generate_umaps(X, y_age, y_strain, cage_ids, config_name, output_dir):
    """Generate UMAP visualizations"""
    print(f"\n=== Generating UMAPs for {config_name} ===")
    
    strains = sorted(set(y_strain))
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>']
    
    umap_dir = os.path.join(output_dir, "umaps")
    os.makedirs(umap_dir, exist_ok=True)
    
    pdf_path = os.path.join(umap_dir, f"umap_{config_name}.pdf")
    
    with PdfPages(pdf_path) as pdf:
        for strain in tqdm(strains, desc=f"UMAP for {config_name}"):
            mask = y_strain == strain
            if np.sum(mask) < 5:
                continue
            
            X_s = X[mask]
            y_age_s = y_age[mask]
            cage_s = cage_ids[mask]
            
            # UMAP transformation
            reducer = umap.UMAP(random_state=42)
            X_scaled = StandardScaler().fit_transform(X_s)
            X_umap = reducer.fit_transform(X_scaled)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            unique_cages = sorted(set(cage_s))
            marker_map = {c: m for c, m in zip(unique_cages, itertools.cycle(markers))}
            
            for cage in unique_cages:
                cage_mask = cage_s == cage
                ax.scatter(X_umap[cage_mask, 0], X_umap[cage_mask, 1], 
                          c=y_age_s[cage_mask], cmap="viridis", s=20, alpha=0.7,
                          marker=marker_map[cage], label=f"Cage {cage}")
            
            sm = plt.cm.ScalarMappable(cmap="viridis")
            sm.set_array(y_age_s)
            fig.colorbar(sm, ax=ax, label="Average Age (days)")
            
            ax.set_title(f"{config_name} - Strain: {strain}")
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
            ax.legend(fontsize=8, loc='best')
            
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    
    print(f"[SAVED] UMAP plots to {pdf_path}")

# =============================
# Main Pipeline
# =============================

def main():
    args = get_args()
    
    print(f"[START] Unified downstream analysis pipeline")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print(f"[INFO] Loading metadata from {args.metadata_path}")
    metadata = pd.read_csv(args.metadata_path, low_memory=False)
    metadata["from_tpt"] = pd.to_datetime(metadata["from_tpt"])
    metadata["key"] = metadata.apply(
        lambda row: f"{row['cage_id']}_{row['from_tpt'].strftime('%Y-%m-%d')}", axis=1)
    
    # Auto-detect embedding levels
    print(f"[INFO] Auto-detecting embedding levels from {args.embeddings_dir}")
    embedding_files, aggregation_names = detect_embedding_levels(args.embeddings_dir)
    
    if not embedding_files:
        raise ValueError(f"No embedding files found in {args.embeddings_dir}")
    
    # Store all results
    all_results = {}
    window_results = {}
    
    # Process each configuration
    for aggregation in aggregation_names:
        for level in sorted(embedding_files[aggregation].keys()):
            for method in args.aggregation_methods:
                config_name = f"{aggregation}_{level}_{method}"
                embedding_path = embedding_files[aggregation][level]
                
                print(f"\n{'='*60}")
                print(f"Processing: {config_name}")
                print(f"Embedding file: {embedding_path}")
                
                try:
                    # Process embeddings
                    X, y_age, y_strain, cage_ids = process_embeddings_for_aggregation(
                        embedding_path, metadata, method)
                    
                    if len(X) == 0:
                        print(f"[SKIP] No valid data for {config_name}")
                        continue
                    
                    # Full dataset analysis
                    results = analyze_single_configuration(
                        X, y_age, y_strain, cage_ids, config_name, 
                        args.output_dir, args.test_size, args.random_state)
                    
                    all_results[config_name] = results
                    
                    # Sliding window analysis
                    window_result = sliding_window_analysis(
                        X, y_age, y_strain, cage_ids, config_name, args.output_dir,
                        args.window_size, args.window_step, args.min_samples_per_window,
                        args.test_size, args.random_state)
                    
                    window_results[config_name] = window_result
                    
                    # UMAP generation (optional)
                    if not args.skip_umaps:
                        generate_umaps(X, y_age, y_strain, cage_ids, config_name, args.output_dir)
                    
                except Exception as e:
                    print(f"[ERROR] Failed processing {config_name}: {e}")
                    continue
    
    # Generate comparison plots and summaries
    print(f"\n{'='*60}")
    print("Generating comparison analyses...")
    
    # Create sliding window trend plots
    plot_sliding_window_trends(window_results, args.output_dir)
    
    # Create overall comparison
    comparison_dir = os.path.join(args.output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Compile RMSE matrix
    rmse_matrix = {}
    for config_name, results in all_results.items():
        for strain, metrics in results["results_json"].items():
            rmse = metrics.get("rmse_age_prediction")
            if strain not in rmse_matrix:
                rmse_matrix[strain] = {}
            rmse_matrix[strain][config_name] = rmse
    
    # Save RMSE matrix
    rmse_df = pd.DataFrame(rmse_matrix).T
    rmse_df.to_csv(os.path.join(comparison_dir, "rmse_matrix.csv"))
    
    # Find best configurations
    best_configs = rmse_df.idxmin(axis=1)
    best_rmses = rmse_df.min(axis=1)
    
    summary_df = pd.DataFrame({
        "best_config": best_configs,
        "best_rmse": best_rmses
    }).sort_values("best_rmse")
    
    summary_df.to_csv(os.path.join(comparison_dir, "best_configurations.csv"))
    
    # Create heatmap
    plt.figure(figsize=(20, 12))
    sns.heatmap(rmse_df, annot=True, fmt=".1f", cmap="coolwarm", 
                linewidths=0.5, cbar_kws={"label": "RMSE"})
    plt.title("RMSE Heatmap by Configuration and Strain")
    plt.xlabel("Configuration")
    plt.ylabel("Strain")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "rmse_heatmap.pdf"))
    plt.close()
    
    # Save pipeline metadata
    pipeline_metadata = {
        "args": vars(args),
        "aggregation_names": aggregation_names,
        "embedding_levels": {agg: list(embedding_files[agg].keys()) for agg in aggregation_names},
        "total_configurations": len(all_results),
        "successful_configurations": len([r for r in all_results.values() if r is not None]),
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(args.output_dir, "pipeline_metadata.json"), "w") as f:
        json.dump(pipeline_metadata, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"✓ Pipeline completed successfully!")
    print(f"Processed {len(all_results)} configurations")
    print(f"Results saved to: {args.output_dir}")
    print(f"Key outputs:")
    print(f"  - Individual results: {args.output_dir}/*/")
    print(f"  - Sliding window analysis: {args.output_dir}/sliding_windows/")
    print(f"  - Sliding window plots: {args.output_dir}/sliding_window_plots/")
    print(f"  - Comparison results: {args.output_dir}/comparison/")
    if not args.skip_umaps:
        print(f"  - UMAP visualizations: {args.output_dir}/umaps/")

if __name__ == "__main__":
    main()