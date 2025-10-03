#!/usr/bin/env python3
# unified_evaluate_embeddings.py

import argparse
import warnings
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (mean_absolute_error, r2_score, classification_report, 
                           ConfusionMatrixDisplay, root_mean_squared_error, mean_squared_error,
                           accuracy_score, f1_score)
import matplotlib.pyplot as plt

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
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Proportion of the dataset for the test split (default: 0.2 for a 80-20 split).")
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
    return p.parse_args()

def load_data(emb_path: str, labels_path: str, target_label: str, task: str, 
              group_label: str, classification_target: str = None, 
              aggregation_method: str = "mean") -> tuple:
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
    
    # Handle frame mapping if available
    if frame_map is not None:
        print(f"[INFO] Using frame mapping for aggregation method: {aggregation_method}")
        return load_data_with_frame_mapping(embeddings, frame_map, labels_df, target_label, 
                                          task, group_label, classification_target, aggregation_method)
    
    # Define required columns based on the task
    required_cols = [target_label]
    if task in ['regression', 'both']:
        required_cols.append(group_label)
    if task in ['classification', 'both'] and classification_target:
        if classification_target not in required_cols:
            required_cols.append(classification_target)

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
                               group_label, classification_target, aggregation_method):
    """Handle data loading when frame mapping is available."""
    # Prepare metadata
    if 'from_tpt' in metadata.columns:
        metadata["from_tpt"] = pd.to_datetime(metadata["from_tpt"])
        metadata["key"] = metadata.apply(lambda row: f"{row['cage_id']}_{row['from_tpt'].strftime('%Y-%m-%d')}", axis=1)
    else:
        # Assume there's already a key column or create one
        if 'key' not in metadata.columns:
            metadata['key'] = metadata.index.astype(str)
    
    X, y_reg, y_class, groups_list, skipped = [], [], [], [], 0
    
    for row in metadata.itertuples():
        key = getattr(row, 'key', str(row.Index))
        if not key or key not in frame_map:
            skipped += 1
            continue
            
        start, end = frame_map[key]
        if end <= start or end > embeddings.shape[0]:
            skipped += 1
            continue
            
        emb = embeddings[start:end]
        if emb.shape[0] == 0 or np.isnan(emb).any():
            skipped += 1
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
            continue

        X.append(flat)
        if task in ['regression', 'both']:
            y_reg.append(getattr(row, target_label))
            groups_list.append(getattr(row, group_label))
        
        if task in ['classification', 'both']:
            class_target = classification_target if classification_target else group_label
            y_class.append(getattr(row, class_target))

    print(f"[INFO] Processed {len(X)} samples, skipped {skipped} due to missing data or frame mapping issues.")
    
    X = np.array(X)
    y_regression = np.array(y_reg) if y_reg else None
    y_classification = np.array(y_class) if y_class else None
    groups = np.array(groups_list) if groups_list else None
    class_names = sorted(list(np.unique(y_classification))) if y_classification is not None else None
    
    return X, y_regression, y_classification, groups, class_names

def evaluate_regression_logo(X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
                           output_dir: str, enable_multi: bool = False):
    """Performs Leave-One-Group-Out regression evaluation with multiple algorithms."""
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(groups=groups)
    unique_groups = np.unique(groups)
    print(f"\n[INFO] Starting Leave-One-Group-Out validation across {n_splits} groups...")

    # Define models
    models = {
        "Ridge": Ridge(alpha=1.0)
    }
    if enable_multi:
        models.update({
            "LinearRegression": LinearRegression(n_jobs=-1),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "SVR": SVR()
        })

    # Store results for each model
    all_results = {}
    
    for model_name, model_class in models.items():
        print(f"\n--- Evaluating {model_name} ---")
        maes, r2s, baseline_maes, rmses = [], [], [], []
        group_results = {}

        for i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            held_out_group = unique_groups[i]
            print(f"  -> Fold {i+1}/{n_splits}: Holding out group '{held_out_group}'...")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = model_class.fit(X_train_scaled, y_train) if hasattr(model_class, 'fit') else model_class.__class__().fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Calculate metrics for this fold
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            
            maes.append(mae)
            r2s.append(r2)
            rmses.append(rmse)
            
            baseline_pred = np.full_like(y_test, fill_value=y_train.mean())
            baseline_maes.append(mean_absolute_error(y_test, baseline_pred))
            
            # Store per-group results
            group_results[held_out_group] = {
                'mae': mae,
                'r2': r2,
                'rmse': rmse,
                'sample_count': len(y_test)
            }

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

def evaluate_classification(X: np.ndarray, y: np.ndarray, test_size: float, class_names: list, 
                          plot_cm: bool, output_dir: str, enable_multi: bool = False):
    """Performs classification evaluation using a train-test split with multiple algorithms."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    }
    if enable_multi:
        models.update({
            "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42, njobs=-1),
            "SVC": SVC(random_state=42)
        })

    all_results = {}
    os.makedirs(output_dir, exist_ok=True)

    for model_name, model in models.items():
        print(f"\n--- Training {model_name} ---")
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

def main():
    """Main function to run the unified evaluation."""
    args = get_args()
    
    try:
        # Load data
        X, y_regression, y_classification, groups, class_names = load_data(
            args.embeddings, args.labels, args.target_label, args.task, 
            args.group_label, args.classification_target, args.aggregation_method
        )
        
        os.makedirs(args.output_dir, exist_ok=True)
        all_results = {}
        
        # Run regression evaluation
        if args.task in ["regression", "both"]:
            if groups is None:
                raise ValueError("Group labels are required for Leave-One-Group-Out regression.")
            print("\n" + "="*60)
            print("      LEAVE-ONE-GROUP-OUT REGRESSION EVALUATION")
            print("="*60)
            regression_results = evaluate_regression_logo(
                X, y_regression, groups, args.output_dir, args.enable_multi_algorithm
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
                args.plot_confusion_matrix, args.output_dir, args.enable_multi_algorithm
            )
            all_results['classification'] = classification_results

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
        
        if 'regression' in all_results:
            for model_name, results in all_results['regression'].items():
                print(f"Regression ({model_name}):")
                print(f"  - RMSE: {results['mean_rmse']:.2f} (±{results['std_rmse']:.2f})")
                print(f"  - R²: {results['mean_r2']:.3f} (±{results['std_r2']:.3f})")
        
        if 'classification' in all_results:
            for model_name, results in all_results['classification'].items():
                print(f"Classification ({model_name}):")
                print(f"  - Accuracy: {results['accuracy']:.3f}")
                print(f"  - F1 Macro: {results['f1_macro']:.3f}")

    except (ValueError, FileNotFoundError) as e:
        print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    main()