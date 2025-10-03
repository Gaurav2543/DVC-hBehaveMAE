import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm
from threading import Lock

def analyze_single_sequence(args):
    """Analyze completeness of a single sequence"""
    row, data_dir = args
    
    cage_id = row["cage_id"]
    from_tpt = row["from_tpt"]
    to_tpt = row["to_tpt"]
    
    cage_file = os.path.join(data_dir, f"{cage_id}.csv")
    
    if not os.path.exists(cage_file):
        return {
            "cage_id": cage_id,
            "start": from_tpt,
            "expected": 0,
            "actual": 0,
            "completeness_%": 0,
            "missing": 0,
            "status": "file_not_found"
        }
    
    try:
        # Read only the timestamp column first to filter
        df = pd.read_csv(cage_file, usecols=["timestamp"], parse_dates=["timestamp"])
        df = df[(df["timestamp"] >= from_tpt) & (df["timestamp"] < to_tpt)]
        
        expected_frames = int((to_tpt - from_tpt).total_seconds() / 60)
        actual_frames = len(df)
        completeness = actual_frames / expected_frames * 100 if expected_frames > 0 else 0
        
        return {
            "cage_id": cage_id,
            "start": from_tpt,
            "expected": expected_frames,
            "actual": actual_frames,
            "completeness_%": completeness,
            "missing": expected_frames - actual_frames,
            "status": "success"
        }
    except Exception as e:
        return {
            "cage_id": cage_id,
            "start": from_tpt,
            "expected": 0,
            "actual": 0,
            "completeness_%": 0,
            "missing": 0,
            "status": f"error: {str(e)}"
        }

def analyze_data_completeness_parallel(summary_csv, data_dir, max_workers=None):
    """
    Multithreaded analysis of data completeness
    
    Args:
        summary_csv: Path to summary CSV
        data_dir: Path to cage data directory
        max_workers: Number of threads (default: CPU count)
    """
    
    if max_workers is None:
        max_workers = cpu_count()
    
    print(f"Using {max_workers} threads for parallel processing")
    
    # Load summary
    summary = pd.read_csv(summary_csv, low_memory=False)
    summary.columns = summary.columns.str.strip()
    summary["from_tpt"] = pd.to_datetime(summary["from_tpt"])
    summary["to_tpt"] = pd.to_datetime(summary["to_tpt"])
    
    total_sequences = len(summary)
    print(f"Analyzing {total_sequences} sequences...")
    
    # Prepare arguments for parallel processing
    args_list = [(row, data_dir) for _, row in summary.iterrows()]
    
    results = []
    
    # Process in parallel with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(analyze_single_sequence, args): args 
                   for args in args_list}
        
        # Collect results with progress bar
        with tqdm(total=total_sequences, desc="Analyzing sequences") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("DATA COMPLETENESS ANALYSIS")
    print(f"{'='*60}")
    print(f"Total sequences: {len(results_df)}")
    
    # Status breakdown
    status_counts = results_df['status'].value_counts()
    print(f"\nStatus breakdown:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    # Filter successful results for statistics
    success_df = results_df[results_df['status'] == 'success']
    
    if len(success_df) > 0:
        print(f"\nCompleteness statistics (successful reads only):")
        print(f"Perfect (100%): {(success_df['completeness_%'] == 100).sum()}")
        print(f"Near-complete (>95%): {(success_df['completeness_%'] > 95).sum()}")
        print(f"Incomplete (<95%): {(success_df['completeness_%'] <= 95).sum()}")
        
        print(f"\nCompleteness percentages:")
        print(success_df['completeness_%'].describe())
        
        print(f"\nBest 10 sequences:")
        print(success_df.nlargest(10, 'completeness_%')[
            ['cage_id', 'expected', 'actual', 'completeness_%']
        ])
        
        print(f"\nWorst 10 sequences:")
        print(success_df.nsmallest(10, 'completeness_%')[
            ['cage_id', 'expected', 'actual', 'completeness_%']
        ])
        
        # Save detailed results
        output_path = os.path.join(os.path.dirname(summary_csv), 
                                   'data_completeness_analysis.csv')
        results_df.to_csv(output_path, index=False)
        print(f"\nDetailed results saved to: {output_path}")
    else:
        print("\nNo successful reads - check data paths and file format")
    
    return results_df

# Run the analysis
if __name__ == "__main__":
    # 4-week analysis
    results_4w = analyze_data_completeness_parallel(
        summary_csv="/scratch/bhole/dvc_data/smoothed/40320/summary_metadata_40320.csv",
        data_dir="/scratch/bhole/dvc_data/smoothed/cage_activations_full_data_final"
    )
    
    # 3-day analysis
    results_3d = analyze_data_completeness_parallel(
        summary_csv="/scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv",
        data_dir="/scratch/bhole/dvc_data/smoothed/cage_activations_full_data_final"
    )