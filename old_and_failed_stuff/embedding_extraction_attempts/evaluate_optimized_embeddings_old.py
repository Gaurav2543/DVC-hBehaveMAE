"""
Comprehensive Embedding Evaluator and Analyzer
Detailed analysis of extracted hBehaveMAE features with temporal and spatial mapping
"""
import argparse
import os
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zarr
import warnings
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_args():
    p = argparse.ArgumentParser(description="Comprehensive Embedding Analysis")
    p.add_argument("--embedding_dir", required=True, type=str, help="Directory containing extracted embeddings")
    p.add_argument("--summary_csv", required=True, type=str, help="Original summary CSV for timestamp mapping")
    p.add_argument("--dvc_root", required=True, type=str, help="DVC root for original data access")
    p.add_argument("--output_dir", required=True, type=str, help="Output directory for analysis results")
    p.add_argument("--aggregation_name", type=str, default="3days", help="Which aggregation to analyze")
    p.add_argument("--detailed_analysis", action="store_true", help="Run detailed statistical analysis")
    p.add_argument("--create_visualizations", action="store_true", help="Create comprehensive visualizations")
    p.add_argument("--temporal_analysis", action="store_true", help="Analyze temporal patterns")
    p.add_argument("--spatial_analysis", action="store_true", help="Analyze spatial patterns")
    p.add_argument("--comparative_analysis", action="store_true", help="Compare different feature types")
    
    return p.parse_args()

class ComprehensiveEmbeddingAnalyzer:
    """Comprehensive analyzer for extracted embeddings"""
    
    def __init__(self, embedding_dir: Path, summary_csv: str, dvc_root: str, output_dir: Path):
        self.embedding_dir = Path(embedding_dir)
        self.summary_csv = summary_csv
        self.dvc_root = dvc_root
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_structure = {}
        self.frame_maps = {}
        self.temporal_metadata = {}
        self.feature_statistics = {}
        
    def discover_embedding_structure(self, aggregation_name: str):
        """Discover and catalog the complete structure of extracted embeddings"""
        print(f"🔍 Discovering embedding structure for {aggregation_name}...")
        
        agg_dir = self.embedding_dir / aggregation_name
        if not agg_dir.exists():
            raise FileNotFoundError(f"Aggregation directory not found: {agg_dir}")
        
        structure = {
            'aggregation_name': aggregation_name,
            'base_directory': str(agg_dir),
            'categories': {},
            'total_files': 0,
            'total_sequences': 0,
            'discovery_timestamp': datetime.datetime.now().isoformat()
        }
        
        # Load frame mapping
        frame_map_path = agg_dir / 'frame_map.npy'
        if frame_map_path.exists():
            self.frame_maps[aggregation_name] = np.load(frame_map_path, allow_pickle=True).item()
            structure['total_sequences'] = len(self.frame_maps[aggregation_name])
            print(f"   📊 Found frame mapping for {structure['total_sequences']} sequences")
        
        # Discover categories
        # Discover categories
        for category_dir in agg_dir.iterdir():
            # Add this check to skip the frame_map file and other non-directories
            if not category_dir.is_dir() or category_dir.name == 'frame_map.zarr':
                continue

            if category_dir.is_dir():
                category_name = category_dir.name
                structure['categories'][category_name] = self._analyze_category(category_dir)
                structure['total_files'] += structure['categories'][category_name]['file_count']
        
        self.embedding_structure[aggregation_name] = structure
        
        # Save structure analysis
        structure_path = self.output_dir / f'{aggregation_name}_structure_analysis.npy'
        np.save(structure_path, structure)
        
        self._create_structure_report(structure, aggregation_name)
        return structure
    
    def _analyze_category(self, category_dir: Path):
        """Analyze a specific feature category"""
        category_info = {
            'category_name': category_dir.name,
            'path': str(category_dir),
            'sequences': {},
            'file_count': 0,
            'total_features': 0,
            'feature_types': set()
        }
        
        # Check for Zarr files (sequence-based storage)
        zarr_files = list(category_dir.glob("*.zarr"))
        if zarr_files:
            for zarr_file in zarr_files:
                seq_name = zarr_file.stem
                zarr_store = zarr.open(str(zarr_file), mode='r')
                
                seq_info = {
                    'sequence_name': seq_name,
                    'storage_type': 'zarr',
                    'features': {}
                }
                
                for feature_name in zarr_store.keys():
                    feature_array = zarr_store[feature_name]
                    seq_info['features'][feature_name] = {
                        'shape': feature_array.shape,
                        'dtype': str(feature_array.dtype),
                        'chunks': getattr(feature_array, 'chunks', None),
                        'size_mb': (np.prod(feature_array.shape) * feature_array.dtype.itemsize) / (1024**2)
                    }
                    category_info['feature_types'].add(feature_name)
                    category_info['total_features'] += 1
                
                category_info['sequences'][seq_name] = seq_info
                category_info['file_count'] += 1
        
        # Check for numpy files (individual feature files)
        npy_files = list(category_dir.glob("*.npy"))
        if npy_files:
            for npy_file in npy_files:
                feature_name = npy_file.stem
                try:
                    # Load array to get metadata (memory-efficient way)
                    with open(npy_file, 'rb') as f:
                        version = np.lib.format.read_magic(f)
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f) if version == (1, 0) else np.lib.format.read_array_header_2_0(f)
                    
                    feature_info = {
                        'feature_name': feature_name,
                        'storage_type': 'numpy',
                        'shape': shape,
                        'dtype': str(dtype),
                        'size_mb': (np.prod(shape) * dtype.itemsize) / (1024**2)
                    }
                    
                    category_info['sequences'][feature_name] = feature_info
                    category_info['feature_types'].add(feature_name)
                    category_info['total_features'] += 1
                    category_info['file_count'] += 1
                    
                except Exception as e:
                    print(f"   ⚠️  Could not analyze {npy_file}: {e}")
        
        category_info['feature_types'] = list(category_info['feature_types'])
        return category_info
    
    def _create_structure_report(self, structure: Dict, aggregation_name: str):
        """Create human-readable structure report"""
        report_path = self.output_dir / f'{aggregation_name}_structure_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("🔬 COMPREHENSIVE EMBEDDING STRUCTURE ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Aggregation: {structure['aggregation_name']}\n")
            f.write(f"Analysis Date: {structure['discovery_timestamp']}\n")
            f.write(f"Base Directory: {structure['base_directory']}\n")
            f.write(f"Total Sequences: {structure['total_sequences']}\n")
            f.write(f"Total Files: {structure['total_files']}\n\n")
            
            # Category breakdown
            f.write("📁 CATEGORY BREAKDOWN\n")
            f.write("-" * 30 + "\n")
            
            total_size_mb = 0
            for cat_name, cat_info in structure['categories'].items():
                f.write(f"\n🗂️  {cat_name.upper()}\n")
                f.write(f"   Files: {cat_info['file_count']}\n")
                f.write(f"   Total Features: {cat_info['total_features']}\n")
                f.write(f"   Feature Types: {len(cat_info['feature_types'])}\n")
                
                # Calculate category size
                cat_size = 0
                for seq_name, seq_info in cat_info['sequences'].items():
                    if isinstance(seq_info, dict):
                        if 'features' in seq_info:  # Zarr format
                            cat_size += sum(feat['size_mb'] for feat in seq_info['features'].values())
                        elif 'size_mb' in seq_info:  # Numpy format
                            cat_size += seq_info['size_mb']
                
                total_size_mb += cat_size
                f.write(f"   Size: {cat_size:.2f} MB\n")
                
                # Feature type details
                f.write("   📊 Feature Types:\n")
                for feat_type in cat_info['feature_types']:
                    f.write(f"      - {feat_type}\n")
            
            f.write(f"\n💾 TOTAL STORAGE: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)\n")
            
            # Frame mapping info
            if aggregation_name in self.frame_maps:
                f.write(f"\n🕒 TEMPORAL MAPPING\n")
                f.write("-" * 20 + "\n")
                frame_map = self.frame_maps[aggregation_name]
                f.write(f"Sequences with frame mapping: {len(frame_map)}\n")
                
                total_frames = sum(end - start for start, end in frame_map.values())
                f.write(f"Total frames: {total_frames:,}\n")
                
                # Sample frame ranges
                f.write("\n📋 Sample Frame Ranges:\n")
                for i, (seq_key, (start, end)) in enumerate(list(frame_map.items())[:5]):
                    f.write(f"   {seq_key}: frames {start:,} to {end:,} ({end-start:,} frames)\n")
                if len(frame_map) > 5:
                    f.write(f"   ... and {len(frame_map)-5} more sequences\n")
        
        print(f"   📝 Structure report saved: {report_path}")
    
    def load_and_analyze_features(self, aggregation_name: str, max_sequences: int = None):
        """Load and perform statistical analysis of features"""
        print(f"📊 Loading and analyzing features for {aggregation_name}...")
        
        if aggregation_name not in self.embedding_structure:
            self.discover_embedding_structure(aggregation_name)
        
        structure = self.embedding_structure[aggregation_name]
        analysis_results = {
            'aggregation_name': aggregation_name,
            'categories': {},
            'cross_category_analysis': {},
            'temporal_analysis': {},
            'spatial_analysis': {}
        }
        
        # Process each category
        for cat_name, cat_info in structure['categories'].items():
            print(f"   🔍 Analyzing category: {cat_name}")
            cat_analysis = self._analyze_category_features(cat_name, cat_info, max_sequences)
            analysis_results['categories'][cat_name] = cat_analysis
        
        self.feature_statistics[aggregation_name] = analysis_results
        
        # Save analysis results
        analysis_path = self.output_dir / f'{aggregation_name}_feature_analysis.npy'
        np.save(analysis_path, analysis_results)
        
        return analysis_results
    
    def _analyze_category_features(self, cat_name: str, cat_info: Dict, max_sequences: int = None):
        """Detailed analysis of features in a category"""
        cat_analysis = {
            'category_name': cat_name,
            'feature_statistics': {},
            'dimensionality_analysis': {},
            'correlation_analysis': {},
            'sample_data': {}
        }
        
        # Determine sequences to analyze
        sequences_to_analyze = list(cat_info['sequences'].keys())
        if max_sequences:
            sequences_to_analyze = sequences_to_analyze[:max_sequences]
        
        feature_data_collection = defaultdict(list)
        
        # Load and analyze each sequence
        for seq_name in sequences_to_analyze[:3]:  # Limit for memory efficiency
            seq_info = cat_info['sequences'][seq_name]
            
            try:
                if seq_info.get('storage_type') == 'zarr':
                    # Load from Zarr
                    zarr_path = Path(cat_info['path']) / f"{seq_name}.zarr"
                    zarr_store = zarr.open(str(zarr_path), mode='r')
                    
                    for feature_name in zarr_store.keys():
                        feature_array = zarr_store[feature_name][:]
                        self._analyze_single_feature(feature_name, feature_array, cat_analysis)
                        feature_data_collection[feature_name].append(feature_array)
                        
                elif seq_info.get('storage_type') == 'numpy':
                    # Load from numpy
                    npy_path = Path(cat_info['path']) / f"{seq_name}.npy"
                    feature_array = np.load(npy_path)
                    self._analyze_single_feature(seq_name, feature_array, cat_analysis)
                    feature_data_collection[seq_name].append(feature_array)
                    
            except Exception as e:
                print(f"      ⚠️  Error loading {seq_name}: {e}")
                continue
        
        # Cross-feature correlation analysis
        if len(feature_data_collection) > 1:
            cat_analysis['correlation_analysis'] = self._compute_feature_correlations(feature_data_collection)
        
        return cat_analysis
    
    def _analyze_single_feature(self, feature_name: str, feature_array: np.ndarray, cat_analysis: Dict):
        """Analyze a single feature array, robustly handling non-finite values."""
        if feature_array.size == 0:
            return

        # --- Definitive Fix Start ---
        # Calculate non-finite count on the original array first
        non_finite_count = int(np.sum(~np.isfinite(feature_array)))
        
        # Create a clean array for statistical calculations
        finite_array = feature_array[np.isfinite(feature_array)]

        # If, after filtering, the array is empty, all values were non-finite.
        if finite_array.size == 0:
            stats = {
                'shape': feature_array.shape, 'dtype': str(feature_array.dtype),
                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'zeros_percentage': 0.0, 'non_finite_count': non_finite_count
            }
            cat_analysis['feature_statistics'][feature_name] = stats
            return
        # --- Definitive Fix End ---

        # Basic statistics are now calculated ONLY on the clean, finite array
        stats = {
            'shape': feature_array.shape,
            'dtype': str(feature_array.dtype),
            'mean': float(np.mean(finite_array)),
            'std': float(np.std(finite_array)),
            'min': float(np.min(finite_array)),
            'max': float(np.max(finite_array)),
            'zeros_percentage': float(np.mean(finite_array == 0) * 100),
            'non_finite_count': non_finite_count
        }

        # Dimensionality analysis
        if feature_array.ndim == 2:
            feature_variances = np.nanvar(feature_array.astype(np.float32), axis=0)
            valid_variances = feature_variances[np.isfinite(feature_variances)]
            stats['feature_variances'] = {
                'mean_variance': float(np.mean(valid_variances)) if valid_variances.size > 0 else 0.0,
                'variance_std': float(np.std(valid_variances)) if valid_variances.size > 0 else 0.0,
                'low_variance_features': int(np.sum(valid_variances < 1e-5))
            }

        cat_analysis['feature_statistics'][feature_name] = stats

        # Store sample data for visualization
        sample_size = min(1000, feature_array.shape[0])
        if sample_size > 0:
            cat_analysis['sample_data'][feature_name] = feature_array[:sample_size]
                
    def _compute_feature_correlations(self, feature_data: Dict):
        """Compute correlations between different features"""
        correlation_results = {}
        
        feature_names = list(feature_data.keys())
        if len(feature_names) < 2:
            return correlation_results
        
        # Prepare data for correlation (use first sequence for each feature)
        prepared_features = {}
        min_length = float('inf')
        
        for feat_name, feat_arrays in feature_data.items():
            if feat_arrays:
                # Take first array and flatten if needed
                arr = feat_arrays[0]
                if arr.ndim > 1:
                    arr = arr.flatten()
                prepared_features[feat_name] = arr
                min_length = min(min_length, len(arr))
        
        # Truncate all to same length and compute correlations
        if len(prepared_features) >= 2 and min_length > 1:
            truncated_features = {name: arr[:min_length] for name, arr in prepared_features.items()}
            
            correlation_matrix = {}
            for i, feat1 in enumerate(feature_names):
                correlation_matrix[feat1] = {}
                for j, feat2 in enumerate(feature_names):
                    if feat1 in truncated_features and feat2 in truncated_features:
                        try:
                            corr, p_val = pearsonr(truncated_features[feat1], truncated_features[feat2])
                            correlation_matrix[feat1][feat2] = {
                                'correlation': float(corr) if np.isfinite(corr) else 0.0,
                                'p_value': float(p_val) if np.isfinite(p_val) else 1.0
                            }
                        except:
                            correlation_matrix[feat1][feat2] = {'correlation': 0.0, 'p_value': 1.0}
            
            correlation_results['correlation_matrix'] = correlation_matrix
            correlation_results['feature_count'] = len(feature_names)
            correlation_results['sample_length'] = min_length
        
        return correlation_results
    
    def create_temporal_mapping(self, aggregation_name: str):
        """Create detailed temporal mapping linking embeddings to original timestamps"""
        print(f"🕒 Creating temporal mapping for {aggregation_name}...")
        
        # Load original summary for timestamp information
        summary_df = pd.read_csv(self.summary_csv)
        summary_df['from_tpt'] = pd.to_datetime(summary_df['from_tpt'])
        summary_df['to_tpt'] = pd.to_datetime(summary_df['to_tpt'])
        
        if aggregation_name not in self.frame_maps:
            self.discover_embedding_structure(aggregation_name)
        
        frame_map = self.frame_maps[aggregation_name]
        
        temporal_mapping = {
            'aggregation_name': aggregation_name,
            'sequences': {},
            'global_timeline': {},
            'temporal_statistics': {}
        }
        
        all_timestamps = []
        
        for seq_key, (start_frame, end_frame) in frame_map.items():
            # Parse sequence key (format: cage_id_date)
            parts = seq_key.split('_')
            if len(parts) >= 2:
                cage_id = parts[0]
                date_str = '_'.join(parts[1:])
                
                # Find corresponding summary entry
                matching_summary = summary_df[
                    (summary_df['cage_id'] == cage_id) & 
                    (summary_df['from_tpt'].dt.strftime('%Y-%m-%d') == date_str)
                ]
                
                if not matching_summary.empty:
                    summary_row = matching_summary.iloc[0]
                    
                    # Create timestamp array for this sequence
                    start_time = summary_row['from_tpt']
                    end_time = summary_row['to_tpt']
                    total_minutes = (end_time - start_time).total_seconds() / 60
                    
                    # Create minute-by-minute timestamps
                    frame_timestamps = pd.date_range(
                        start=start_time,
                        end=end_time,
                        periods=end_frame - start_frame
                    )
                    
                    seq_temporal_info = {
                        'cage_id': cage_id,
                        'date': date_str,
                        'start_time': start_time.isoformat(),
                        'end_time': end_time.isoformat(),
                        'frame_range': (start_frame, end_frame),
                        'total_frames': end_frame - start_frame,
                        'total_minutes': total_minutes,
                        'timestamps': [t.isoformat() for t in frame_timestamps],
                        'embedding_frame_to_timestamp': {
                            i: frame_timestamps[i].isoformat() 
                            for i in range(len(frame_timestamps))
                        }
                    }
                    
                    temporal_mapping['sequences'][seq_key] = seq_temporal_info
                    all_timestamps.extend(frame_timestamps)
        
        # Global timeline statistics
        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            temporal_mapping['global_timeline'] = {
                'earliest_timestamp': all_timestamps[0].isoformat(),
                'latest_timestamp': all_timestamps[-1].isoformat(),
                'total_timespan_hours': (all_timestamps[-1] - all_timestamps[0]).total_seconds() / 3600,
                'total_frames_mapped': len(all_timestamps),
                'unique_dates': len(set(t.date() for t in all_timestamps)),
                'unique_cages': len(set(seq['cage_id'] for seq in temporal_mapping['sequences'].values()))
            }
        
        self.temporal_metadata[aggregation_name] = temporal_mapping
        
        # Save temporal mapping
        temporal_path = self.output_dir / f'{aggregation_name}_temporal_mapping.npy'
        np.save(temporal_path, temporal_mapping)
        
        self._create_temporal_report(temporal_mapping, aggregation_name)
        
        return temporal_mapping
    
    def _create_temporal_report(self, temporal_mapping: Dict, aggregation_name: str):
        """Create detailed temporal analysis report"""
        report_path = self.output_dir / f'{aggregation_name}_temporal_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("🕒 TEMPORAL MAPPING ANALYSIS\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Aggregation: {temporal_mapping['aggregation_name']}\n")
            
            if 'global_timeline' in temporal_mapping:
                gt = temporal_mapping['global_timeline']
                f.write(f"\n📅 GLOBAL TIMELINE\n")
                f.write("-" * 20 + "\n")
                f.write(f"Earliest: {gt['earliest_timestamp']}\n")
                f.write(f"Latest: {gt['latest_timestamp']}\n")
                f.write(f"Timespan: {gt['total_timespan_hours']:.2f} hours\n")
                f.write(f"Total Frames: {gt['total_frames_mapped']:,}\n")
                f.write(f"Unique Dates: {gt['unique_dates']}\n")
                f.write(f"Unique Cages: {gt['unique_cages']}\n")
            
            f.write(f"\n📊 SEQUENCE Details\n")
            f.write("-" * 20 + "\n")
            for seq_key, seq_info in list(temporal_mapping['sequences'].items())[:10]:
                f.write(f"\n🔹 {seq_key}\n")
                f.write(f"   Cage: {seq_info['cage_id']}\n")
                f.write(f"   Date: {seq_info['date']}\n")
                f.write(f"   Duration: {seq_info['total_minutes']:.1f} minutes\n")
                f.write(f"   Frames: {seq_info['total_frames']:,}\n")
                f.write(f"   Start: {seq_info['start_time']}\n")
                f.write(f"   End: {seq_info['end_time']}\n")
            
            if len(temporal_mapping['sequences']) > 10:
                f.write(f"\n   ... and {len(temporal_mapping['sequences'])-10} more sequences\n")
        
        print(f"   📝 Temporal report saved: {report_path}")
    
    def create_comprehensive_visualizations(self, aggregation_name: str):
        """Create comprehensive visualizations of the embeddings"""
        print(f"📊 Creating comprehensive visualizations for {aggregation_name}...")
        
        if aggregation_name not in self.feature_statistics:
            self.load_and_analyze_features(aggregation_name)
        
        viz_dir = self.output_dir / f'{aggregation_name}_visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        analysis = self.feature_statistics[aggregation_name]
        
        # 1. Feature Statistics Overview
        self._plot_feature_statistics_overview(analysis, viz_dir)
        
        # 2. Dimensionality Analysis
        self._plot_dimensionality_analysis(analysis, viz_dir)
        
        # 3. Category Comparison
        self._plot_category_comparison(analysis, viz_dir)
        
        # 4. Sample Data Distributions
        self._plot_sample_distributions(analysis, viz_dir)
        
        # 5. Interactive Dashboard
        self._create_interactive_dashboard(analysis, viz_dir)
        
        print(f"   🎨 Visualizations saved to: {viz_dir}")
    
    def _plot_feature_statistics_overview(self, analysis: Dict, viz_dir: Path):
        """Plot overview of feature statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Feature Statistics Overview', fontsize=16, fontweight='bold')
        
        # Collect statistics across categories
        all_means = []
        all_stds = []
        all_shapes = []
        category_names = []
        
        for cat_name, cat_data in analysis['categories'].items():
            for feat_name, feat_stats in cat_data['feature_statistics'].items():
                all_means.append(feat_stats['mean'])
                all_stds.append(feat_stats['std'])
                all_shapes.append(np.prod(feat_stats['shape']))
                category_names.append(f"{cat_name}_{feat_name}")
        
        # Mean distribution
        axes[0, 0].hist(all_means, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Feature Means')
        axes[0, 0].set_xlabel('Mean Value')
        axes[0, 0].set_ylabel('Count')
        
        # Standard deviation distribution
        axes[0, 1].hist(all_stds, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Distribution of Feature Standard Deviations')
        axes[0, 1].set_xlabel('Standard Deviation')
        axes[0, 1].set_ylabel('Count')
        
        # Feature sizes
        axes[1, 0].scatter(range(len(all_shapes)), all_shapes, alpha=0.6, s=50)
        axes[1, 0].set_title('Feature Sizes (Total Elements)')
        axes[1, 0].set_xlabel('Feature Index')
        axes[1, 0].set_ylabel('Total Elements')
        axes[1, 0].set_yscale('log')
        
        # Mean vs Std scatter
        axes[1, 1].scatter(all_means, all_stds, alpha=0.6, s=50)
        axes[1, 1].set_title('Mean vs Standard Deviation')
        axes[1, 1].set_xlabel('Mean')
        axes[1, 1].set_ylabel('Standard Deviation')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'feature_statistics_overview.png', dpi=150)
        plt.close(fig)

    def _plot_dimensionality_analysis(self, analysis: Dict, viz_dir: Path):
        """Perform PCA and plot explained variance."""
        # Find a suitable feature set for PCA (e.g., a pooled hierarchical level)
        pca_data = None
        for cat_name, cat_data in analysis['categories'].items():
            if 'sample_data' in cat_data:
                for feat_name, sample in cat_data['sample_data'].items():
                    if sample.ndim == 2 and 'pooled' in feat_name:
                        pca_data = sample
                        break
            if pca_data is not None:
                break
        
        if pca_data is None or pca_data.shape[0] < 2 or pca_data.shape[1] < 2:
            print("   ⚠️ Not enough suitable data for PCA. Skipping dimensionality plot.")
            return

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_data)
        
        pca = PCA()
        pca.fit(scaled_data)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Dimensionality Analysis (PCA)', fontsize=16, fontweight='bold')
        
        # Scree Plot
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        axes[0].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
        axes[0].set_title('Cumulative Explained Variance')
        axes[0].set_xlabel('Number of Principal Components')
        axes[0].set_ylabel('Cumulative Variance Ratio')
        axes[0].grid(True)
        
        # 2D PCA projection
        pca_2d = PCA(n_components=2)
        projected_data = pca_2d.fit_transform(scaled_data)
        axes[1].scatter(projected_data[:, 0], projected_data[:, 1], alpha=0.5)
        axes[1].set_title('2D Projection of Data (First Two Principal Components)')
        axes[1].set_xlabel('Principal Component 1')
        axes[1].set_ylabel('Principal Component 2')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / 'dimensionality_analysis_pca.png', dpi=150)
        plt.close(fig)
        
    def _plot_category_comparison(self, analysis: Dict, viz_dir: Path):
        """Compare statistics across different feature categories."""
        cat_names = list(analysis['categories'].keys())
        num_features = [len(cat_data['feature_statistics']) for cat_data in analysis['categories'].values()]
        mean_variances = []
        
        for cat_data in analysis['categories'].values():
            variances = [s['feature_variances']['mean_variance'] 
                         for s in cat_data['feature_statistics'].values() 
                         if 'feature_variances' in s]
            mean_variances.append(np.mean(variances) if variances else 0)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Category Comparison', fontsize=16, fontweight='bold')

        sns.barplot(x=cat_names, y=num_features, ax=axes[0], palette='viridis')
        axes[0].set_title('Number of Feature Types per Category')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        sns.barplot(x=cat_names, y=mean_variances, ax=axes[1], palette='plasma')
        axes[1].set_title('Average Feature Variance per Category')
        axes[1].set_ylabel('Mean Variance')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / 'category_comparison.png', dpi=150)
        plt.close(fig)

    def _plot_sample_distributions(self, analysis: Dict, viz_dir: Path):
        """Plot distributions of sample data from different features."""
        sample_data_list = []
        for cat_name, cat_data in analysis['categories'].items():
            if 'sample_data' in cat_data:
                for feat_name, sample in cat_data['sample_data'].items():
                    sample_data_list.append({'name': f"{cat_name}_{feat_name}", 'data': sample.flatten()})

        if not sample_data_list:
            print("   ⚠️ No sample data available for distribution plots.")
            return

        num_plots = len(sample_data_list)
        if num_plots == 0: return
        
        cols = 3
        rows = (num_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten()
        fig.suptitle('Sample Data Distributions', fontsize=16, fontweight='bold')

        for i, item in enumerate(sample_data_list):
            sns.histplot(item['data'], ax=axes[i], kde=True, bins=50)
            axes[i].set_title(item['name'], fontsize=10)
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Density')
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(viz_dir / 'sample_data_distributions.png', dpi=150)
        plt.close(fig)

    def _plot_correlation_heatmaps(self, analysis: Dict, viz_dir: Path):
        """Plot heatmaps for correlation matrices within categories."""
        for cat_name, cat_data in analysis['categories'].items():
            if 'correlation_analysis' in cat_data and 'correlation_matrix' in cat_data['correlation_analysis']:
                corr_matrix_dict = cat_data['correlation_analysis']['correlation_matrix']
                if not corr_matrix_dict: continue
                
                corr_df = pd.DataFrame(corr_matrix_dict)
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
                plt.title(f'Feature Correlation Heatmap for Category: {cat_name}', fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(viz_dir / f'correlation_heatmap_{cat_name}.png', dpi=150)
                plt.close()

    def _create_interactive_dashboard(self, analysis: Dict, viz_dir: Path):
        """Create an interactive Plotly dashboard."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Mean Distribution', 'Mean vs. Std Deviation', 'PCA 2D Projection', 'Category Feature Counts'),
            specs=[[{'type': 'histogram'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'bar'}]]
        )

        # Data for plots
        all_means, all_stds = [], []
        for cat_data in analysis['categories'].values():
            for feat_stats in cat_data['feature_statistics'].values():
                all_means.append(feat_stats['mean'])
                all_stds.append(feat_stats['std'])
        
        cat_names = list(analysis['categories'].keys())
        num_features = [len(cat_data['feature_statistics']) for cat_data in analysis['categories'].values()]
        
        # Plot 1: Mean Distribution
        fig.add_trace(go.Histogram(x=all_means, name='Mean'), row=1, col=1)

        # Plot 2: Mean vs Std
        fig.add_trace(go.Scatter(x=all_means, y=all_stds, mode='markers', name='Features'), row=1, col=2)
        
        # Plot 3: PCA (if data available)
        pca_data = None
        for cat_data in analysis['categories'].values():
            for sample in cat_data.get('sample_data', {}).values():
                if sample.ndim == 2: pca_data = sample; break
            if pca_data is not None: break
        
        if pca_data is not None and pca_data.shape[0] > 2 and pca_data.shape[1] > 2:
            pca_2d = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(pca_data))
            fig.add_trace(go.Scatter(x=pca_2d[:, 0], y=pca_2d[:, 1], mode='markers', name='PCA'), row=2, col=1)
        
        # Plot 4: Category Counts
        fig.add_trace(go.Bar(x=cat_names, y=num_features, name='Count'), row=2, col=2)
        
        fig.update_layout(
            title_text=f'Interactive Embedding Analysis Dashboard ({analysis["aggregation_name"]})',
            height=800,
            showlegend=False
        )
        fig.write_html(str(viz_dir / 'interactive_dashboard.html'))


def main():
    """Main execution function"""
    args = get_args()
    
    analyzer = ComprehensiveEmbeddingAnalyzer(
        embedding_dir=Path(args.embedding_dir),
        summary_csv=args.summary_csv,
        dvc_root=args.dvc_root,
        output_dir=Path(args.output_dir)
    )
    
    print("="*60)
    print("🔬 STARTING COMPREHENSIVE EMBEDDING ANALYSIS 🔬")
    print(f"Analysis for aggregation: {args.aggregation_name}")
    print(f"Output will be saved to: {args.output_dir}")
    print("="*60 + "\n")

    # Always discover structure first
    analyzer.discover_embedding_structure(args.aggregation_name)
    
    if args.temporal_analysis or args.run_all:
        analyzer.create_temporal_mapping(args.aggregation_name)

    if args.detailed_analysis or args.run_all:
        analyzer.load_and_analyze_features(args.aggregation_name)
        
    if args.create_visualizations or args.run_all:
        # Check if detailed analysis has been run, if not, run it
        if args.aggregation_name not in analyzer.feature_statistics:
            print("\nDetailed analysis required for visualizations. Running it now...")
            analyzer.load_and_analyze_features(args.aggregation_name)
        
        analyzer.create_comprehensive_visualizations(args.aggregation_name)

    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE ✅")
    print("="*60)


if __name__ == "__main__":
    main()