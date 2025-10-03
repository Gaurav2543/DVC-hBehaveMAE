# Biological Interpretability Analysis

This toolkit provides comprehensive biological interpretability analyses for hBehaveMAE embeddings.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your data paths are correct in the scripts:
   - Model checkpoint: `/scratch/bhole/dvc_hbehave_others/model_checkpoints/outputs_lz/checkpoint-00040.pth`
   - Embeddings: `/scratch/bhole/dvc_hbehave_others/extracted_embeddings/extracted_embeddings2_new_lz`
   - Labels: `dvc-data/arrays_sub20_with_cage_complete_correct_strains.npy`
   - Summary CSV: `../dvc_project/summary_table_imputed_with_sets_sub_20_CompleteAge_Strains.csv`

## Running Analyses

### CPU-based Analyses (Recommended to run first)
```bash
python cpu_interpretability.py
```

This includes:
- UMAP visualizations colored by metadata
- Clustering analysis and behavioral phenotyping
- Age trajectory analysis with GAM fitting
- Phylogenetic distance correlation (if phylo data available)
- Animated trajectory GIFs

### GPU-based Analyses (Requires CUDA)
```bash
python gpu_interpretability.py
```

This includes:
- Attention weight heatmaps
- Integrated Gradients for time importance
- Gradient-based saliency analysis

## Output Structure

```
biological_interpretability/
├── umap_visualizations/
│   ├── umap_low_metadata.png
│   ├── umap_mid_metadata.png
│   ├── umap_high_metadata.png
│   └── umap_comb_metadata.png
├── clustering_analysis/
│   ├── cluster_results_*.json
│   ├── cluster_phenotypes_*.png
│   └── cluster_metadata_*.png
├── age_trajectories/
│   ├── age_trajectory_*.png
│   └── gam_dimensions_*.png
├── phylogenetic_analysis/
│   ├── phylo_correlation_*.json
│   └── phylo_correlation_*.png
├── animations/
│   └── age_trajectory_*.gif
├── attention_analysis/     # GPU analyses
│   ├── attention_heatmap.png
│   └── attention_weights.npy
└── attribution_analysis/   # GPU analyses
    ├── ig_importance_*.png
    ├── saliency_*.png
    └── attribution_results.json
```

## Computing Requirements

### CPU Script:
- **Memory**: 16-32 GB RAM recommended
- **Time**: 30-60 minutes depending on dataset size
- **Storage**: ~500 MB for output files
- **Dependencies**: numpy, pandas, matplotlib, seaborn, scikit-learn, umap-learn, pygam, imageio

### GPU Script:
- **GPU**: CUDA-compatible GPU with 8+ GB VRAM
- **Memory**: 8-16 GB RAM
- **Time**: 15-30 minutes
- **Storage**: ~200 MB for output files
- **Dependencies**: torch, captum (plus all CPU dependencies)

## Key Features

### CPU-based Analyses:
1. **Multi-level UMAP Visualization**: Creates 2D projections of embeddings colored by biological metadata
2. **Behavioral Clustering**: K-means clustering with phenotype characterization
3. **Age Trajectory Analysis**: GAM fitting to show how embeddings change with age
4. **Phylogenetic Analysis**: Correlates embedding distances with genetic relationships
5. **Animated Trajectories**: GIF animations showing temporal progression

### GPU-based Analyses:
1. **Attention Visualization**: Heatmaps showing which time points the model focuses on
2. **Integrated Gradients**: Attribution method showing time-point importance for predictions
3. **Gradient Saliency**: Alternative attribution method for feature importance

## Troubleshooting

- If model loading fails, the GPU script will continue with synthetic data for demonstration
- If Captum is not available, attribution analyses will be skipped
- If phylogenetic distance matrix is not provided, that analysis will be skipped
- All analyses are designed to handle missing metadata gracefully

## Customization

You can modify the following parameters in the scripts:
- Number of clusters for behavioral analysis
- UMAP parameters (n_neighbors, min_dist)
- Target tasks for attribution analysis
- Animation frame rate and duration
- Color schemes and plot aesthetics