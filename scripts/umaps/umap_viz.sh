#!/usr/bin/env bash

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python3 downstream_tasks/umap_viz.py --labels /scratch/bhole/dvc_data/smoothed/1440/final_summary_metadata_1440.csv --embeddings /scratch/bhole/dvc_data/smoothed/models_3days/embeddings/test_3days_low.npy --out-dir umaps
python3 downstream_tasks/umap_viz.py --labels /scratch/bhole/dvc_data/smoothed/1440/final_summary_metadata_1440.csv --embeddings /scratch/bhole/dvc_data/smoothed/models_3days/embeddings/test_3days_mid.npy --out-dir umaps
python3 downstream_tasks/umap_viz.py --labels /scratch/bhole/dvc_data/smoothed/1440/final_summary_metadata_1440.csv --embeddings /scratch/bhole/dvc_data/smoothed/models_3days/embeddings/test_3days_high.npy --out-dir umaps
python3 downstream_tasks/umap_viz.py --labels /scratch/bhole/dvc_data/smoothed/1440/final_summary_metadata_1440.csv --embeddings /scratch/bhole/dvc_data/smoothed/models_3days/embeddings/test_3days_comb.npy --out-dir umaps
