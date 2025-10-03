export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "Extracting embeddings..."
python3 downstream_tasks/embeddings.py \
  --dvc_root /scratch/bhole/dvc_data/smoothed/cage_activations_full_data_final \
  --summary_csv /scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv \
  --ckpt_dir /scratch/bhole/dvc_data/smoothed/models_3days \
  --ckpt_name checkpoint-09999.pth \
  --output_dir /scratch/bhole/dvc_data/smoothed/models_3days/embeddingsssssss \
  --time_aggregations 4320 \
  --aggregation_names 3days \
  --batch_size 32 \

echo "Embedding extraction completed"




export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "Extracting embeddings..."
python3 downstream_tasks/optimized_embeddings.py \
  --dvc_root /scratch/bhole/dvc_data/smoothed/cage_activations_full_data_final \
  --summary_csv /scratch/bhole/dvc_data/smoothed/4320/summary_metadata_4320.csv \
  --output_dir /scratch/bhole/dvc_data/smoothed/models_3days/stupid_optimized_embeddings \
  --zarr_path /scratch/bhole/dvc_data/smoothed/models_3days/stupid_optimized_embeddings_zarr \
  --ckpt_dir /scratch/bhole/dvc_data/smoothed/models_3days \
  --ckpt_name checkpoint-09999.pth \
  --time_aggregations 4320 \
  --aggregation_names 3days \
  --batch_size 32 --extract_all

echo "Embedding extraction completed"
