# hBehaveMAE Embedding Extraction Pipeline (VERSION 7.0)

This version (7.0) is the most robust and contains:
- Integrated README and documentation.
- Corrected Dask logic to fix the "horizontal lines" and broken x-axis bug.
- Multiprocessing.Manager to prevent the script from hanging.
- Optional normalization, in-script summary, and day/night plotting per cage.

---

## 1. Overview

This script provides a high-performance pipeline for extracting hierarchical 
behavioral embeddings from time-series data using a pre-trained `hBehaveMAE` model.

The workflow is designed for scalability and scientific validity. It operates
in two main stages: a one-time data conversion to a Zarr store, and a parallelized
embedding extraction process. The output is a set of high-resolution, per-minute
feature vectors (embeddings) with corresponding timestamps, which are essential for
downstream tasks like age regression, strain classification, or circadian analysis.

---

## 2. How to Run

The script is run from the command line. You must provide paths to your data,
model, and desired outputs, along with the model's architectural parameters, which
**must exactly match** the ones used for training.

### Example Command
```bash
python extract_embeddings.py \
  --dvc_root /path/to/raw/csv_files \
  --summary_csv /path/to/summary_metadata_40320.csv \
  --ckpt_path /path/to/model/checkpoint-best.pth \
  --stats_path /path/to/zscore_stats.npy \
  --zarr_path /path/to/your_zarr_store.zarr \
  --output_dir /path/to/your/output_embeddings \
  --time_aggregations 40320 \
  --aggregation_names 4weeks \
  --batch_size 16 \
  --multi_gpu \
  \
  # --- Model Architecture (MUST MATCH TRAINING) ---
  --model hbehavemae \
  --stages 2 3 4 4 5 5 6 \
  --q_strides "6,1,1;4,1,1;3,1,1;4,1,1;2,1,1;2,1,1" \
  --out_embed_dims 128 160 192 192 224 224 256
```

---

## 3. Output Files

For each level of the hierarchy (e.g., `level1`, `comb`), the script generates:

**a) `.npz` files**
(e.g., `embeddings_4weeks_level1.npz`) containing:

* `embeddings`: 2D NumPy array `(num_windows, embedding_dim)`.
* `timestamps`: 1D array of `datetime64` objects.
* `frame_map`: dictionary mapping `cage_id` → slice of rows.

**b) Day/Night Heatmaps**
(e.g., `heatmap_4weeks_level1_CAGE-ID_DAY.png`):

* Separate visualizations for daytime (7am–7pm) and nighttime (7pm–7am).

---

## 4. Interpreting the Embeddings

* **Dimensions:** abstract features learned by the model. Their meaning is inferred by correlating with known behaviors.
* **Hierarchy:**

  * *Low-level embeddings* (e.g., `level1`, `level2`): short-term, fine-grained behaviors.
  * *High-level embeddings* (e.g., `level7`): long-term, coarse-grained states.

For analyzing specific time blocks (e.g., a single night), low-level embeddings are usually most appropriate.

---

## 5. Post-processing Example

```python
import numpy as np
import pandas as pd

# Load the data
data = np.load('embeddings_4weeks_level1.npz', allow_pickle=True)
embeddings = data['embeddings']
timestamps = data['timestamps']
frame_map = data['frame_map'].item()

# Create a DataFrame for easy filtering
df = pd.DataFrame(embeddings)
df['timestamp'] = timestamps

# Filter for a specific night
start_night = pd.to_datetime('2020-02-15 19:00:00')
end_night = pd.to_datetime('2020-02-16 07:00:00')

night_df = df[(df['timestamp'] >= start_night) & (df['timestamp'] < end_night)]
nocturnal_embeddings = night_df.drop(columns=['timestamp']).values
```

Now, `nocturnal_embeddings` are ready for downstream analysis.

