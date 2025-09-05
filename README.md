# hBehaveMAE - Down-stream Prediction Bench

This repository shows how to

1. **extract multi-scale embeddings** from the raw DVC electrical‐activity traces with the pre-trained **hBehaveMAE** encoder;
2. **predict cage–day attributes** (currently **Age_Days** and **Strain**) from those embeddings with a library of classical regressors / classifiers; and
3. produce *turn-key* PDF reports, JSON metrics, and CSV prediction dumps for every experiment.

---

## 📂 Directory Layout
```
.
├── data_pipeline/               
│   ├── load_dvc.py
│   ├── pose_traj_dataset.py
│   └── dvc_dataset.py
│
├── downstream_tasks/    
│   ├── extract_embeddings.py 
│   ├── evaluate_embeddings.py
│   └── umap_viz.py        
│
├── engine/                      
│   └── engine_pretrain.py
│
├── models/                      
│   ├── models_defs.py
│   ├── hbehave_mae.py
│   ├── general_hiera.py
│   └── hiera_utils.py
│
├── scripts/
│    ├── train_hbehavemae.sh
│    ├── batchjobs
│    │   ├── train.sh
│    │   └── extract_embeddings.sh
│    └── downstream_tasks    
│        ├── age_regression.sh
│        └── strain_classification.sh   
│        └── umap_viz.sh
├── train.py     
├── environment.yml             
└── README.md                    
```
---

## 🔧 Installation

```bash
# 1. create the dedicated conda env
conda env create -f environment.yml
conda activate behavemae

# 2. install the repo in editable mode
pip install -e .
```

---

## How to run

1. Training

From the project root (`base_dir`), submit the training job via SLURM:

```{bash}
sbatch scripts/batchjobs/train_dvc.sh
```

This command runs `train.py`, whose arguments are introduced in `scripts/train_test/train_dvc_hBehaveMEA.sh`. In this later file, one modification is needed:

- ``--path_to_data_dir <path_to_data>``

Then on `data_pipeline/dvc_dataset.py` line 120: introduce name of `.cvs` corresponding to current experiment.

2. Embeddings generation

From the project root (`base_dir`), submit the training job via SLURM:

```{bash}
sbatch scripts/batchjobs/extract_dvc_embeddings.sh
```

This command runs `downstream_tasks/extract_dvc_embeddings.py`. Modifications required:

- `--ckpt_dir <path_to_folder_weights> --ckpt_name <best_checkpoint_name> `
- `--dvc_root <path_to_data>`

Where `<path_to_folder_weights>` is the location of the checkpoints produced during training.

3. Age classifier

The embeddings will be saved in the `output_dir` specified in the previous `.sh` file (`extracted_embeddings`):

- `test_comb.npy`
- `test_high.npy`
- `test_low.npy`
- `test_mid.npy`

Then, for this step we will run:

```{bash}
sbatch scripts/batchjobs/age_classifier.sh
```

The modifications required:

- `LABELS_PATH= <path_to_labels_file>`
- `REGRESSOR= <regressor_to_use>`

4. UMAP visualization of embeddings

This step can be runned in parallel with the age classifier. We will just need to run:

```{bash}
sbatch scripts/batchjobs/umap_viz.sh
```

Where we will need to just make sure that embeddings are in the same folder as before. 
