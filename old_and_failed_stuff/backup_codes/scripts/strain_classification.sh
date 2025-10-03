#!/bin/bash

python dvc_downstream_tasks/classification/strain_classifier.py \
    --embeddings_path "/path/to/save/extracted_embeddings/test_submission_combined.npy" \
    --embeddings_suffix "combined_level" \
    --labels_path "/path/to/your/dvc_downstream_tasks/dvc_labels_for_classification.npy" \
    --output_dir "/path/to/save/strain_classification_results" \
    --classifier_type "LogisticRegression" # Or 'mlp', 'SVC', etc.