experiment=experiment5

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python run_test.py \
    --path_to_data_dir dvc-data/Imputed_single_cage \
    --dataset dvc \
    --embedsum False \
    --fast_inference True \
    --batch_size 1 \
    --model gen_hiera \
    --input_size 1440 1 12 \
    --stages 2 3 4 \
    --q_strides "15,1,1;6,1,1" \
    --mask_unit_attn True False False \
    --patch_kernel 2 1 12 \
    --init_embed_dim 96 \
    --init_num_heads 2 \
    --out_embed_dims 64 96 128 \
    --distributed \
    --num_frames 1440 \
    --pin_mem \
    --num_workers 8 \
    --output_dir dvc-data/outputs/${experiment} \
    --combine_embeddings True \

cd hierAS-eval_new

nr_submissions=$(ls ../outputs/dvc/${experiment}/test_submission_* 2>/dev/null | wc -l)
files=($(seq 0 $((nr_submissions - 1))))
