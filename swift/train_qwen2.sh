NPROC_PER_NODE=3 \
CUDA_VISIBLE_DEVICES=0,1,2 \
swift sft \
    --model_type qwen1half-7b-chat \
    --model_id_or_path Qwen/Qwen2-7B-Instruct \
    --dataset 'Diabetica-SFT.json' \
    --num_train_epochs 2 \
    --sft_type lora \
    --output_dir output \
    --lora_target_modules 'ALL' \
    --gradient_accumulation_steps 16 \
    --lora_dtype 'AUTO' \
    --train_dataset_sample -1 \
    --max_length 4096 \
