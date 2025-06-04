# Experimental environment: A10, 3090, V100
# 20GB GPU memory
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=3,4,5,6 \
swift sft \
    --model_type deepseek-vl-7b-chat \
    --model_id_or_path /data/weilai/weilai_code/model_weights/deepseek-vl-7b-chat \
    --dataset /data/weilai/weilai_code/swift-old/dataset/detail23k/test.json \
    --num_train_epochs 3 \
    --sft_type lora \
    --output_dir output \
    # --lora_target_modules 'ALL' \
    # --gradient_accumulation_steps 16 \
    # --lora_dtype 'AUTO' \