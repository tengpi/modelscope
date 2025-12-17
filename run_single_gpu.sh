#!/bin/bash

# 单机单 GPU DeepSpeed 训练脚本

export CUDA_VISIBLE_DEVICES=0

deepspeed --num_gpus=1 train_lora_deepspeed.py \
    --model_path /mnt/workspace/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B-Instruct \
    --data_prefix /mnt/workspace/projects/modelscope/data/ \
    --output_dir ./qwen2-1.5b-en2zh-qlora-deepspeed-1k \
    --num_samples 1000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 100 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --use_4bit \
    --deepspeed ds_config_single_gpu.json
