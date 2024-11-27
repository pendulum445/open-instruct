#!/bin/bash

# 检查输入参数
if [ -z "$1" ]; then
    echo "Usage: $0 <CUDA_VISIBLE_DEVICES> (e.g., $0 0,1)"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$1

MODEL=Meta-Llama-3-8B
DATASET=alpaca
MODEL_SIZE=8B
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)  # 自动计算使用的GPU数量
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=256
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training $MODEL using GPUs $CUDA_VISIBLE_DEVICES, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# 训练开始上报
bash scripts/report_to_lark.sh "Training $MODEL-$DATASET is starting using GPUs $CUDA_VISIBLE_DEVICES."

# 错误处理函数
report_error() {
    local error_message="$1"
    bash scripts/report_to_lark.sh "Training $MODEL-$DATASET failed: $error_message"
    exit 1
}

# Lora training
accelerate launch \
    --main_process_port 29502 \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path /data/shared/$MODEL \
    --use_flash_attn \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --tokenizer_name /data/shared/$MODEL \
    --use_slow_tokenizer \
    --train_file /data/lyj/FinLLM/alpaca-train.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 5 \
    --report_to "tensorboard" \
    --output_dir output/$MODEL-$DATASET\_lora/ \
    --logging_steps 1 \
    --with_tracking || report_error "Error during Lora training."

# Merging LoRA layers
python open_instruct/merge_lora.py \
    --base_model_name_or_path /data/shared/$MODEL \
    --lora_model_name_or_path output/$MODEL-$DATASET\_lora/ \
    --output_dir output/$MODEL-$DATASET\_lora_merged/ \
    --save_tokenizer || report_error "Error during merging LoRA layers."

# 成功完成上报
bash scripts/report_to_lark.sh "Training $MODEL-$DATASET finished successfully."
