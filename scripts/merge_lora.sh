#!/bin/bash

# 检查输入参数
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <GPU_ID> <BASE_MODEL> <LORA_MODEL>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$1
BASE_MODEL=$2
LORA_MODEL=$3
LORA_MODEL_NAME=$(basename "$LORA_MODEL")

python open_instruct/merge_lora.py \
    --base_model_name_or_path $BASE_MODEL \
    --lora_model_name_or_path $LORA_MODEL \
    --output_dir output/$LORA_MODEL_NAME\_lora_merged \
    --save_tokenizer