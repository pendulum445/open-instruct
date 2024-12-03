#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <DATASET_NAME> <CUDA_VISIBLE_DEVICES> (e.g., $0 0,1 my_dataset)"
    exit 1
fi

DATASET=$1
export CUDA_VISIBLE_DEVICES=$2

MODEL=Meta-Llama-3-8B
MODEL_SIZE=8B
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=256
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training $MODEL using GPUs $CUDA_VISIBLE_DEVICES, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

bash scripts/report_to_lark.sh "Training $MODEL-$DATASET is starting using GPUs $CUDA_VISIBLE_DEVICES."

report_error() {
    local error_message="$1"
    bash scripts/report_to_lark.sh "Training $MODEL-$DATASET failed: $error_message"
    exit 1
}

# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory, 
# but it will trade off speed.
accelerate launch \
    --main_process_port 29500 \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path ../$MODEL \
    --use_flash_attn \
    --tokenizer_name ../$MODEL \
    --use_slow_tokenizer \
    --train_file $DATASET.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --checkpointing_steps epoch \
    --output_dir output/$MODEL-$DATASET \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --with_tracking || report_error "Error during training."

bash scripts/report_to_lark.sh "Training $MODEL-$DATASET finished successfully. Begin evaluation."

bash scripts/eval/all.sh $MODEL-$DATASET $CUDA_VISIBLE_DEVICES

bash scripts/report_to_lark.sh "Evaluation of $MODEL-$DATASET finished."

EVAL_RESULTS=$(python show_eval_results.py --models $MODEL-$DATASET) || report_error "Error while fetching evaluation results."

echo "Evaluation Results for $MODEL-$DATASET:"
echo "$EVAL_RESULTS"

bash scripts/report_to_lark.sh "Evaluation results for $MODEL-$DATASET: $EVAL_RESULTS"