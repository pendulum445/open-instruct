# you need 8 GPUs for full finetuning
export CUDA_VISIBLE_DEVICES=2,3

MODEL=llama-2-7b-hf
DATASET=alpaca-train
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=32
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
# echo "Training $MODEL using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# accelerate launch \
#     --main_process_port 29501 \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes $NUM_GPUS \
#     --use_deepspeed \
#     --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
#     open_instruct/dpo_tune.py \
#     --model_name_or_path /data/lyj/hf_models/$MODEL \
#     --use_lora \
#     --lora_rank 64 \
#     --lora_alpha 16 \
#     --lora_dropout 0.1 \
#     --use_flash_attn \
#     --gradient_checkpointing \
#     --tokenizer_name /data/lyj/hf_models/$MODEL \
#     --use_slow_tokenizer \
#     --train_file /data/lyj/open-instruct/alpaca-train-dpo.json \
#     --max_seq_length 2048 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --learning_rate 1e-6 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0. \
#     --num_train_epochs 3 \
#     --output_dir output/$MODEL-$DATASET-dpo_lora \
#     --with_tracking \
#     --torch_dtype bfloat16 \
#     --report_to tensorboard \
#     --seed 42 \
#     --logging_steps 1 &&

# python open_instruct/merge_lora.py \
#     --base_model_name_or_path /data/lyj/hf_models/$MODEL \
#     --lora_model_name_or_path output/$MODEL-$DATASET-dpo_lora \
#     --output_dir output/$MODEL-$DATASET-dpo_lora_merged/ \
#     --save_tokenizer \
#     --pad_to_multiple_of 1 &&

bash scripts/eval/all.sh $MODEL-$DATASET-dpo_lora_merged