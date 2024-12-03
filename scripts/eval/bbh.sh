# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0

# List of models to evaluate
models=(
    Meta-Llama-3-8B-alpaca_lora_merged
    Meta-Llama-3-8B-alpaca-level-1_lora_merged
    Meta-Llama-3-8B-alpaca-level-3_lora_merged
    Meta-Llama-3-8B-alpaca-level-5_lora_merged
    Meta-Llama-3-8B-alpaca-dpo-level-3-dpo_lora_merged
    Meta-Llama-3-8B-alpaca-dpo-level-5-dpo_lora_merged
    # Add more models here
)

echo "Starting BBH evaluation for the following models:"
echo "${models[@]}"
echo "-----------------------------------------------"

# Loop through each model and run evaluation
for MODEL in "${models[@]}"; do
    echo "Evaluating model: $MODEL"
    python -m eval.bbh.run_eval \
        --data_dir data/eval/bbh \
        --save_dir results/bbh/$MODEL \
        --model_name_or_path output/$MODEL \
        --use_vllm
done
