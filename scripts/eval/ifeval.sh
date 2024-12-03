# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=3

models=(
    Meta-Llama-3-8B-alpaca_lora_merged
    Meta-Llama-3-8B-alpaca-level-1_lora_merged
    Meta-Llama-3-8B-alpaca-level-3_lora_merged
    Meta-Llama-3-8B-alpaca-level-5_lora_merged
    Meta-Llama-3-8B-alpaca-dpo-level-3-dpo_lora_merged
    Meta-Llama-3-8B-alpaca-dpo-level-5-dpo_lora_merged
    # Add more models here
)

echo "Starting GSM evaluation for the following models:"
echo "${models[@]}"
echo "-----------------------------------------------"

# Loop through each model and run evaluation
for MODEL in "${models[@]}"; do
    echo "Evaluating model: $MODEL"
    python -m eval.ifeval.run_eval \
    --data_dir data/eval/ifeval/ \
    --save_dir results/ifeval/$MODEL \
    --model output/$MODEL \
    --tokenizer output/$MODEL \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm
done


# Evaluating tulu 7B model using chat format
# python -m eval.ifeval.run_eval \
#     --data_dir data/eval/ifeval/ \
#     --save_dir results/ifeval/llama-2-7b-hf-alpaca-train_lora_merged \
#     --model /data/lyj/open-instruct/output/llama-2-7b-hf-alpaca-train_lora_merged \
#     --tokenizer /data/lyj/open-instruct/output/llama-2-7b-hf-alpaca-train_lora_merged \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --use_vllm


# # Evaluating tulu 70B dpo model using chat format
# python -m eval.ifeval.run_eval \
#     --data_dir data/eval/ifeval/ \
#     --save_dir results/ifeval/tulu-70B-dpo \
#     --model allenai/tulu-2-dpo-70b \
#     --tokenizer allenai/tulu-2-dpo-70b \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --use_vllm

# # Evaluating chatgpt
# python -m eval.ifeval.run_eval \
#     --data_dir data/eval/ifeval/ \
#     --save_dir results/ifeval/chatgpt-no-cot \
#     --openai_engine "gpt-3.5-turbo-0125" \
#     --eval_batch_size 20


# # Evaluating gpt4
# python -m eval.ifeval.run_eval \
#     --data_dir data/eval/ifeval/ \
#     --save_dir results/ifeval/gpt4-cot \
#     --openai_engine "gpt-4-0613" \
#     --eval_batch_size 20