#!/bin/bash

# 检查输入参数
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <GPU_ID> <MODEL>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$1"
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM="false"

MODEL="$2"

python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/$MODEL \
    --model_name_or_path output/$MODEL \
    --use_vllm &&

python -m eval.ifeval.run_eval \
    --data_dir data/eval/ifeval/ \
    --save_dir results/ifeval/$MODEL \
    --model output/$MODEL \
    --tokenizer output/$MODEL \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm &&

python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --save_dir results/codex_humaneval/$MODEL \
    --model_name_or_path output/$MODEL \
    --use_vllm &&

python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --save_dir results/gsm/$MODEL \
    --model_name_or_path output/$MODEL \
    --n_shot 8 \
    --use_vllm &&

python -m eval.MATH.run_eval \
    --data_dir data/eval/MATH/ \
    --save_dir results/MATH/$MODEL \
    --model_name_or_path output/$MODEL \
    --n_shot 4 \
    --use_vllm &&

python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/$MODEL \
    --model_name_or_path output/$MODEL \
    --tokenizer_name_or_path output/$MODEL \
    --eval_batch_size 4 \
    --load_in_8bit &&

python -m eval.truthfulqa.run_eval \
    --data_dir data/eval/truthfulqa \
    --save_dir results/truthfulqa/$MODEL \
    --model_name_or_path output/$MODEL \
    --tokenizer_name_or_path output/$MODEL \
    --metrics mc \
    --preset qa \
    --eval_batch_size 20 \
    --load_in_8bit