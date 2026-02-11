#!/bin/bash

GPUS_PER_NODE=1  # Reduced for testing
# Note: Running without torchrun for single GPU to reduce memory overhead
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001
 
MODEL="openbmb/MiniCPM-o-2_6"
# or openbmb/MiniCPM-V-2, openbmb/MiniCPM-Llama3-V-2_5, openbmb/MiniCPM-V-2_6
# ATTENTION: paths to your training and (optional) eval data.
# Supported: .json (list of {image, conversations}) or .parquet.
# For SNEI parquet (image.bytes + ground_truth with social-navigation prompt): set SNEI_FORMAT=true and DATA/EVAL_DATA to .parquet paths.
# Otherwise: .parquet with columns "image" and "conversations", or .json. See dataset_guidance.md.
DATA="~/SNEI_structured_train_840.parquet"
EVAL_DATA=""
SNEI_FORMAT=true   # set true for SNEI parquet (image bytes + ground_truth)
# if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm, if use openbmb/MiniCPM-Llama3-V-2_5, please set LLM_TYPE="llama3",
# if use openbmb/MiniCPM-o-2_6 or openbmb/MiniCPM-V-2_6, please set LLM_TYPE=qwen
LLM_TYPE="qwen"   
MODEL_MAX_Length=256 # Reduced for testing (original: 2048, for multi-images: 4096)

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

SNEI_ARGS=""
if [ "$SNEI_FORMAT" = "true" ]; then SNEI_ARGS="--snei_format true"; fi

EVAL_ARGS=""
if [ -n "$EVAL_DATA" ]; then
    EVAL_ARGS="--eval_data_path $EVAL_DATA --do_eval --evaluation_strategy steps --eval_steps 10"
fi

# Try without torchrun first (single process, less overhead)
python finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    $EVAL_ARGS \
    $SNEI_ARGS \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 false \
    --bf16_full_eval false \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --tune_vision false \
    --tune_llm false \
    --use_lora true \
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)" \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 1 \
    --max_steps 20 \
    --output_dir output/output__lora \
    --logging_dir output/output_lora \
    --logging_strategy "steps" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 2 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --report_to "tensorboard" # wandb
    # --deepspeed ds_config_zero2.json  # Disabled for testing (requires compilation)
