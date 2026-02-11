#!/bin/bash

# Optimiertes Script für 8GB GPU (RTX 4060)
# Verwendet: QLoRA (4-bit) + DeepSpeed ZeRO Stage 2 + FP16

GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001
 
MODEL="openbmb/MiniCPM-o-2_6"
DATA="~/SNEI_structured_train_840.parquet"
EVAL_DATA=""
SNEI_FORMAT=true
LLM_TYPE="qwen"   
MODEL_MAX_Length=256  # Reduziert für 8GB GPU

SNEI_ARGS=""
if [ "$SNEI_FORMAT" = "true" ]; then SNEI_ARGS="--snei_format true"; fi

EVAL_ARGS=""
if [ -n "$EVAL_DATA" ]; then
    EVAL_ARGS="--eval_data_path $EVAL_DATA --do_eval --evaluation_strategy steps --eval_steps 10"
fi

# OPTIMIERUNGEN FÜR 8GB GPU:
# 1. QLoRA aktivieren (4-bit Quantization) - reduziert Modell von ~5GB auf ~1.2GB
# 2. DeepSpeed ZeRO Stage 2 - verteilt Optimizer States
# 3. FP16 Precision - halbiert Memory-Verbrauch
# 4. Gradient Checkpointing - spart Memory bei Aktivierungen
# 5. Kleine Batch Size + Gradient Accumulation

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
    --fp16 true \
    --fp16_full_eval true \
    --do_train \
    --tune_vision false \
    --tune_llm false \
    --use_lora true \
    --q_lora true \
    --lora_r 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)" \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 1 \
    --max_steps 20 \
    --output_dir output/output_lora_optimized \
    --logging_dir output/output_lora_optimized \
    --logging_strategy "steps" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
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
    --report_to "tensorboard" \
    --deepspeed ds_config_zero2.json
