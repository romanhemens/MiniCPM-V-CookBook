#!/bin/bash

# Optimiertes Script für A100 (40GB) mit LoRA
# Nutzt die volle Power der A100 für schnelleres Training
# LoRA wird verwendet für Jetson-Kompatibilität

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
MODEL_MAX_Length=2048  # Volle Sequence Length für A100 (wie im Original-Kommentar)

SNEI_ARGS=""
if [ "$SNEI_FORMAT" = "true" ]; then SNEI_ARGS="--snei_format true"; fi

EVAL_ARGS=""
if [ -n "$EVAL_DATA" ]; then
    EVAL_ARGS="--eval_data_path $EVAL_DATA --do_eval --evaluation_strategy steps --eval_steps 10"
fi

# OPTIMIERUNGEN FÜR A100:
# 1. BF16 Precision - nutzt Tensor Cores der A100 optimal
# 2. Größere Batch Size - A100 hat 40GB VRAM
# 3. Volle Sequence Length (2048) - wie im Original vorgesehen
# 4. LoRA Rank 64 (Default) - gute Balance zwischen Performance und Größe
# 5. Gradient Checkpointing - optional, kann bei Bedarf deaktiviert werden
# 6. DeepSpeed optional - kann aktiviert werden für noch mehr Optimierung

echo "=========================================="
echo "Starting LoRA training for A100 (40GB)"
echo "Model: $MODEL"
echo "Data: $DATA"
echo "Max Length: $MODEL_MAX_Length"
echo "Using BF16 + Larger Batch Size"
echo "=========================================="

python finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    $EVAL_ARGS \
    $SNEI_ARGS \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --tune_vision false \
    --tune_llm false \
    --use_lora true \
    --q_lora false \
    --lora_r 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)" \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 1 \
    --max_steps 1000 \
    --output_dir output/output_lora_a100 \
    --logging_dir output/output_lora_a100 \
    --logging_strategy "steps" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --gradient_checkpointing true \
    --report_to "tensorboard"
    # DeepSpeed optional - aktivieren falls gewünscht:
    # --deepspeed ds_config_zero2.json
