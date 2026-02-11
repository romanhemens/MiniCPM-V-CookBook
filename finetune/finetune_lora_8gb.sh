#!/bin/bash

# Optimiertes Script für 8GB GPU (RTX 4060)
# Verwendet: DeepSpeed ZeRO Stage 2 + FP16 + Gradient Checkpointing
# Falls QLoRA nicht funktioniert, verwende diese Version

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
MODEL_MAX_Length=512  # Erhöht - 128 war zu kurz (keine Assistant-Tokens für Loss)


SNEI_ARGS=""
if [ "$SNEI_FORMAT" = "true" ]; then SNEI_ARGS="--snei_format true"; fi

EVAL_ARGS=""
if [ -n "$EVAL_DATA" ]; then
    EVAL_ARGS="--eval_data_path $EVAL_DATA --do_eval --evaluation_strategy steps --eval_steps 10"
fi

# OPTIMIERUNGEN FÜR 8GB GPU (OHNE DeepSpeed):
# 1. BF16 Precision - halbiert Memory-Verbrauch (besser als FP16 für Training)
# 2. Gradient Checkpointing - spart Memory bei Aktivierungen
# 3. Sequence Length 512 (erhöht von 128 - benötigt für Assistant-Tokens)
# 4. Batch Size 1 + Gradient Accumulation (effektive Batch Size = 8)
# 5. Kleine LoRA-Parameter (Rank 32)
# 
# Hinweis: DeepSpeed wurde entfernt (benötigt mpi4py).
# Falls du DeepSpeed verwenden willst: pip install mpi4py
# 
# FP16 wurde zu BF16 geändert wegen Kompatibilität mit AMP GradScaler

echo "=========================================="
echo "Starting optimized training for 8GB GPU"
echo "Model: $MODEL"
echo "Data: $DATA"
echo "Max Length: $MODEL_MAX_Length"
echo "Using BF16 + Gradient Checkpointing (NO DeepSpeed)"
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
    --lora_r 32 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)" \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 1 \
    --max_steps 5 \
    --output_dir output/output_lora_8gb \
    --logging_dir output/output_lora_8gb \
    --logging_strategy "steps" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
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
    --report_to "tensorboard"
    # DeepSpeed entfernt - benötigt mpi4py für Single-GPU
    # Falls du DeepSpeed verwenden willst: pip install mpi4py
