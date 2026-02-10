#!/bin/bash
# LoRA fine-tuning on Mac (CPU or MPS, no CUDA/DeepSpeed).
# For a quick pipeline test: few steps, small batch, no distributed training.
#
# Gated model (e.g. MiniCPM-V-2_6): accept the license at the model page on Hugging Face,
# then either run `huggingface-cli login` once, or: HF_TOKEN="hf_xxx" ./finetune_lora_mac.sh

MODEL="openbmb/MiniCPM-o-2_6"
# Paths to your data (JSON or Parquet). For SNEI parquet set SNEI_FORMAT=true below.
DATA="~/Documents/1_Projects/Masterarbeit/datasets/SNEI_structured_train_840.parquet"
EVAL_DATA=""   # optional; leave empty for no eval
SNEI_FORMAT=true

LLM_TYPE="qwen"
# Lower to reduce RAM (2048 → 1024); increase again if you have enough memory
MODEL_MAX_Length=512
# Fewer image slices = less memory (9 → 4)
MAX_SLICE_NUMS=2

# --- Mac: single process, no DeepSpeed, no torchrun ---
# Float32 for compatibility (no fp16/bf16 on CPU; MPS can have dtype limits)
# Few steps so you can verify the pipeline quickly (increase for real training)
MAX_STEPS=20
EVAL_STEPS=10
SAVE_STEPS=10

EVAL_ARGS=""
if [ -n "$EVAL_DATA" ]; then
  EVAL_ARGS="--eval_data_path $EVAL_DATA --do_eval --evaluation_strategy steps --eval_steps $EVAL_STEPS"
fi

python finetune.py \
    --model_name_or_path "$MODEL" \
    --llm_type "$LLM_TYPE" \
    --data_path "$DATA" \
    --snei_format true \
    $EVAL_ARGS \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 false \
    --bf16_full_eval false \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --tune_vision true \
    --tune_llm false \
    --use_lora true \
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)" \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums $MAX_SLICE_NUMS \
    --max_steps $MAX_STEPS \
    --output_dir output/output_lora_mac \
    --logging_dir output/output_lora_mac \
    --logging_strategy "steps" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 2 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --report_to "none"

# Note: No --deepspeed; no torchrun. Runs on CPU by default (HuggingFace Trainer
# uses CPU when CUDA is not available). On Apple Silicon, training is slow but
# the pipeline should complete. Set DATA/EVAL_DATA and optionally SNEI_FORMAT=true.
