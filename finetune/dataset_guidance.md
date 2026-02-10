# Dataset Preparation Guide

This tutorial will guide you on how to prepare datasets for supervised fine-tuning of MiniCPM-o. Please strictly follow the format and requirements below to organize your data.

## 1. Data Format Description

Each data sample should be a dictionary containing the image path and multi-turn conversation content. For example:

```json
{
  "image": "path/to/image.jpg",
  "conversations": [
    {"role": "user", "content": "Describe this image."},
    {"role": "assistant", "content": "This is a cat."}
  ]
}
```

- `image`: Image path, supports both single and multiple images (see below).
- `conversations`: Multi-turn conversation, must start with user, and role only supports "user" and "assistant".

### Multi-image Input Format

To support multiple images, the `image` field should be a dictionary, with keys like `<image_00>`, `<image_01>`, and values as image paths:

```json
{
  "image": {
    "<image_00>": "path/to/image1.jpg",
    "<image_01>": "path/to/image2.jpg"
  },
  "conversations": [
    {"role": "user", "content": "Compare <image_00> and <image_01>."},
    {"role": "assistant", "content": "The first image is a cat, the second is a dog."}
  ]
}
```

## 2. Dataset Loading and Preprocessing

Dataset loading and preprocessing mainly rely on the `SupervisedDataset` class. The core process is as follows:

- Read raw data (in json or list format).
- Load and transform images.
- Tokenize and encode conversation content to generate input_ids, labels, position_ids, etc.
- Support advanced image processing such as slicing and patching (slice_config).

### Main Parameter Description

- `raw_data`: List of raw data samples.
- `transform`: Image preprocessing method (e.g., normalization, resizing, etc.).
- `tokenizer`: Tokenizer, should match the model.
- `slice_config`: Image slicing configuration (optional).
- `llm_type`: Large model type (e.g., "minicpm", "llama3", "qwen").
- `patch_size`: Image patch size, default is 14.
- `query_nums`: Number of image tokens, default is 64.
- `batch_vision`: Whether to process images in batch, default is False.
- `max_length`: Maximum text length, default is 2048.

## 3. Dataset Examples

Single-image multi-turn conversation example:

```json
{
  "image": "images/cat.jpg",
  "conversations": [
    {"role": "user", "content": "<image> What animal is this?"},
    {"role": "assistant", "content": "This is a cat."},
    {"role": "user", "content": "What is it doing?"},
    {"role": "assistant", "content": "It is sleeping."}
  ]
}
```

Multi-image conversation example:

```json
{
  "image": {
    "<image_00>": "images/cat.jpg",
    "<image_01>": "images/dog.jpg"
  },
  "conversations": [
    {"role": "user", "content": "Compare <image_00> and <image_01>."},
    {"role": "assistant", "content": "The first image is a cat, the second is a dog."}
  ]
}
```

## 4. SNEI Parquet (social navigation)

If your parquet has **image** (dict with `"bytes"`: raw image bytes) and **ground_truth** (JSON with `Interaction_Type`, `Description.Perception`, `Description.Prediction`, etc.), use **SNEI format** so the same user/assistant prompt as in your existing pipeline is used:

- In `finetune_lora.sh`: set `DATA` and optionally `EVAL_DATA` to your `.parquet` paths, and `SNEI_FORMAT=true`.
- Or pass `--snei_format true` to `finetune.py`.

The loader builds one user turn (with the social-navigation instruction and `<image>`) and one assistant turn (from ground_truth: REASONING, PERCEPTION, PREDICTION, INTERACTION_TYPE, CONSTRAINTS). No need to rename columns.

## 5. Parquet Datasets (generic)

You can use a **Parquet** file instead of JSON. The loader expects one row per sample.

### Required columns (default names)

- **image**: string path to the image (or a path per row). For multi-image, this column can store a JSON string like `{"<image_00>": "path1.jpg", "<image_01>": "path2.jpg"}`.
- **conversations**: list of `{"role": "user"|"assistant", "content": "..."}`. Can be stored as a JSON string (e.g. `'[{"role":"user","content":"..."}]'`) or as a list column.

If your Parquet uses different column names (e.g. `image_path`, `messages`), pass:

- `--parquet_image_column image_path`
- `--parquet_conversations_column messages`

Example: training with a Parquet file

```bash
torchrun ... finetune.py \
  --data_path /path/to/train.parquet \
  --eval_data_path /path/to/eval.parquet \
  ...
```

## 6. Finetuning Steps (Overview)

1. **Prepare data**: JSON or Parquet with `image` (+ paths) and `conversations` (user/assistant turns). Conversations must start with `user`; use `<image>` (or `<image_00>`, …) in content where the model should attend to the image(s).
2. **Set paths in the script**: In `finetune_lora.sh` set `DATA` and `EVAL_DATA` to your train (and optional eval) file.
3. **Choose model and LLM type**: Set `MODEL` (e.g. `openbmb/MiniCPM-o-2_6`) and `LLM_TYPE` (e.g. `qwen` for that model).
4. **Run training**: Execute the shell script (e.g. `bash finetune_lora.sh`) or call `torchrun` with the same arguments. The script loads data via `make_supervised_data_module` (JSON or Parquet), builds `SupervisedDataset`, and runs the trainer with LoRA/DeepSpeed as configured.

## 7. Common Issues and Notes

- The conversation must start with user, and role only supports "user" and "assistant".
- Image paths must be valid and support local paths.
- For multiple images, use `<image_xx>` placeholders in conversations to correspond to the image dictionary.
- For large images, it is recommended to configure `slice_config` for slicing.
- If data loading fails, the logger will automatically resample a data sample.

---
