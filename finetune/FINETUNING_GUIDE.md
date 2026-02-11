# Finetuning Guide - MiniCPM-o-2_6 mit LoRA

## 📋 Was passiert beim Aufruf von `finetune_lora.sh`?

### Schritt-für-Schritt Ablauf:

1. **Script startet** (`finetune_lora.sh`)
   - Setzt Umgebungsvariablen (MODEL, DATA, etc.)
   - Konfiguriert Distributed Training Parameter (für Multi-GPU, hier: Single GPU)

2. **Python Training Script startet** (`finetune.py`)
   - Lädt das Modell: `openbmb/MiniCPM-o-2_6` (~2.4B Parameter)
   - Lädt Tokenizer
   - Initialisiert Vision Encoder (VPM)

3. **Daten werden geladen**
   - Parquet-Datei wird eingelesen (`SNEI_structured_train_840.parquet`)
   - SNEI-Format: Bilder als Bytes, Ground Truth als JSON
   - Konvertiert zu Conversations-Format mit User/Assistant Prompts

4. **LoRA wird konfiguriert**
   - Nur Attention-Layer werden trainiert: `q_proj`, `k_proj`, `v_proj`, `o_proj`
   - Vision Encoder wird **nicht** trainiert (`tune_vision=false`)
   - LLM Basis wird **nicht** trainiert (`tune_llm=false`)
   - Nur kleine LoRA-Adapter werden trainiert (~1-2% der Parameter)

5. **Training startet**
   - Batch Size: 1 (sehr klein!)
   - Gradient Accumulation: 1
   - Max Steps: 20
   - Sequence Length: 256 (reduziert von 2048)
   - Gradient Checkpointing: aktiviert (spart Memory)

## 🔴 Warum bekommst du OOM (Out of Memory)?

### Memory-Verbrauch Analyse:

**Deine GPU:** RTX 4060 mit **8GB VRAM**

**Memory-Verbrauch pro Komponente:**

1. **Modell (FP32):** ~9.6 GB (2.4B × 4 bytes)
2. **Modell (FP16/BF16):** ~4.8 GB (2.4B × 2 bytes)
3. **Vision Encoder:** ~500 MB - 1 GB
4. **Aktivierungen (Forward Pass):** ~1-2 GB
5. **Gradienten:** ~4.8 GB (bei FP16)
6. **Optimizer States (AdamW):** ~9.6 GB (2× Modell-Größe)
7. **LoRA Adapter:** ~50-100 MB (vernachlässigbar)

**Gesamt (ohne Optimierungen):** ~20-25 GB VRAM benötigt!

### Warum trotz LoRA OOM?

- **Vision Encoder** bleibt im Memory (auch wenn nicht trainiert)
- **Optimizer States** für alle Parameter (auch wenn nur LoRA trainiert wird)
- **Aktivierungen** während Forward/Backward Pass
- **Gradient Checkpointing** hilft, aber nicht genug bei 8GB

## ✅ Lösungen für 8GB GPU

### Option 1: QLoRA (4-bit Quantization) - **EMPFOHLEN**

QLoRA reduziert Modell-Größe auf ~1.2 GB (4-bit statt 16-bit):
- Modell: ~1.2 GB (4-bit)
- Vision Encoder: ~500 MB
- Aktivierungen: ~1-2 GB
- Optimizer States: nur für LoRA (~100 MB)
- **Gesamt: ~3-4 GB** ✅

### Option 2: DeepSpeed ZeRO Stage 2/3

ZeRO verteilt Optimizer States und Gradienten:
- ZeRO Stage 2: Optimizer States werden partitioniert
- ZeRO Stage 3: Auch Parameter werden partitioniert
- **Reduziert Memory um 50-70%**

### Option 3: Kombination (QLoRA + DeepSpeed)

Beste Lösung für 8GB GPU!

## 🚀 Optimiertes Script

Siehe `finetune_lora_optimized.sh` für optimierte Version mit:
- QLoRA aktiviert
- DeepSpeed ZeRO Stage 2
- FP16/BF16 Precision
- Optimierte Batch Size und Sequence Length

## 📊 Brauchst du eine A100?

**Nein!** Mit den richtigen Optimierungen funktioniert es auf 8GB GPU:

- **A100 (40GB):** Für Full Fine-Tuning ohne Optimierungen
- **RTX 4060 (8GB):** Mit QLoRA + DeepSpeed ✅
- **RTX 3090 (24GB):** Mit LoRA + DeepSpeed ✅
- **RTX 4090 (24GB):** Mit LoRA + DeepSpeed ✅

## 🔧 Installation für QLoRA

```bash
pip install bitsandbytes
```

## 📝 Nächste Schritte

1. Installiere `bitsandbytes` falls noch nicht vorhanden
2. Verwende `finetune_lora_optimized.sh`
3. Überwache Memory mit `nvidia-smi -l 1`
4. Falls immer noch OOM: Reduziere `model_max_length` weiter oder verwende ZeRO Stage 3
