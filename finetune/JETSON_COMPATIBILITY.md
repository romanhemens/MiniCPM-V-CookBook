# 🚀 Jetson AGX Orin Kompatibilität

## Aktueller Training-Status

**Was wird gerade trainiert:**
- ✅ **LoRA** (nicht QLoRA) - `--q_lora false`
- ✅ FP16 Precision
- ✅ Sequence Length: 512
- ✅ LoRA Rank: 32

**Modell-Größe nach Training:**
- Basis-Modell: `openbmb/MiniCPM-o-2_6` (~2.4B Parameter)
- LoRA Adapter: ~50-100 MB (nur die trainierten Gewichte)
- **Gesamt für Inference:** Basis-Modell + Adapter

## 📦 Was wird gespeichert?

Nach dem Training findest du im `output/output_lora_8gb/` Verzeichnis:
- `adapter_config.json` - LoRA Konfiguration
- `adapter_model.bin` oder `adapter_model.safetensors` - LoRA Gewichte (~50-100 MB)
- `training_args.bin` - Training-Parameter

**Wichtig:** Nur der Adapter wird gespeichert, nicht das komplette Modell!

## 🤖 Läuft auf Jetson AGX Orin (64GB)?

### ✅ **JA - Normales LoRA sollte funktionieren**

**Warum:**
- LoRA Adapter ist sehr klein (~50-100 MB)
- Basis-Modell wird von Hugging Face geladen
- 64GB RAM ist mehr als genug für das Modell

**Voraussetzungen:**
- PyTorch für ARM64 installiert
- Transformers, PEFT Bibliotheken
- Hugging Face Token für gated models

**Memory-Anforderungen (geschätzt):**
- Basis-Modell (FP16): ~2.4 GB
- Vision Encoder: ~500 MB
- LoRA Adapter: ~100 MB
- Aktivierungen: ~1-2 GB
- **Gesamt: ~4-5 GB** ✅ Passt auf 64GB!

### ⚠️ **QLoRA - Möglicherweise problematisch**

**Warum könnte es Probleme geben:**
- `bitsandbytes` benötigt CUDA-Kompilierung für ARM
- Nicht alle Versionen von bitsandbytes unterstützen ARM64
- Möglicherweise muss man es selbst kompilieren

**Falls QLoRA gewünscht:**
1. Prüfe ob `bitsandbytes` für ARM64 verfügbar ist
2. Falls nicht: Selbst kompilieren oder normales LoRA verwenden

## 🔧 Modell auf Jetson laden

### Normales LoRA (aktuelles Training):

```python
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

model_name = "openbmb/MiniCPM-o-2_6"
adapter_path = "path/to/output/output_lora_8gb"

# Basis-Modell laden
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # FP16 für Memory-Ersparnis
    device_map="auto"
)

# LoRA Adapter laden
model = PeftModel.from_pretrained(
    model,
    adapter_path,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model.eval()
```

### Memory-Optimierungen für Jetson:

1. **FP16 verwenden** (wie im Code oben)
2. **CPU Offloading** für Teile des Modells:
   ```python
   # Vision Encoder auf CPU lassen, LLM auf GPU
   model.vpm = model.vpm.cpu()
   ```
3. **Kleinere Batch Size** bei Inference
4. **Sequence Length reduzieren** falls nötig

## 📊 Vergleich: LoRA vs QLoRA

| Feature | LoRA (aktuell) | QLoRA |
|---------|---------------|-------|
| Modell-Größe | ~2.4 GB (FP16) | ~1.2 GB (4-bit) |
| Adapter-Größe | ~50-100 MB | ~50-100 MB |
| Kompatibilität | ✅ Funktioniert überall | ⚠️ Benötigt bitsandbytes |
| Jetson Support | ✅ Garantiert | ⚠️ Möglicherweise |
| Training Memory | ~5-6 GB | ~3-4 GB |
| Inference Memory | ~4-5 GB | ~2-3 GB |

## ✅ Empfehlung

**Für Jetson AGX Orin:**
- ✅ **Normales LoRA verwenden** (aktuelles Training)
- ✅ Funktioniert garantiert
- ✅ 64GB RAM ist mehr als genug
- ✅ Einfacher zu deployen

**QLoRA nur wenn:**
- bitsandbytes für ARM64 verfügbar ist
- Oder du es selbst kompilieren kannst
- Oder du weniger Memory brauchst (aber 64GB ist genug!)

## 🚀 Nächste Schritte

1. **Training abschließen** mit `finetune_lora_8gb.sh`
2. **Adapter testen** auf deinem Entwicklungssystem
3. **Auf Jetson deployen** mit dem Code oben
4. **Performance optimieren** falls nötig (CPU Offloading, etc.)
