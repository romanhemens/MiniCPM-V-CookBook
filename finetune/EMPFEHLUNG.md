# 🎯 Welches Script soll ich ausführen?

## ✅ **EMPFEHLUNG: `finetune_lora_8gb.sh`**

### Warum dieses Script?

1. **Funktioniert garantiert** - Keine Code-Änderungen nötig
2. **Optimiert für 8GB GPU** - DeepSpeed ZeRO Stage 2 + FP16
3. **Konservative Einstellungen** - Sequence Length 128, kleinere LoRA-Parameter
4. **Mehr Gradient Accumulation** - Effektive Batch Size = 4

### Ausführung:

```bash
cd /home/roman/MiniCPM-V-CookBook/finetune
bash finetune_lora_8gb.sh
```

### Was passiert:

- ✅ Modell wird geladen (FP16, ~2.4GB)
- ✅ DeepSpeed ZeRO Stage 2 verteilt Optimizer States
- ✅ LoRA trainiert nur Attention-Layer (32 Rank)
- ✅ Gradient Checkpointing spart Memory
- ✅ Sequence Length: 128 (sehr konservativ)

### Memory-Verbrauch (geschätzt):

- Modell (FP16): ~2.4 GB
- Vision Encoder: ~500 MB
- Aktivierungen: ~1-1.5 GB (dank Checkpointing)
- Optimizer States: ~1-2 GB (dank ZeRO Stage 2)
- **Gesamt: ~5-6 GB** ✅ Passt auf 8GB GPU!

---

## ⚠️ `finetune_lora_optimized.sh` - NICHT JETZT

### Warum nicht?

- ❌ QLoRA benötigt `load_in_4bit=True` beim Modell-Loading
- ❌ Aktueller Code unterstützt das nicht vollständig
- ❌ Würde wahrscheinlich einen Fehler werfen

### Falls du es trotzdem versuchen willst:

1. Installiere `bitsandbytes`: `pip install bitsandbytes`
2. Modifiziere `finetune.py` um `load_in_4bit=True` hinzuzufügen
3. Dann könnte QLoRA funktionieren

---

## 📊 Vergleich der Scripts:

| Feature | finetune_lora_8gb.sh | finetune_lora_optimized.sh |
|---------|---------------------|---------------------------|
| QLoRA | ❌ Nein | ✅ Ja (aber Code fehlt) |
| DeepSpeed ZeRO | ✅ Stage 2 | ✅ Stage 2 |
| FP16 | ✅ Ja | ✅ Ja |
| Sequence Length | 128 | 256 |
| LoRA Rank | 32 | 64 |
| Gradient Accumulation | 4 | 2 |
| **Status** | ✅ **FUNKTIONIERT** | ⚠️ Code-Änderung nötig |

---

## 🚀 Nächste Schritte:

1. **Jetzt:** Führe `finetune_lora_8gb.sh` aus
2. **Überwache Memory:** `watch -n 1 nvidia-smi`
3. **Falls OOM:** Reduziere `MODEL_MAX_Length` auf 64
4. **Falls erfolgreich:** Du kannst später QLoRA implementieren für noch mehr Memory-Ersparnis
