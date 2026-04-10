# TBM Detection Training Module

Modul training untuk deteksi TBM (Tanaman Belum Menghasilkan / Young Palm Oil Trees) dari citra drone.

## Struktur Proyek

```
Train/
├── __init__.py           # Package init
├── config.py             # Konfigurasi training & hyperparameters
├── train.py              # Script training utama
├── evaluate.py           # Evaluasi model
├── inference.py          # Inference (single/ensemble)
├── analyze_dataset.py    # Analisis dataset
└── README.md             # Dokumentasi
```

## Quick Start

### 1. Analisis Dataset
```bash
python Train/analyze_dataset.py
```
Melihat distribusi tree, density, dan rekomendasi augmentasi.

### 2. Training Model
```bash
python Train/train.py --model yolo11n --epochs 100
```

**Dengan opsi tambahan:**
```bash
# Model lebih besar (lebih akurat tapi lebih lambat)
python Train/train.py --model yolo11s --epochs 150 --batch 4

# Training dari checkpoint yang terputus
python Train/train.py --resume

# Custom learning rate
python Train/train.py --lr 0.005 --epochs 200
```

### 3. Evaluasi
```bash
python Train/evaluate.py --weights Train/runs/exp1_yolo11n_640/weights/best.pt
```

**Analisis confidence threshold untuk high recall:**
```bash
python Train/evaluate.py --weights best.pt --save-json
```

### 4. Inference
```bash
# Single image
python Train/inference.py --weights best.pt --source image.jpg --output results/

# Folder
python Train/inference.py --weights best.pt --source ./images/ --output results/

# Ensemble (multiple models)
python Train/inference.py --weights "model1.pt,model2.pt" --source image.jpg
```

## Konfigurasi

### Model
| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `--model` | yolo11n | Arsitektur: yolov8n, yolov8s, yolo11n, yolo11s, etc. |

### Training
| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `--epochs` | 150 | Jumlah epoch |
| `--batch` | 8 | Batch size (CPU: 4-8, GPU: 16-32) |
| `--img-size` | 640 | Ukuran input |
| `--lr` | 0.01 | Learning rate |
| `--optimizer` | AdamW | Optimizer |

### Augmentasi (Small Object Focus)
| Parameter | Default | Efek pada Recall |
|-----------|--------|-----------------|
| `--mosaic` | 1.0 | Meningkatkan context learning |
| `--mixup` | 0.1 | Augmentasi MixUp |
| `--copy-paste` | 0.1 | Duplikasi objek kecil |

### Inference (High Recall)
| Parameter | Default | Efek |
|-----------|---------|------|
| `--conf` | 0.1 | Confidence threshold (rendah = lebih banyak deteksi) |
| `--iou` | 0.3 | IoU threshold untuk NMS |

## Tips untuk High Recall

1. **Confidence Threshold Rendah**
   - Gunakan `--conf 0.001` atau `--conf 0.01` saat inference
   - trade-off: lebih banyak false positives

2. **IoU Threshold Rendah**
   - Gunakan `--iou 0.3` atau `--iou 0.4`
   - Membolehkan boxes yang overlap lebih banyak

3. **TTA (Test Time Augmentation)**
   - Gunakan `--augment` saat inference
   - Meningkatkan detection capability

4. **Ensemble Models**
   - Gabungkan 2-3 model dengan arsitektur berbeda
   - Rata-rata predictions untuk hasil lebih robust

## Multi-Model Inference

Untuk meningkatkan detection capability, gunakan ensemble:

```bash
python Train/inference.py \
    --weights "Train/runs/exp1/weights/best.pt,Train/runs/exp2/weights/best.pt" \
    --source image.jpg \
    --conf 0.01
```

## CPU vs GPU

| Aspek | CPU | GPU |
|-------|-----|-----|
| Batch Size | 4-8 | 16-32 |
| Training Time | ~10x lebih lambat | Baseline |
| Workers | 2-4 | 8 |
| Memory | < 8GB RAM cukup | VRAM 8GB+ |

Konfigurasi default sudah di-set untuk **CPU only**.

## Output

Training outputs tersimpan di:
```
Train/runs/<experiment_name>/
├── weights/
│   ├── best.pt      # Best model berdasarkan mAP
│   └── last.pt      # Last checkpoint
├── results.csv      # Training metrics
├── results.png      # Training curves
└── args.yaml        # Training arguments
```
