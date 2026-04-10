# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Oil palm tree detection and counting from drone imagery using YOLO deep learning. Focus: **TBM (young/small palm trees)** detection with high recall and multi-model inference support.

**Challenge**: Trees appear as small objects (~10-20 pixels) with high density (2000+ per frame).

---

## Common Commands

### Dataset Analysis
```bash
python Train/analyze_dataset.py
```

### Training (CPU only)
```bash
# Default: yolo11n, 150 epochs, batch 8
python Train/train.py

# Custom settings
python Train/train.py --model yolo11s --epochs 200 --batch 4
python Train/train.py --resume
```

### Evaluation
```bash
python Train/evaluate.py --weights Train/runs/exp1/weights/best.pt
python Train/evaluate.py --weights best.pt --save-json
```

### Inference
```bash
# Single/folder inference
python Train/inference.py --weights best.pt --source image.jpg --output results/

# Ensemble multiple models
python Train/inference.py --weights "model1.pt,model2.pt" --source image.jpg --augment
```

---

## Architecture

### Project Structure
```
Training Deteksi Sawit Multi Model/
├── Tools/                      # Dataset preparation tools
│   ├── prepare_multi_image_dataset.py
│   ├── prepare_detection_dataset.py
│   ├── clip_raster_by_polygon.py
│   └── debug_shapefile.py
├── Dataset/
│   └── dataset_pakai_ini/      # YOLO dataset (17 tiles, ~5k trees)
├── Train/                       # Training module (structured)
│   ├── config.py               # Hyperparameters
│   ├── train.py                # Main training script
│   ├── evaluate.py             # Model evaluation
│   ├── inference.py            # Inference (single/ensemble)
│   └── analyze_dataset.py      # Dataset analysis
└── CLAUDE.md
```

### Data Flow
1. **Raw Data**: Drone TIFF + shapefile annotations
2. **Clip**: `clip_raster_by_polygon.py` crops to plantation blocks
3. **Prepare**: `prepare_multi_image_dataset.py` tiles (640x640, 20% overlap) → YOLO format
4. **Train**: `Train/train.py` - YOLO training with small-object augmentation
5. **Eval**: `Train/evaluate.py` - Metrics + confidence threshold analysis
6. **Infer**: `Train/inference.py` - Single model or ensemble

### Geometry Handling
- `Point`, `MultiPoint`, `Polygon` geometries supported
- CRS reprojection handled automatically
- Tile overlap prevents edge tree loss

---

## Key Files

| File | Purpose |
|------|---------|
| `Train/config.py` | All hyperparameters - easy to adjust |
| `Train/train.py` | Main training with CLI args |
| `Train/evaluate.py` | Recall-focused evaluation + threshold analysis |
| `Train/inference.py` | Multi-model ensemble inference |

---

## Dataset Structure
```
Dataset/dataset_pakai_ini/
├── images/train/  (14 tiles)
├── images/val/    (3 tiles)
├── labels/train/
├── labels/val/
└── data.yaml
```

---

## Important Notes

- **Default device: CPU** - already configured in config.py
- **bbox_size=10px** hardcoded - verify with analyze_dataset.py
- **High recall focus**: Lower conf (0.001-0.01) and IoU (0.3-0.4) thresholds
- **Multi-model ensemble** supported for better detection capability
- Training output: `Train/runs/<experiment_name>/weights/best.pt`
