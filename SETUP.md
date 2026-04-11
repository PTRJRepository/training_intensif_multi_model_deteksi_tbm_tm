# Setup Complete - TBM Detection Multi-Model

## ✅ Installation Summary

### Repository Cloned
- **Location**: `D:\Server\Services\train_tbm_tm\training_intensif_multi_model_deteksi_tbm_tm`
- **Source**: https://github.com/PTRJRepository/training_intensif_multi_model_deteksi_tbm_tm

### Core Modules Installed ✓

| Module | Version | Status |
|--------|---------|--------|
| ultralytics | 8.4.36 | ✅ Installed |
| torch | 2.11.0+cpu | ✅ Installed |
| torchvision | 0.26.0+cpu | ✅ Installed |
| opencv-python | 4.13.0 | ✅ Installed |
| numpy | 2.4.4 | ✅ Installed |
| pandas | 3.0.2 | ✅ Installed |
| pillow | 12.2.0 | ✅ Installed |
| matplotlib | 3.10.8 | ✅ Installed |
| scikit-learn | 1.8.0 | ✅ Installed |
| rasterio | 1.5.0 | ✅ Installed |
| shapely | 2.1.2 | ✅ Installed |
| pyproj | 3.7.2 | ✅ Installed |
| tqdm | 4.67.3 | ✅ Installed |
| seaborn | 0.13.2 | ✅ Installed |

### Optional Modules (Requires GDAL)

| Module | Status | Notes |
|--------|--------|-------|
| fiona | ❌ Not installed | Requires GDAL C library |
| geopandas | ❌ Not installed | Requires fiona |
| GDAL | ❌ Not installed | Requires system-level GDAL |

**To install optional geospatial modules**, use conda:
```bash
conda install -c conda-forge gdal fiona geopandas
```

## 📊 Dataset Status

### Dataset Structure
```
Dataset/dataset_pakai_ini/
├── images/train/    (17 tiles)
├── images/val/      (6 tiles)
├── labels/train/    (17 files)
├── labels/val/      (6 files)
└── data.yaml        (✓ Path corrected)
```

### Dataset Statistics
- **Training tiles**: 17 images with 5,430 trees
- **Validation tiles**: 6 images with 2,501 trees
- **Total trees**: 7,931 TBM annotations
- **Average trees per tile**: 319 (train), 417 (val)
- **BBox size**: ~10x10 pixels (small objects)
- **Average nearest neighbor distance**: 21.4 pixels

## 🛠️ Configuration Updates

### Fixed Issues
1. ✅ **data.yaml path**: Updated from old path to current location
2. ✅ **geopandas optional**: Modified `analyze_dataset.py` to work without geopandas
3. ✅ **requirements.txt**: Created with clear documentation of optional dependencies

## 🚀 Quick Start Guide

### 1. Analyze Dataset (✓ Already tested)
```bash
python Train/analyze_dataset.py
```

### 2. Train Model
```bash
# Default training (CPU, yolo11n, 150 epochs)
python Train/train.py

# Custom settings
python Train/train.py --model yolo11s --epochs 200 --batch 4

# Resume from checkpoint
python Train/train.py --resume
```

### 3. Evaluate Model
```bash
# After training
python Train/evaluate.py --weights Train/runs/exp1_yolo11n_640/weights/best.pt

# With JSON output
python Train/evaluate.py --weights best.pt --save-json
```

### 4. Run Inference
```bash
# Single image
python Train/inference.py --weights best.pt --source image.jpg --output results/

# Folder of images
python Train/inference.py --weights best.pt --source ./images/ --output results/

# Ensemble multiple models
python Train/inference.py --weights "model1.pt,model2.pt" --source image.jpg --augment
```

## 📁 Project Structure

```
training_intensif_multi_model_deteksi_tbm_tm/
├── Dataset/
│   ├── dataset_pakai_ini/          # YOLO dataset (ready to use)
│   └── Raw Dataset/                # Original drone imagery
├── Train/                          # Training modules
│   ├── config.py                   # Hyperparameters
│   ├── train.py                    # Main training script
│   ├── evaluate.py                 # Model evaluation
│   ├── inference.py                # Inference (single/ensemble)
│   └── analyze_dataset.py          # Dataset analysis
├── Tools/                          # Dataset preparation utilities
│   ├── prepare_multi_image_dataset.py
│   ├── prepare_detection_dataset.py
│   ├── clip_raster_by_polygon.py
│   └── debug_shapefile.py
├── requirements.txt                # Python dependencies
└── SETUP.md                        # This file
```

## ⚙️ Training Configuration (Default)

- **Model**: YOLO11n (nano)
- **Device**: CPU
- **Epochs**: 150
- **Batch size**: 8
- **Image size**: 640x640
- **Optimizer**: AdamW
- **Learning rate**: 0.01
- **Augmentations**: Mosaic (100%), MixUp (10%), Copy-Paste (10%)
- **Focus**: High recall for small object detection

## 📝 Notes

1. **CPU Training**: Default configuration is optimized for CPU training. GPU support available by changing `--device 0`.
2. **Small Objects**: BBoxes are ~10 pixels. Training is optimized for high recall with low confidence thresholds (0.001-0.01).
3. **Geospatial Tools**: The Tools/ scripts require GDAL/fiona/geopandas for full functionality. Install via conda if needed.
4. **Output Location**: Training results saved to `Train/runs/<experiment_name>/`

## 🔧 Troubleshooting

### If you want full geospatial support:
```bash
# Option 1: Use conda (recommended)
conda install -c conda-forge gdal fiona geopandas

# Option 2: Install from pre-compiled wheels
pip install --only-binary :all: GDAL
```

### If training is slow on CPU:
- Reduce batch size: `--batch 4`
- Use smaller model: `--model yolo11n` (already default)
- Reduce workers: Edit `config.py` → `workers = 2`

### To use GPU (if available):
```bash
python Train/train.py --device 0
```
