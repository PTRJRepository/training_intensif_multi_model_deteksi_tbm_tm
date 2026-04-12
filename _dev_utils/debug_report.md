"""
TBM Training Analysis & Debug Report
Generated: 2026-04-11
"""

# =============================================================================
# ISSUE 1: OSError when saving last.pt
# =============================================================================
# Cause: File path issue with backslashes on Windows
# Fix: Use forward slashes in path

# Current (broken):
#   project_cfg.save_dir = Path(__file__).parent / "runs"
#   last.pt path: D:\Server\Services\train_tbm_tm\...\weights\last.pt

# Solution: Convert to forward slashes or use raw string
# =============================================================================


# =============================================================================
# ISSUE 2: LOW ACCURACY ANALYSIS (mAP50=0.453, mAP50-95=0.147)
# =============================================================================

## Root Causes:

### 1. SMALL DATASET
- Train images: 17 tiles
- Val images: 6 tiles
- Total: Only 23 images

### 2. VERY SMALL OBJECTS
- BBox size: 0.015625 normalized = ~10 pixels at 640x640
- YOLO struggles with objects < 16px
- Trees are densely packed (2000+ per frame)

### 3. HIGH DENSITY OVERLAP
- Trees overlap visually at small size
- Hard for model to distinguish individual trees

### 4. TRAINING SETTINGS
- CPU only training (slow iteration)
- Low batch size (4)
- Limited epochs shown (42/300)

## Recommendations for HIGHER ACCURACY:

### A. Dataset Improvements
1. **Increase tile count** - Use more overlapping tiles from source imagery
   - Current: 640x640 tiles with 20% overlap
   - Suggestion: 50% overlap to generate more training samples

2. **Use larger image size** - Train at 1280 instead of 640
   - This makes 10px trees appear as 20px (better for YOLO)
   - trade-off: 4x slower training, need smaller batch

3. **Add more diversity** - Different plantation blocks, lighting conditions

### B. Training Improvements
1. **Use yolo11s or yolo11m** instead of yolo11n
   - More capacity to learn small object patterns

2. **Increase epochs** - 300+ for small dataset
   - But save checkpoint more frequently (every 10 epochs)

3. **Use mosaic=1.0, copy_paste=0.2** - Critical for small objects
   - copy_paste especially helps isolated small objects

4. **Lower conf_thres during training augmentation**

### C. Small Object Specific (Advanced)
1. **Use YOLO's small object layer** if available
2. **Tiling at higher resolution** - 1280 or 1920
3. **Two-stage detection** - Coarse then fine

# =============================================================================
# RECOMMENDED CONFIG CHANGES
# =============================================================================

## Option 1: Moderate Improvement
- Model: yolo11s (small, not nano)
- img_size: 1280
- batch: 2 (half size due to memory)
- epochs: 300
- copy_paste: 0.2
- mosaic: 1.0

## Option 2: Aggressive for Max Accuracy
- Model: yolo11m (medium)
- img_size: 1280
- batch: 1
- epochs: 500
- copy_paste: 0.3
- mosaic: 1.0
- patience: 100 (wait longer for improvement)

# =============================================================================
# COMMAND TO TRAIN WITH HIGHER ACCURACY
# =============================================================================

# Run this instead:
python Train/train.py --model yolo11s --img-size 1280 --epochs 300 --batch 2 --copy-paste 0.2

# For best results (slower):
python Train/train.py --model yolo11m --img-size 1280 --epochs 500 --batch 1 --copy-paste 0.3

# =============================================================================
# FILE: _dev_utils/debug_training.py - Auto-generate this report
# =============================================================================
