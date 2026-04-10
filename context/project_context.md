# 🌴 Palm Tree Detection Project - Context

## 📋 **Project Overview**
Deteksi dan penghitungan pohon sawit dari citra drone menggunakan deep learning (YOLO).

---

## 📂 **Data yang Tersedia**

### **Image 1: Dataset_1A_Atha.tif**
- **Lokasi:** `Dataset/Raw Dataset/TBM/Processed/Clipped/Dataset_1A_Atha.tif`
- **Ukuran:** 2622 x 2137 pixels (4 bands)
- **CRS:** EPSG:32748
- **Resolution:** 0.37m/pixel
- **Shapefile annotations:** `TBM_1A_Kundo.shp`
  - Tipe: **Point** geometry
  - Jumlah: **2,237 trees**

### **Image 2: tbm_1a_isnin.tif**
- **Lokasi:** `Dataset/Raw Dataset/TBM/Processed/Clipped/tbm_1a_isnin.tif`
- **Ukuran:** 1615 x 2523 pixels (4 bands)
- **CRS:** EPSG:32748
- **Resolution:** 0.37m/pixel
- **Shapefile annotations:** `TBM/TBM.shp`
  - Tipe: **MultiPoint** geometry
  - Jumlah: **2,761 trees**

### **Image 3 (Original): Sub Divisi Air Kundo 2025.tif**
- **Lokasi:** `../Training Tree Counter Sawit Current/CITRA DRONE DATASET BARU 2025/Divisi 1 A/`
- **Ukuran:** 15000 x 5729 pixels
- **Shapefile:** `TBM_1A_Kundo.shp` (sudah di-clip ke `Potong_layer_1A.shp`)

---

## 🎯 **Goal & Rencana**

### **Tujuan Utama:**
1. **Deteksi otomatis** pohon sawit dari citra drone
2. **Counting/penghitungan** jumlah pohon per area/plantation block
3. Model yang **generalize** ke berbagai area (bukan hanya satu lokasi)

### **Challenge:**
- Objek pohon **sangat kecil** (~10-20 pixels diameter)
- Satu frame **banyak sekali pohon** (2000+)
- Variasi ukuran pohon di area berbeda
- Background kompleks (gulma, jalan sawit, area kosong)

---

## 🛠️ **Strategi yang Sudah Dilakukan**

### **1. Data Preparation**
✅ Script untuk clip TIFF berdasarkan polygon:
- `Tools/clip_raster_by_polygon.py`
- Fungsi: Memotong raster sesuai area polygon dari shapefile

### **2. Dataset Creation untuk Training**
✅ Script untuk prepare dataset YOLO format:
- `Tools/prepare_multi_image_dataset.py`
- Multi-image support (combine dari beberapa area)
- Handle berbagai tipe geometry: Point, MultiPoint, Polygon
- Strategi: **Tiled chips** (640x640 dengan overlap 20%)

### **3. Dataset yang Sudah Dibuat**
✅ **Lokasi:** `Dataset/dataset_pakai_ini/`

**Struktur:**
```
dataset_pakai_ini/
├── images/
│   ├── train/     (14 tiles)
│   └── val/       (3 tiles)
├── labels/
│   ├── train/     (YOLO format labels)
│   └── val/
└── data.yaml      (YOLO config)
```

**Statistik:**
- **Total tiles:** 17
- **Total trees annotated:** 4,998
  - Train: 14 tiles
  - Val: 3 tiles
- **Class name:** `TBM` (class 0)

---

## 🤔 **Pertanyaan yang Perlu Dijawab**

### **Training Strategy:**
- ❓ **Training per frame** atau **per tile/chip**?
  - ✅ **Sudah diputuskan: TILED/CHIP approach** (640x640)
  - Alasan: Small object detection butuh detail, context tetap terjaga
  
- ❓ **Arsitektur model terbaik?**
  - Opsi: YOLOv8n, YOLOv8s, YOLOv11n, YOLOv11s
  - Rekomendasi: **YOLOv8n atau YOLOv11n** (fast, lightweight, cukup untuk small objects)

### **Data Augmentation:**
- ❓ Perlu augmentasi lebih lanjut?
  - Multi-scale training (sudah ada di script tapi belum diaktifkan)
  - Rotation, flip, brightness variation
  - Mosaic augmentation (built-in YOLO)

### **Annotation Quality:**
- ❓ Apakah titik di shapefile **sudah tepat di center pohon**?
  - Jika tidak, bounding box estimation bisa meleset
  - Perlu sampling manual untuk verifikasi

---

## 📝 **Next Steps yang Direncanakan**

### **1. Model Training** 🔜
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='Dataset/dataset_pakai_ini/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='palm_tree_detection'
)
```

**Hyperparameters yang direkomendasikan:**
- `epochs`: 100-200
- `batch`: 16 (sesuaikan dengan VRAM)
- `imgsz`: 640
- `lr0`: 0.01 (default)
- **Important:** Enable mosaic augmentation untuk small objects

### **2. Model Evaluation** 🔜
- Test pada area yang belum di-train
- Metrics: Precision, Recall, mAP@50, mAP@50-95
- Analisis: False positives (deteksi bukan pohon) & False negatives (pohon terlewat)

### **3. Inference/Prediction** 🔜
- Apply model ke area baru (misal: Divisi lain)
- Counting otomatis per polygon block
- Export hasil ke shapefile/CSV dengan koordinat pohon

### **4. Deployment** 🔜
- Script untuk batch processing banyak area
- Output: 
  - Count per block
  - Shapefile dengan lokasi pohon terdeteksi
  - Visualisasi (gambar dengan bounding boxes)

---

## 💡 **Insights & Lessons Learned**

### **Yang Sudah Dipelajari:**
1. ✅ Shapefile bisa berisi **Point, MultiPoint, atau Polygon** → perlu handle berbeda
2. ✅ CRS harus **match** antara image dan shapefile
3. ✅ Small objects butuh **tile-based approach**, bukan full frame
4. ✅ Overlap antar tile penting (20% default) agar pohon di edge tidak terpotong
5. ✅ Bounding box size estimation untuk point annotations: ~10px (perlu tuning)

### **Yang Perlu Diperbaiki:**
- ⚠️ Dataset masih **kecil** (17 tiles) → perlu lebih banyak variasi area
- ⚠️ Bounding box size **hardcoded** (10px) → perlu estimasi lebih akurat
- ⚠️ Belum ada **visual verification** → perlu cek apakah tiles & labels benar
- ⚠️ Belum test **training** → belum tahu akurasi model

---

## 🔧 **Tools yang Sudah Dibuat**

| File | Fungsi |
|------|--------|
| `Tools/clip_raster_by_polygon.py` | Clip TIFF berdasarkan polygon shapefile |
| `Tools/prepare_multi_image_dataset.py` | Create YOLO dataset dari multiple images |
| `Tools/debug_shapefile.py` | Debug CRS dan geometry issues |

---

## 📊 **File Paths Reference**

```
Project Root: D:\Gawean Rebinmas\Tree Counting Project\Training Deteksi Sawit Multi Model

Dataset:
  ├── Raw Dataset/TBM/
  │   ├── Processed/Clipped/
  │   │   ├── Dataset_1A_Atha.tif
  │   │   └── tbm_1a_isnin.tif
  │   ├── TBM_1A_Kundo.shp (Point, 2237 trees)
  │   └── TBM/TBM.shp (MultiPoint, 2761 trees)
  └── dataset_pakai_ini/ (YOLO training dataset)
       ├── images/train & val
       ├── labels/train & val
       └── data.yaml

Tools:
  ├── clip_raster_by_polygon.py
  ├── prepare_multi_image_dataset.py
  └── debug_shapefile.py
```

---

## ❓ **Pertanyaan untuk User**

1. **Apakah titik di shapefile benar-benar di center pohon?**
   - Atau perlu offset adjustment?

2. **Berapa perkiraan ukuran pohon dalam pixels?**
   - Untuk set bounding box size lebih akurat (sekarang hardcoded 10px)

3. **Ada berapa total area/divisi yang akan diproses?**
   - Untuk planning dataset yang lebih representative

4. **Target akurasi berapa %?**
   - 80%? 90%? 95%?
   - Ini affect choice model size & training epochs

5. **Perlu real-time detection atau batch processing ok?**
   - Real-time butuh model lebih kecil (nano)
   - Batch bisa pakai model lebih besar (small/medium)

---

**Last Updated:** April 2026
