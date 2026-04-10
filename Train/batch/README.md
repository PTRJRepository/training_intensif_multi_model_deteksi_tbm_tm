# Batch Scripts - TBM Detection Training

## Cara Penggunaan

### Method 1: Menu Interaktif
Jalankan `00_MAIN_RUNNER.bat` untuk menu interaktif.

### Method 2: Jalankan Langsung
Jalankan batch file sesuai kebutuhan.

---

## Daftar Batch Files

### Basic Pipeline
| File | Fungsi |
|------|--------|
| `00_MAIN_RUNNER.bat` | **Menu utama** - pilih tahap yang ingin dijalankan |
| `00_analyze_dataset.bat` | Analisis dataset (distribusi, density) |
| `01_train_start.bat` | Training model default (yolo11n, 150 epochs) |
| `02_evaluate.bat` | Evaluasi model |
| `03_inference.bat` | Inference pada validation images |

### Advanced
| File | Fungsi |
|------|--------|
| `01_train_custom.bat` | Custom training dengan parameter sendiri |
| `02_train_ensemble.bat` | Training multiple models untuk ensemble |
| `03_inference_ensemble.bat` | Ensemble inference (multiple models) |

### Utility
| File | Fungsi |
|------|--------|
| `99_QUICK_START.bat` | Training langsung tanpa konfirmasi |
| `98_cleanup.bat` | Hapus semua hasil training |

---

## Quick Start

```batch
# 1. Jalankan menu
00_MAIN_RUNNER.bat

# 2. Pilih:
#    [1] Analisis Dataset
#    [2] Training Model
#    [3] Evaluasi
#    [4] Inference
#    [5] Jalankan semua tahap
```

## Advanced Usage

### Custom Training
```batch
01_train_custom.bat
# Masukkan: model, epochs, batch, lr, experiment name
```

### Multi-Model Ensemble
```batch
# 1. Train multiple models
02_train_ensemble.bat
# Masukkan: model1, model2, epochs

# 2. Ensemble inference
03_inference_ensemble.bat
# Gabungkan predictions dari 2 model
```

## Output Locations

| Type | Lokasi |
|------|--------|
| Model weights | `Train/runs/<exp>/weights/best.pt` |
| Training logs | `Train/runs/<exp>/results.csv` |
| Inference results | `Train/runs/inference_results/` |
| Ensemble results | `Train/runs/ensemble_results/` |

---

## Notes

- Semua batch menggunakan **CPU**
- Konfirmasi sebelum hapus/overwrite
- Training output tersimpan di `Train/runs/`
