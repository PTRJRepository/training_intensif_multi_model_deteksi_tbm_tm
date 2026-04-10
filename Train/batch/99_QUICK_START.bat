@echo off
chcp 65001 >nul
echo.
echo ================================================
echo  QUICK START - TRAINING TBM
echo ================================================
echo.
echo Training langsung dimulai dengan setting default:
echo   Model: yolo11n
echo   Epochs: 150
echo   Batch: 8
echo   Device: CPU
echo.

cd /d "%~dp0..\"

:: Check if dataset exists
if not exist "Dataset\dataset_pakai_ini\data.yaml" (
    echo.
    echo [ERROR] Dataset tidak ditemukan!
    echo Jalankan preparation script terlebih dahulu:
    echo   python Tools\prepare_multi_image_dataset.py
    echo.
    pause
    exit /b 1
)

echo [OK] Dataset ditemukan
echo.
echo Memulai training...
echo.

python Train/train.py --model yolo11n --epochs 150 --batch 8 --name quickstart_exp

echo.
echo ================================================
echo  TRAINING COMPLETE
echo ================================================
echo.
echo Model tersimpan di:
echo   Train\runs\quickstart_exp\weights\best.pt
echo.
pause
