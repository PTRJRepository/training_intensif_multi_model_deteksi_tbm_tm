@echo off
chcp 65001 >nul
echo.
echo ================================================
echo  INFERENCE TBM - UJI MODEL
echo ================================================
echo.

cd /d "%~dp0..\"

:: Default test image - cari di dataset val
set TEST_IMAGE=Dataset\dataset_pakai_ini\images\val

:: Check if best.pt exists
if exist "Train\runs\exp1_yolo11n_640\weights\best.pt" (
    echo Menggunakan model: Train\runs\exp1_yolo11n_640\weights\best.pt
    set WEIGHT=Train\runs\exp1_yolo11n_640\weights\best.pt
) else (
    echo [WARNING] best.pt tidak ditemukan, mencari model terbaru...
    for /f "delims=" %%f in ('dir /s /b /o-n "Train\runs\*\weights\best.pt" 2^>nul') do (
        set WEIGHT=%%f
        goto :found
    )
)

:found
if not defined WEIGHT (
    echo [ERROR] Tidak ada model ditemukan!
    echo Jalankan training terlebih dahulu.
    pause
    exit /b 1
)

echo Model: %WEIGHT%
echo.

:: Run inference on validation images with LOW conf for high recall
echo Menjalankan inference pada validation images...
echo (conf=0.01 untuk high recall)
echo.

python Train/inference.py --weights "%WEIGHT%" --source "%TEST_IMAGE%" --output Train\runs\inference_results --conf 0.01 --save-txt --save-conf

echo.
echo ================================================
echo  INFERENCE SELESAI
echo ================================================
echo Hasil tersimpan di: Train\runs\inference_results
echo.
pause
