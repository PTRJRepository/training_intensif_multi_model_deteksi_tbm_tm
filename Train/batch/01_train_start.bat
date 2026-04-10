@echo off
chcp 65001 >nul
echo.
echo ================================================
echo  TRAINING MODEL TBM - START
echo ================================================
echo.
echo Konfigurasi:
echo   Model    : yolo11n (nano - cepat, ringan)
echo   Epochs   : 150
echo   Batch    : 8
echo   Device   : CPU
echo   Img Size : 640
echo.

cd /d "%~dp0..\"
python Train/train.py --model yolo11n --epochs 150 --batch 8

echo.
echo ================================================
echo  TRAINING SELESAI
echo ================================================
echo.
echo Model tersimpan di:
echo   Train\runs\<experiment>\weights\best.pt
echo   Train\runs\<experiment>\weights\last.pt
echo.
echo Langkah selanjutnya:
echo   02_evaluate.bat - Evaluasi model
echo   03_inference.bat - Uji inference
echo.
pause
