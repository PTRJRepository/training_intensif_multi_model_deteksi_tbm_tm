@echo off
chcp 65001 >nul
echo.
echo ================================================
echo  MULTI-MODEL TRAINING
echo ================================================
echo.
echo Script ini melatih multiple model untuk ensemble.
echo Model yang berbeda akan mendeteksi pola berbeda.
echo.
echo Contoh kombinasi:
echo   - yolo11n + yolo11s
echo   - yolov8n + yolov8s
echo   - yolo11n + yolov8n
echo.

cd /d "%~dp0..\"

:: Model 1
set /p MODEL1="Model 1 (default: yolo11n): "
if "%MODEL1%"=="" set MODEL1=yolo11n

:: Model 2
set /p MODEL2="Model 2 (default: yolov8n): "
if "%MODEL2%"=="" set MODEL2=yolov8n

:: Epochs
set /p EPOCHS="Epochs (default: 100): "
if "%EPOCHS%"=="" set EPOCHS=100

echo.
echo ================================================
echo  TRAINING MODELS FOR ENSEMBLE
echo ================================================
echo.
echo [MODEL 1] %MODEL1%
echo.
python Train/train.py --model %MODEL1% --epochs %EPOCHS% --batch 8 --name ensemble_%MODEL1%_exp1

echo.
echo [MODEL 2] %MODEL2%
echo.
python Train/train.py --model %MODEL2% --epochs %EPOCHS% --batch 8 --name ensemble_%MODEL2%_exp2

echo.
echo ================================================
echo  ENSEMBLE TRAINING COMPLETE
echo ================================================
echo.
echo Models tersimpan di:
echo   Train\runs\ensemble_%MODEL1%_exp1\weights\best.pt
echo   Train\runs\ensemble_%MODEL2%_exp2\weights\best.pt
echo.
echo Untuk ensemble inference, jalankan:
echo   03_inference_ensemble.bat
echo.
pause
