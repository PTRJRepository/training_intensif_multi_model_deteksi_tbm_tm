@echo off
chcp 65001 >nul
echo.
echo ================================================
echo  CUSTOM TRAINING - TBM DETECTION
echo ================================================
echo.

cd /d "%~dp0..\"

:: Default values
set MODEL=yolo11n
set EPOCHS=150
set BATCH=8
set LR=0.01
set IMG_SIZE=640
set NAME=exp1_yolo11n_640

echo Konfigurasi default:
echo   Model   : %MODEL%
echo   Epochs  : %EPOCHS%
echo   Batch   : %BATCH%
echo   LR      : %LR%
echo   ImgSize : %IMG_SIZE%
echo.

echo Tekan ENTER untuk menggunakan default,
echo atau masukkan nilai baru:
echo.

set /p custom="Model (yolo11n/yolo11s/yolov8n/yolov8s) [%MODEL%]: "
if not "%custom%"=="" set MODEL=%custom%

set /p custom="Epochs [%EPOCHS%]: "
if not "%custom%"=="" set EPOCHS=%custom%

set /p custom="Batch Size [%BATCH%]: "
if not "%custom%"=="" set BATCH=%custom%

set /p custom="Learning Rate [%LR%]: "
if not "%custom%"=="" set LR=%custom%

set /p custom="Experiment Name [%NAME%]: "
if not "%custom%"=="" set NAME=%custom%

echo.
echo ================================================
echo  STARTING CUSTOM TRAINING
echo ================================================
echo  Model: %MODEL%
echo  Epochs: %EPOCHS%
echo  Batch: %BATCH%
echo  LR: %LR%
echo  Name: %NAME%
echo ================================================
echo.

python Train/train.py --model %MODEL% --epochs %EPOCHS% --batch %BATCH% --lr %LR% --name %NAME%

echo.
echo ================================================
echo  TRAINING COMPLETE
echo ================================================
pause
