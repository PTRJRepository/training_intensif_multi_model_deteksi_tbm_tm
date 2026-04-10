@echo off
chcp 65001 >nul
echo.
echo ================================================
echo  ENSEMBLE INFERENCE
echo ================================================
echo.
echo Melakukan inference dengan multiple models
echo dan menggabungkan predictions.
echo.

cd /d "%~dp0..\"

:: Find all best.pt files
echo Mencari model yang tersedia...
echo.

set MODEL1=
set MODEL2=

for /f "delims=" %%f in ('dir /s /b /o-n "Train\runs\*\weights\best.pt" 2^>nul') do (
    if not defined MODEL1 (
        set MODEL1=%%f
    ) else if not defined MODEL2 (
        set MODEL2=%%f
    )
)

if not defined MODEL1 (
    echo [ERROR] Tidak ada model ditemukan!
    echo Jalankan training terlebih dahulu.
    pause
    exit /b 1
)

if not defined MODEL2 (
    echo [WARNING] Hanya 1 model ditemukan: %MODEL1%
    echo Using single model inference instead.
    set /p confirm="Lanjutkan dengan single model? (y/n): "
    if /i not "%confirm%"=="y" exit /b 1
    set WEIGHTS=%MODEL1%
    goto :run_inference
)

echo Model 1: %MODEL1%
echo Model 2: %MODEL2%
echo.

:: Source
set /p SOURCE="Source folder (default: Dataset\dataset_pakai_ini\images\val): "
if "%SOURCE%"=="" set SOURCE=Dataset\dataset_pakai_ini\images\val

:: Combine weights
set WEIGHTS=%MODEL1%,%MODEL2%

:run_inference
echo.
echo ================================================
echo  STARTING ENSEMBLE INFERENCE
echo ================================================
echo Weights: %WEIGHTS%
echo Source: %SOURCE%
echo.

python Train/inference.py --weights "%WEIGHTS%" --source "%SOURCE%" --output Train\runs\ensemble_results --conf 0.01 --save-txt --save-conf --augment

echo.
echo ================================================
echo  ENSEMBLE INFERENCE COMPLETE
echo ================================================
echo Output: Train\runs\ensemble_results
pause
