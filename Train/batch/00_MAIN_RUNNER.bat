@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo #############################################################
echo #
echo #     TBM DETECTION - TRAINING PIPELINE
echo #     Multi-Model Inference Ready
echo #
echo #############################################################
echo.

:: Get script directory
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%\.."

:: Main menu
:menu
echo.
echo ================================================
echo  MENU TRAINING TBM
echo ================================================
echo.
echo  [1] Analisis Dataset
echo  [2] Training Model
echo  [3] Evaluasi Model
echo  [4] Inference (Uji Model)
echo  [5] Jalankan SEMUA tahap (1^>2^>3^>4)
echo.
echo  [0] Keluar
echo.
echo ================================================
echo.

set /p choice="Pilih menu: "

if "%choice%"=="1" goto analyze
if "%choice%"=="2" goto train
if "%choice%"=="3" goto evaluate
if "%choice%"=="4" goto inference
if "%choice%"=="5" goto full_pipeline
if "%choice%"=="0" goto end

echo [ERROR] Pilihan tidak valid!
goto menu

:analyze
echo.
echo ================================================
echo  TAHAP 1: ANALISIS DATASET
echo ================================================
echo.
call "%SCRIPT_DIR%00_analyze_dataset.bat"
goto menu

:train
echo.
echo ================================================
echo  TAHAP 2: TRAINING MODEL
echo ================================================
echo.
call "%SCRIPT_DIR%01_train_start.bat"
goto menu

:evaluate
echo.
echo ================================================
echo  TAHAP 3: EVALUASI MODEL
echo ================================================
echo.
call "%SCRIPT_DIR%02_evaluate.bat"
goto menu

:inference
echo.
echo ================================================
echo  TAHAP 4: INFERENCE
echo ================================================
echo.
call "%SCRIPT_DIR%03_inference.bat"
goto menu

:full_pipeline
echo.
echo #############################################################
echo #
echo #  MENJALANKAN SEMUA TAHAP
echo #  1. Analisis ^> 2. Training ^> 3. Evaluasi ^> 4. Inference
echo #
echo #############################################################
echo.

echo [TAHAP 1/4] ANALISIS DATASET
echo ================================================
call "%SCRIPT_DIR%00_analyze_dataset.bat"

echo.
echo [TAHAP 2/4] TRAINING MODEL
echo ================================================
call "%SCRIPT_DIR%01_train_start.bat"

echo.
echo [TAHAP 3/4] EVALUASI MODEL
echo ================================================
call "%SCRIPT_DIR%02_evaluate.bat"

echo.
echo [TAHAP 4/4] INFERENCE
echo ================================================
call "%SCRIPT_DIR%03_inference.bat"

echo.
echo #############################################################
echo #
echo #  PIPELINE SELESAI!
echo #
echo #  Output locations:
echo #    Model  : Train\runs\<exp>\weights\best.pt
echo #    Results: Train\runs\inference_results
echo #
echo #############################################################
pause

:end
echo.
echo Terima kasih!
exit /b 0
