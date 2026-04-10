@echo off
chcp 65001 >nul
echo.
echo ================================================
echo  EVALUASI MODEL TBM
echo ================================================
echo.

cd /d "%~dp0..\"

:: Check if best.pt exists
if exist "Train\runs\exp1_yolo11n_640\weights\best.pt" (
    echo Ditemukan best.pt, menjalankan evaluasi...
    echo.
    python Train/evaluate.py --weights "Train\runs\exp1_yolo11n_640\weights\best.pt" --save-json
) else (
    echo.
    echo [WARNING] best.pt tidak ditemukan.
    echo Pastikan training sudah selesai terlebih dahulu.
    echo.
    echo Mencari model terbaru...
    for /f "delims=" %%f in ('dir /s /b /o-n "Train\runs\*\weights\best.pt" 2^>nul') do (
        echo Ditemukan: %%f
        set LATEST_WEIGHT=%%f
    )
    if defined LATEST_WEIGHT (
        echo.
        echo Menjalankan evaluasi dengan model terbaru...
        python Train\evaluate.py --weights "%LATEST_WEIGHT%" --save-json
    ) else (
        echo.
        echo [ERROR] Tidak ada model yang ditemukan.
        echo Jalankan 01_train_start.bat terlebih dahulu.
    )
)

echo.
echo ================================================
echo  EVALUASI SELESAI
echo ================================================
pause
