@echo off
chcp 65001 >nul
echo.
echo ================================================
echo  ANALISIS DATASET TBM
echo ================================================
echo.
echo Menjalankan analisis dataset untuk melihat:
echo   - Distribusi tree per tile
echo   - Density analysis
echo   - Rekomendasi augmentasi
echo.

cd /d "%~dp0..\"
python Train/analyze_dataset.py

echo.
echo ================================================
echo  ANALISIS SELESAI
echo ================================================
echo Tekan任意按键退出...
pause >nul
