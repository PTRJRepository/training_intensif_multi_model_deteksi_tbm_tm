@echo off
chcp 65001 >nul
echo.
echo ================================================
echo  CLEANUP - Hapus Training Results
echo ================================================
echo.

cd /d "%~dp0..\"

echo Peringatan: Ini akan menghapus folder Train\runs
echo dan semua model hasil training.
echo.

set /p confirm="Lanjutkan? (y/n): "
if /i not "%confirm%"=="y" (
    echo Dibatalkan.
    pause
    exit /b 0
)

echo.
echo Menghapus folder runs...
if exist "Train\runs" (
    rd /s /q "Train\runs"
    echo [OK] Folder runs dihapus
) else (
    echo [INFO] Folder runs tidak ditemukan
)

echo.
echo ================================================
echo  CLEANUP COMPLETE
echo ================================================
pause
