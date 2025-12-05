@echo off
REM 設置環境變數以解決 OpenMP 衝突
set KMP_DUPLICATE_LIB_OK=TRUE

echo ========================================
echo 開始訓練腦腫瘤分割模型
echo ========================================
echo.
echo 環境設定:
echo - KMP_DUPLICATE_LIB_OK=TRUE (解決 OpenMP 衝突)
echo - NUM_WORKERS=0 (Windows 相容性)
echo.

REM 執行訓練
C:\Users\Kslab\miniconda3\envs\brain_tumor\python.exe train.py

echo.
echo ========================================
echo 訓練完成或已中斷
echo ========================================
pause
