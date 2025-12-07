@echo off
echo ========================================
echo   啟動修改後的訓練 Notebook
echo   (已解決 CUDA Timeout 問題)
echo ========================================
echo.
echo 正在啟動 Jupyter Notebook...
echo.
echo 將打開: brain_tumor_complete_size448.ipynb
echo 圖像大小: 448x448
echo 速度提升: ~2 倍
echo.
echo 下一步:
echo   1. 在瀏覽器中運行所有 cells (Cell -^> Run All)
echo   2. 開始訓練！
echo.
pause
jupyter notebook brain_tumor_complete_size448.ipynb
