# Windows 系統訓練設定說明

## 問題解決

為了在 Windows 系統上順利訓練，我們進行了以下修改：

### 1. DataLoader 多進程問題
**問題**: Windows 的多進程支援與 Linux 不同，使用 `num_workers > 0` 會導致 worker 進程崩潰。

**解決方案**: 將 `NUM_WORKERS` 設為 0
```python
NUM_WORKERS = 0  # Windows 系統建議設為 0，避免多進程問題
```

**影響**: 
- ✅ 訓練穩定性提高
- ⚠️ 資料載入速度稍慢（但使用 GPU 後整體速度仍然很快）

---

### 2. OpenMP Runtime 衝突
**問題**: 多個套件（NumPy、PyTorch 等）都包含自己的 OpenMP runtime，導致初始化衝突。

**解決方案**: 在程式開頭設置環境變數
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

**影響**:
- ✅ 避免 OMP Error #15
- ⚠️ 理論上可能影響性能，但實際影響微乎其微

---

## 如何執行訓練

### 方法 1: 直接執行 Python 腳本（推薦）
```bash
python train.py
```

所有環境變數已在腳本中設置好，直接執行即可。

### 方法 2: 使用批次檔案
```bash
run_train.bat
```

這個批次檔案會：
1. 設置環境變數
2. 執行訓練腳本
3. 訓練完成後暫停以查看結果

---

## GPU 加速確認

訓練開始時應該會看到：
```
使用裝置: cuda
```

如果看到 `cpu`，請執行 `check_gpu.py` 檢查 GPU 設定。

---

## 訓練參數

當前設定（可在 `train.py` 中修改）：
- `BATCH_SIZE`: 8
- `NUM_EPOCHS`: 100
- `LEARNING_RATE`: 1e-4
- `PATIENCE`: 15（Early Stopping）
- `NUM_WORKERS`: 0（Windows 相容性）

---

## 預期訓練時間

使用 NVIDIA GeForce GTX 960 (4GB)：
- 每個 epoch: 約 5-10 分鐘
- 總訓練時間: 8-16 小時（取決於 Early Stopping）

建議：
- 讓訓練在夜間執行
- 定期檢查 `models/best_model.pth` 確保模型有被儲存
- 可以隨時按 Ctrl+C 終止訓練，已儲存的最佳模型不會丟失

---

## 訓練輸出

訓練過程中會即時顯示：
- Loss 和 Dice Score
- 進度條
- 最佳模型儲存提示

訓練完成後會產生：
- `models/best_model.pth` - 最佳模型權重
- `results/training_history.json` - 訓練歷史紀錄
- `results/training_curves.png` - 訓練曲線圖
- `results/test_metrics.json` - 測試集評估結果
- `results/predictions.png` - 預測結果視覺化

---

## 疑難排解

### 如果遇到記憶體不足錯誤
降低 batch size:
```python
BATCH_SIZE = 4  # 或更小
```

### 如果訓練太慢
這是正常的！深度學習訓練本來就需要時間。使用 GPU 已經大幅加速了。

### 如果想提前停止
按 `Ctrl+C` 即可，最佳模型已經儲存在 `models/best_model.pth`

---

## 已修改的檔案

1. `train.py`:
   - 添加 OpenMP 環境變數設置
   - NUM_WORKERS 改為 0

2. 新增檔案:
   - `run_train.bat` - Windows 批次執行檔
   - `check_gpu.py` - GPU 診斷工具
   - `WINDOWS_SETUP.md` - 本說明檔案

所有修改都是為了 Windows 相容性，不影響訓練效果！
