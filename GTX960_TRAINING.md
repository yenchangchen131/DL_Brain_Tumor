# GTX 960 訓練注意事項

## GPU 限制說明

您的系統配置：
- **GPU**: NVIDIA GeForce GTX 960
- **顯存**: 4GB
- **用途**: 同時用於顯示和計算
- **影像大小**: 640x640 像素

由於 GPU 同時用於顯示桌面，Windows 有一個 **TDR (Timeout Detection and Recovery)** 機制：
- 如果 GPU 操作超過約 2 秒沒有回應，Windows 會強制終止
- 這是為了防止畫面凍結

---

## 已優化的設定

為了避免 CUDA timeout 錯誤，我們已經優化了以下參數：

### 1. Batch Size: 2
```python
BATCH_SIZE = 2
```

**為什麼這麼小？**
- 640x640 的影像 × 3 通道 = 每張圖約 1.2MB
- U-Net 模型的中間特徵圖會更大（特徵通道數增加）
- Batch size 8 會超過 4GB 顯存限制
- Batch size 2 是安全且穩定的選擇

**影響**：
- ✅ 穩定性提高，不會 timeout
- ✅ 仍然可以正常訓練
- ⚠️ 每個 epoch 會稍微慢一點（但仍比 CPU 快很多）
- ⚠️ 梯度更新頻率增加（可能需要調整學習率）

---

### 2. Num Workers: 0
```python
NUM_WORKERS = 0
```
Windows 多進程相容性

---

## 訓練預期時間

使用 Batch Size = 2：
- **每個 epoch**: 約 10-15 分鐘
- **總訓練時間**: 16-25 小時（100 epochs，但通常會提前 Early Stop）

---

## 如果還是遇到 CUDA timeout

### 方法 1: 進一步降低 Batch Size（推薦）
在 `train.py` 中修改：
```python
BATCH_SIZE = 1  # 最保守的設定
```

### 方法 2: 減少影像大小
在 `brain_tumor_segmentation.py` 中修改：
```python
class TrainTransform:
    def __init__(self, image_size=512):  # 從 640 改為 512
        # ...
```

### 方法 3: 訓練時關閉其他應用程式
- 關閉瀏覽器（特別是硬體加速的）
- 關閉影片播放器
- 最小化其他程式

### 方法 4: 使用更簡單的模型（不推薦）
減少 U-Net 的 features，但會降低模型性能。

---

## 記憶體使用監控

訓練時可以開另一個終端機執行：
```bash
nvidia-smi -l 1
```

這會每秒更新 GPU 使用狀況，確保顯存不會溢出。

---

## 最佳實踐

1. **訓練前**：
   - 關閉不必要的應用程式
   - 確保只有基本的系統程式在運行

2. **訓練中**：
   - 不要操作需要大量 GPU 的應用（影片、3D 軟體等）
   - 可以使用瀏覽器瀏覽網頁（不要開太多分頁）

3. **監控**：
   - 定期檢查訓練進度
   - 確認模型有在儲存（`models/best_model.pth`）

---

## 當前最終設定

```python
BATCH_SIZE = 2       # 適合 GTX 960 4GB
NUM_EPOCHS = 100     # 最多訓練 100 輪
LEARNING_RATE = 1e-4 # 標準學習率
PATIENCE = 15        # Early stopping
NUM_WORKERS = 0      # Windows 相容性
```

這些設定已經在 `train.py` 中設置好了，**直接執行即可**！

---

## 執行訓練

```bash
python train.py
```

如果成功開始訓練，您會看到：
```
使用裝置: cuda
載入資料...
訓練集樣本數: 1504
驗證集樣本數: 214
測試集樣本數: 75

開始訓練...
Epoch 1/100
Training: 0%|          | 0/752 [00:00<?, ?it/s]
```

注意 batch size 改為 2 後，每個 epoch 的 iterations 會從 188 變成 **752**（1504/2）。

---

**祝訓練順利！** 🚀
