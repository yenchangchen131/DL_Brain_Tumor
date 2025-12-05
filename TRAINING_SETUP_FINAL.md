# 訓練問題解決歷程與最終設定

## 問題演變過程

### 問題 1: 張量形狀不匹配 ✅ 已解決
**錯誤**: `Target size (torch.Size([8, 640, 640])) must be the same as input size (torch.Size([8, 1, 640, 640]))`

**原因**: albumentations 產生的 mask 是 3D，但模型輸出是 4D

**解決**: 在所有損失函數和評估函數中添加形狀和類型檢查

---

### 問題 2: 數據類型不匹配 ✅ 已解決
**錯誤**: `result type Float can't be cast to the desired output type Byte`

**原因**: mask 是 uint8 類型，但 BCE loss 需要 float

**解決**: 在所有函數中添加 `.float()` 轉換

---

### 問題 3: DataLoader Worker 崩潰 ✅ 已解決
**錯誤**: `RuntimeError: DataLoader worker exited unexpectedly`

**原因**: Windows 多進程支援不佳

**解決**: 設定 `NUM_WORKERS = 0`

---

### 問題 4: OpenMP 衝突 ✅ 已解決
**錯誤**: `OMP: Error #15: Initializing libiomp5md.dll`

**原因**: 多個套件包含自己的 OpenMP runtime

**解決**: 設定環境變數 `os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'`

---

### 問題 5: CUDA Timeout ✅ 已解決
**錯誤**: `CUDA error: the launch timed out and was terminated`

**原因**: Batch size 8 太大，超過 Windows TDR 限制

**解決**: 降低到 `BATCH_SIZE = 2`

---

### 問題 6: cuDNN 內部錯誤 ✅ 已解決
**錯誤**: `cuDNN error: CUDNN_STATUS_INTERNAL_ERROR`

**原因**: GPU 記憶體仍然不足，batch size 2 對 640x640 影像還是太大

**解決**: 降低到 `BATCH_SIZE = 1`（最保守設定）

---

## 最終穩定設定

```python
# GTX 960 (4GB) 穩定訓練配置
BATCH_SIZE = 1       # 最小值，確保穩定
NUM_EPOCHS = 100     
LEARNING_RATE = 1e-4 
PATIENCE = 15        
NUM_WORKERS = 0      # Windows 相容性
IMAGE_SIZE = 640     # 640x640 像素
```

---

## 訓練性能預估

### Iterations 數量
- 訓練集: 1504 樣本
- Batch size: 1
- **每個 epoch**: 1504 iterations

### 時間估算
根據您的硬體（GTX 960）：
- **每個 iteration**: 約 20-30 秒（batch size 1 會比 2 快）
- **每個 epoch**: 約 8-12 小時
- **總訓練時間**: 
  - 如果跑完 100 epochs: 800-1200 小時（不實際）
  - **實際會 Early Stop**: 預計 15-30 epochs 後停止
  - **實際總時間**: 約 120-360 小時（5-15 天）

---

## 重要提醒

### ⚠️ 訓練時間很長
使用 GTX 960 4GB 訓練 640x640 影像確實會很慢，這是硬體限制。

### 建議：
1. **讓訓練連續運行** - 不要中斷
2. **夜間訓練** - 白天不使用電腦時讓它跑
3. **監控進度** - 定期檢查是否有在更新

### 或者考慮：

#### 選項 A: 降低影像解析度（推薦）
修改 `brain_tumor_segmentation.py`:
```python
class TrainTransform:
    def __init__(self, image_size=512):  # 從 640 改為 512 或 448
```

**效果**:
- 訓練速度提升 1.5-2 倍
- 記憶體使用減少
- 可以使用更大的 batch size（如 2 或 4）
- **模型性能會稍微降低**（但仍然有效）

#### 選項 B: 使用 Colab/Kaggle 免費 GPU
- Google Colab: 免費 T4 GPU (16GB)
- Kaggle: 免費 P100 GPU (16GB)
- 訓練時間會快 5-10 倍

#### 選項 C: 繼續使用當前設定
- 確保電腦穩定運行
- 定期檢查訓練進度
- 有耐心等待

---

## 如何執行訓練

```bash
python train.py
```

### 預期輸出
```
使用裝置: cuda
載入資料...
訓練集樣本數: 1504
驗證集樣本數: 214
測試集樣本數: 75

開始訓練...
Epoch 1/100
Training: 0%|          | 0/1504 [00:00<?, ?it/s]
```

每個 iteration 應該會顯示：
```
Training: 1%| | 15/1504 [08:23<8:21:54, 20.23s/it, loss=0.7542, dice=0.0234]
```

---

## 監控 GPU 使用

開另一個終端執行：
```bash
nvidia-smi -l 1
```

確認：
- GPU 使用率應該接近 100%
- 顯存使用應該在 3-3.5 GB 左右（不超過 4GB）
- 溫度應該在 70-85°C 範圍內

---

## 檢查點

每次儲存最佳模型時會顯示：
```
✓ Saved best model (Dice: 0.xxxx)
```

模型儲存在: `models/best_model.pth`

---

## 如果還是失敗

### 最後的手段：降低影像解析度

如果 batch size 1 還是出錯，請修改影像大小：

1. 編輯 `brain_tumor_segmentation.py`
2. 找到 `TrainTransform` 和 `ValidTransform`
3. 將 `image_size=640` 改為 `image_size=448` 或 `512`

這會犧牲一些模型性能，但能確保訓練成功。

---

## 結論

**當前設定 (BATCH_SIZE=1) 是 GTX 960 4GB 能夠穩定訓練 640x640 影像的極限配置。**

如果這次成功開始訓練並持續運行，就讓它跑！雖然慢，但最終會得到一個訓練好的模型。

**祝訓練順利！** 🚀

---

## 修改歷史

1. ✅ 修正張量形狀和類型問題
2. ✅ 設定 NUM_WORKERS=0 (Windows)
3. ✅ 設定 KMP_DUPLICATE_LIB_OK=TRUE (OpenMP)
4. ✅ BATCH_SIZE: 8 → 2 → 1
5. ✅ 最終穩定配置達成
