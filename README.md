# 腦腫瘤影像分割專案

使用 U-Net 深度學習架構進行腦腫瘤 MRI 影像自動分割。

---

## 📋 目錄

- [專案簡介](#專案簡介)
- [快速開始](#快速開始)
- [環境設定](#環境設定)
- [資料準備](#資料準備)
- [訓練模型](#訓練模型)
- [CUDA Timeout 問題解決](#cuda-timeout-問題解決)
- [檢視結果](#檢視結果)
- [專案結構](#專案結構)
- [常見問題](#常見問題)

---

## 專案簡介

### 目標
- 建立自動化的腦腫瘤分割系統
- 達到高精度的分割效果（Dice Score > 0.80）
- 提供視覺化的預測結果以輔助醫學判讀

### 技術特色
- ✅ 完整的 U-Net 實作
- ✅ 資料增強（Albumentations）
- ✅ 組合損失函數（Dice + BCE）
- ✅ 完整的訓練與評估流程
- ✅ 視覺化功能
- ✅ Windows / GTX 960 優化設定

### 資料集
- **來源**: Roboflow TumorSegmentation Dataset
- **格式**: COCO Segmentation
- **影像數量**: 2,146 張
  - 訓練集: 1,504 張
  - 驗證集: 214 張
  - 測試集: 75 張
- **影像尺寸**: 640×640 pixels（可調整）

---

## 快速開始

### 🚀 最快速的方式

#### Windows 用戶（已解決 CUDA Timeout 問題）

```bash
# 雙擊這個檔案即可啟動：
start_training_fixed.bat
```

然後在 Jupyter 中：
1. 點擊 **Cell → Run All**
2. 開始訓練！

---

## 環境設定

### 硬體需求
- **建議**: NVIDIA GPU (CUDA 支援)
- **本專案優化**: GTX 960 4GB
- **最低**: CPU（訓練會非常慢）

### 軟體安裝

#### 方法 1: 使用 pip
```bash
pip install -r requirements.txt
```

#### 方法 2: 使用 conda（推薦）
```bash
# 建立新環境
conda create -n brain_tumor python=3.10
conda activate brain_tumor

# 安裝 PyTorch（根據您的 CUDA 版本）
# CUDA 12.1
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# 或 CUDA 11.8
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 或 CPU 版本
conda install pytorch torchvision cpuonly -c pytorch

# 安裝其他套件
pip install albumentations opencv-python tqdm pandas matplotlib
```

### 驗證安裝

運行以下程式碼確認安裝成功：

```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")
    print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

**預期輸出範例：**
```
PyTorch 版本: 2.5.1+cu121
CUDA 可用: True
CUDA 版本: 12.1
GPU 名稱: NVIDIA GeForce GTX 960
GPU 記憶體: 4.00 GB
```

### 📌 套件版本說明

本專案的 `requirements.txt` 已鎖定為經過測試的穩定版本：

```txt
torch==2.5.1
torchvision==0.20.1
numpy==2.0.1
opencv-python==4.12.0.88
Pillow==11.1.0
matplotlib==3.10.7
albumentations==2.0.8
tqdm==4.67.1
pandas==2.3.3
scikit-learn==1.7.2
```

**關於 CUDA 版本：**
- 如果您的系統有 CUDA 12.1，`pip install torch==2.5.1` 會自動安裝 `torch-2.5.1+cu121`
- 如果您的系統有 CUDA 11.8，會自動安裝 `torch-2.5.1+cu118`
- 如果沒有 CUDA，會安裝 CPU 版本
- 因此 requirements.txt 不需要指定 `+cu121` 後綴

**驗證套件版本：**
```bash
# Windows PowerShell
pip list | Select-String "torch|albumentations|opencv|numpy|pandas|matplotlib|tqdm|Pillow|scikit"

# Linux/Mac
pip list | grep -E "torch|albumentations|opencv|numpy|pandas|matplotlib|tqdm|Pillow|scikit"
```

### Windows 特別注意事項

本專案已針對 Windows 系統優化，解決了以下問題：
- ✅ OpenMP 衝突問題
- ✅ DataLoader 多進程問題
- ✅ CUDA Timeout 問題

---

## 資料準備

### 資料結構
確認您的資料結構如下：
```
DL_Brain_Tumor/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── valid/
│   ├── _annotations.coco.json
│   └── ...
└── test/
    ├── _annotations.coco.json
    └── ...
```

---

## 訓練模型

### 方法 1: 使用 Jupyter Notebook（推薦）

#### GTX 960 / 入門級 GPU 用戶（推薦）
```bash
# 使用優化版本（圖像大小 448×448）
jupyter notebook brain_tumor_complete_size448.ipynb
```

**優點：**
- 避免 CUDA timeout
- 訓練速度快 2 倍
- 可使用更大的 batch size (2-4)

#### 高階 GPU 用戶
```bash
# 使用原始版本（圖像大小 640×640）
jupyter notebook brain_tumor_complete.ipynb
```

### 方法 2: 使用 Python 腳本
```bash
python train.py
```

訓練腳本會自動：
- 載入資料
- 建立 U-Net 模型
- 使用資料增強
- 訓練模型（含 Early Stopping）
- 儲存最佳模型
- 在測試集上評估
- 產生視覺化結果

### 訓練參數設定

在 notebook 或 `train.py` 中可調整以下參數：

```python
# GTX 960 建議設定（448×448 版本）
BATCH_SIZE = 2-4        # 較小GPU使用2，較大記憶體可用4
NUM_EPOCHS = 100        # 最大訓練輪數
LEARNING_RATE = 1e-4    # 學習率
PATIENCE = 15           # Early Stopping patience
NUM_WORKERS = 0         # Windows 建議設為 0
```

```python
# 高階 GPU 設定（640×640 版本）
BATCH_SIZE = 8          # RTX 3080/4090 可用更大值
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
PATIENCE = 15
NUM_WORKERS = 4         # 根據 CPU 核心數調整
```

---

## CUDA Timeout 問題解決

### ⚠️ 問題現象
```
RuntimeError: CUDA error: the launch timed out and was terminated
```

### ✅ 解決方案

#### 方案 1: 使用優化版 Notebook（最簡單，推薦）

已為您準備好以下版本：

| Notebook | 圖像大小 | 速度提升 | 建議 Batch Size | 適用 GPU |
|----------|---------|---------|----------------|----------|
| `brain_tumor_complete_size448.ipynb` | 448×448 | ~2x | 2-4 | GTX 960/1060 |
| `brain_tumor_complete_size512.ipynb` | 512×512 | ~1.6x | 2 | GTX 1070/1080 |
| `brain_tumor_complete.ipynb` | 640×640 | 1x | 1-8 | RTX 3080+ |

**使用方式：**
```bash
# 雙擊啟動
start_training_fixed.bat

# 或手動啟動
jupyter notebook brain_tumor_complete_size448.ipynb
```

#### 方案 2: 修改 Windows TDR 設定（進階）

如果您想保持 640×640 解析度：

1. 按 `Win + R`，輸入 `regedit`
2. 導航到：`HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers`
3. 新增兩個 DWORD (32位) 值：
   - `TdrDelay` = `60` (十進位)
   - `TdrLevel` = `0` (十進位)
4. 重啟電腦

**警告：** 這會禁用 GPU 超時保護，如果 GPU 掛起可能導致系統無響應。

#### 方案 3: 使用雲端 GPU（最快）

- **Google Colab**: 免費 T4 GPU (16GB)，速度快 5-10 倍
- **Kaggle**: 免費 P100 GPU (16GB)，每週 30 小時配額

### 效果對比

| 項目 | 原始 (640) | 優化後 (448) | 改善 |
|-----|-----------|-------------|------|
| CUDA Timeout | ❌ 出錯 | ✅ 正常 | - |
| 訓練速度 | 1x | 2x | +100% |
| 每 epoch 時間 | 8-12 小時 | 4-6 小時 | -50% |
| 總訓練時間 | 5-15 天 | 2-7 天 | -50% |

---

## 檢視結果

### 1. 訓練曲線

訓練完成後，結果儲存在 `results/` 目錄：

```python
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('results/training_curves.png')
plt.figure(figsize=(15, 5))
plt.imshow(img)
plt.axis('off')
plt.show()
```

### 2. 評估指標

```python
import json

with open('results/test_metrics.json', 'r') as f:
    metrics = json.load(f)

print("平均指標:")
for key, value in metrics['average'].items():
    print(f"  {key}: {value:.4f}")
```

### 3. 預測視覺化

```python
img = Image.open('results/predictions.png')
plt.figure(figsize=(16, 20))
plt.imshow(img)
plt.axis('off')
plt.show()
```

### 4. 預期結果

#### 訓練時間（參考）
- **GTX 960 (448×448)**: 每 epoch 約 4-6 小時
- **RTX 3080 (640×640)**: 每 epoch 約 30-60 分鐘
- **CPU**: 不建議（太慢）

#### 預期指標
- Validation Dice Score: > 0.85
- Test Dice Score: > 0.80
- Test IoU: > 0.70

---

## 專案結構

```
DL_Brain_Tumor/
├── brain_tumor_complete.ipynb          # 完整訓練 notebook (640×640)
├── brain_tumor_complete_size448.ipynb  # 優化版 (448×448，推薦)
├── brain_tumor_complete_size512.ipynb  # 中等版 (512×512)
├── brain_tumor_segmentation.py         # 核心功能模組
├── train.py                            # 訓練腳本
├── start_training_fixed.bat            # 快速啟動工具
├── requirements.txt                    # 套件依賴
├── REPORT.md                           # 專案報告
├── train/                              # 訓練集
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/                              # 驗證集
│   ├── _annotations.coco.json
│   └── *.jpg
├── test/                               # 測試集
│   ├── _annotations.coco.json
│   └── *.jpg
├── models/                             # 訓練模型
│   └── best_model.pth
└── results/                            # 訓練結果
    ├── training_curves.png
    ├── predictions.png
    └── test_metrics.json
```

---

## 常見問題

### Q1: CUDA out of memory

**解決方案：**
1. 減少 `BATCH_SIZE`（例如改為 2 或 1）
2. 使用較小解析度的 notebook（448 或 512）
3. 減少模型的 feature channels

### Q2: CUDA error: the launch timed out and was terminated

**解決方案：**
1. 使用 `brain_tumor_complete_size448.ipynb`（最簡單）
2. 降低 batch size 為 1
3. 修改 Windows TDR 設定（進階，見上方說明）

### Q3: DataLoader worker exited unexpectedly（Windows）

**解決方案：**
```python
NUM_WORKERS = 0  # Windows 系統建議設為 0
```

### Q4: OMP: Error #15: Initializing libiomp5md.dll

**解決方案：**
在程式開頭加入：
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```
（所有 notebook 已包含此設定）

### Q5: 訓練速度很慢

**解決方案：**
1. 確認使用 GPU：`device = torch.device('cuda')`
2. 檢查 CUDA 是否正確安裝：`torch.cuda.is_available()`
3. 使用較小解析度提升速度（448×448）
4. 考慮使用雲端 GPU（Colab/Kaggle）

### Q6: Dice Score 不高

**可能原因和解決方案：**
1. 訓練不夠久：增加 epochs 或減少 patience
2. 學習率不適合：嘗試 1e-3 或 1e-5
3. 資料增強不足：增加更多 augmentation
4. 模型容量問題：調整 U-Net 的 feature 數量

### Q7: 過擬合（訓練集好但驗證集差）

**解決方案：**
1. 增強資料增強
2. 增加 weight decay
3. 減少模型複雜度
4. 使用 Dropout

---

## 進階使用

### 1. 繼續訓練

```python
# 載入 checkpoint
checkpoint = torch.load('models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 繼續訓練
history, best_dice = train_model(
    model, train_loader, valid_loader,
    criterion, optimizer, scheduler, device,
    num_epochs=50  # 再訓練 50 個 epochs
)
```

### 2. 單一影像預測

```python
# 載入模型
model.eval()

# 預測單一影像
image, mask = test_dataset[0]
with torch.no_grad():
    pred = model(image.unsqueeze(0).to(device))
    pred_mask = torch.sigmoid(pred).cpu().squeeze().numpy()

# 視覺化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image.permute(1,2,0))
axes[1].imshow(mask.squeeze(), cmap='gray')
axes[2].imshow(pred_mask, cmap='gray')
plt.show()
```

### 3. 匯出模型為 ONNX

```python
dummy_input = torch.randn(1, 3, 640, 640).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "models/brain_tumor_model.onnx",
    export_params=True,
    input_names=['input'],
    output_names=['output']
)
```

---

## 技術細節

### U-Net 架構

```
編碼器 (Encoder):
├─ Conv Block 1: 3 → 64 channels
├─ Conv Block 2: 64 → 128 channels
├─ Conv Block 3: 128 → 256 channels
└─ Conv Block 4: 256 → 512 channels

瓶頸層 (Bottleneck):
└─ Conv Block: 512 → 1024 channels

解碼器 (Decoder):
├─ UpConv + Skip + Conv Block: 1024 → 512
├─ UpConv + Skip + Conv Block: 512 → 256
├─ UpConv + Skip + Conv Block: 256 → 128
└─ UpConv + Skip + Conv Block: 128 → 64

輸出層:
└─ Conv: 64 → 1 channel
```

**總參數量**: 31,043,521

### 損失函數

組合損失函數：
```
Total Loss = 0.5 × Dice Loss + 0.5 × BCE Loss
```

### 資料增強

**訓練集：**
- 水平翻轉 (p=0.5)
- 垂直翻轉 (p=0.5)
- 隨機旋轉 ±15° (p=0.5)
- 隨機亮度/對比度 (p=0.3)
- 高斯模糊 (p=0.2)
- 彈性變形 (p=0.2)
- ImageNet 正規化

**驗證/測試集：**
- 僅調整大小和正規化

---

## 參考資源

- [U-Net 論文](https://arxiv.org/abs/1505.04597) - Ronneberger et al., 2015
- [PyTorch 官方文檔](https://pytorch.org/docs/)
- [Albumentations 文檔](https://albumentations.ai/)
- [Dataset: Roboflow TumorSegmentation](https://universe.roboflow.com/tumorsegmentation) (CC BY 4.0)

---

## 授權與引用

### Dataset License
本專案使用的資料集採用 CC BY 4.0 授權。

### 專案作者
[您的名字]

### 完成日期
2025-12-07

### 最後更新
- **2025-12-07**: 套件版本已鎖定為經測試的穩定版本（使用 `==` 精確版本）
- **2025-12-06**: CUDA Timeout 問題解決方案完成
- **2025-12-05**: GPU 訓練環境設定完成

---

## 支援

如有問題，請按以下順序檢查：
1. 閱讀本 README 的常見問題部分
2. 檢查錯誤訊息
3. 確認環境配置正確
4. 查看 `REPORT.md` 獲取更詳細的專案報告

---

**針對 GTX 960 (4GB) 優化 ✅**  
**CUDA Timeout 問題已解決 ✅**
