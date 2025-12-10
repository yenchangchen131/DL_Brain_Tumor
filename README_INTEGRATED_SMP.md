# 腦腫瘤影像分割 - 整合版本 (SMP + 先進技術)

本專案整合了 `notebookadfcb42d18.ipynb` 中的先進技術到 `brain_tumor_integrated_backup.ipynb`。

## 🎯 整合的技術特色

### 1. **SMP Library (segmentation_models_pytorch)**
- 使用預訓練的 **UNet++** 架構（比原始 UNet 更強大）
- **ResNet34** 作為 encoder backbone（ImageNet 預訓練權重）
- 更好的特徵提取能力

### 2. **Mixed Precision Training (AMP)**
```python
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda')

with autocast('cuda'):
    outputs = model(images)
    loss = criterion(outputs, masks)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
- 加速訓練約 2-3 倍
- 降低記憶體使用約 40-50%
- 幾乎不影響精度

### 3. **AdamW Optimizer**
```python
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
```
- 比 Adam 更好的正則化
- weight decay 防止過擬合

### 4. **ReduceLROnPlateau Scheduler**
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10
)
```
- 自動調整學習率
- 當驗證指標停止改善時降低學習率

### 5. **進階資料增強 (Albumentations)**
```python
train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), 
             rotate=(-15, 15), p=0.5),
    A.ElasticTransform(p=0.3),      # 新增
    A.GridDistortion(p=0.3),         # 新增
    A.RandomBrightnessContrast(p=0.5),
    A.ColorJitter(p=0.5),            # 新增
    A.GaussNoise(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
```

### 6. **Focal Tversky Loss（可選）**
```python
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=4/3):
        # 對於不平衡資料特別有效
```

## 📊 技術對比

| 特性 | 原版本 | 整合版本 |
|------|--------|----------|
| 模型架構 | 自定義 UNet | **SMP UNet++ + ResNet34** |
| 預訓練權重 | ❌ 無 | ✅ ImageNet |
| Mixed Precision | ❌ 無 | ✅ torch.amp |
| 優化器 | Adam | **AdamW** (更好的正則化) |
| 學習率調整 | 手動 | **ReduceLROnPlateau** (自動) |
| 資料增強 | 基本 | **進階** (Affine, Elastic, Grid) |
| 訓練速度 | 基準 | **快 2-3 倍** ⚡ |
| 記憶體使用 | 基準 | **少 40-50%** 💾 |

## 🚀 快速開始

### 1. 安裝必要套件

```bash
pip install segmentation-models-pytorch
pip install albumentations
pip install opencv-python
pip install torch torchvision
```

或使用 requirements:

```bash
pip install -r requirements_smp.txt
```

### 2. 執行訓練

```bash
python brain_tumor_integrated_smp.py
```

### 3. 自訂參數

在檔案中修改這些超參數：

```python
# 超參數設定
IMG_SIZE = 256        # 影像大小（可改為 384 或 512）
BATCH_SIZE = 16       # 批次大小（依 GPU 記憶體調整）
EPOCHS = 80           # 訓練輪數
LR = 1e-4            # 學習率
NUM_WORKERS = 4      # DataLoader 工作執行緒數
```

## 📁 專案結構

```
DL_Brain_Tumor/
├── train/
│   ├── *.jpg                          # 訓練影像
│   └── _annotations.coco.json         # COCO 格式標註
├── valid/
│   ├── *.jpg
│   └── _annotations.coco.json
├── test/
│   ├── *.jpg
│   └── _annotations.coco.json
├── brain_tumor_integrated_smp.py      # ⭐ 整合版訓練腳本
├── brain_tumor_integrated_backup.ipynb # 原始 notebook
├── notebookadfcb42d18.ipynb           # 參考 notebook
├── unet_plusplus_best.pth             # 訓練好的模型
└── training_history.png               # 訓練曲線圖
```

## 🎓 主要差異說明

### UNet vs UNet++ with ResNet34

**原版 UNet:**
```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        # 從頭開始訓練
        # 約 31M 參數
```

**整合版 UNet++ + ResNet34:**
```python
model = smp.UnetPlusPlus(
    encoder_name='resnet34',      # 預訓練的 ResNet34
    encoder_weights='imagenet',   # ImageNet 權重
    in_channels=3,
    classes=1,
)
# 更好的特徵提取
# 更快的收斂
# 更高的精度
```

### 訓練循環改進

**原版:**
```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for images, masks in loader:
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
```

**整合版 (Mixed Precision):**
```python
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    for images, masks in loader:
        with autocast('cuda'):  # 🔥 Mixed Precision
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

## 🔧 常見問題

### Q1: CUDA Out of Memory 怎麼辦？

**解決方案 1: 降低 BATCH_SIZE**
```python
BATCH_SIZE = 8  # 從 16 降到 8
```

**解決方案 2: 降低影像大小**
```python
IMG_SIZE = 224  # 從 256 降到 224
```

**解決方案 3: 使用 Gradient Accumulation**
```python
# 每 2 個 batch 才更新一次參數
accumulation_steps = 2
```

### Q2: 訓練太慢怎麼辦？

確保：
- ✅ 使用 GPU (`DEVICE = "cuda"`)
- ✅ 使用 Mixed Precision Training
- ✅ 設定 `pin_memory=True` 在 DataLoader
- ✅ 調整 `NUM_WORKERS` (通常是 CPU 核心數)

### Q3: 如何使用訓練好的模型？

```python
import torch
import segmentation_models_pytorch as smp

# 載入模型
model = smp.UnetPlusPlus(
    encoder_name='resnet34',
    encoder_weights=None,  # 不需要預訓練權重
    in_channels=3,
    classes=1,
)
model.load_state_dict(torch.load('unet_plusplus_best.pth'))
model.eval()

# 預測
with torch.no_grad():
    output = model(image_tensor)
    pred_mask = torch.sigmoid(output) > 0.5
```

## 📈 預期效果

使用這些技術後，你應該能看到：

- ✅ **更快的訓練速度** (約快 2-3 倍)
- ✅ **更低的記憶體使用** (約少 40-50%)
- ✅ **更高的分割精度** (Dice Score 提升 2-5%)
- ✅ **更穩定的訓練** (學習率自動調整)
- ✅ **更快的收斂** (ImageNet 預訓練權重)

## 🎯 訓練建議

1. **先用小圖訓練快速驗證**
   ```python
   IMG_SIZE = 224
   EPOCHS = 10
   ```

2. **然後用中圖訓練**
   ```python
   IMG_SIZE = 256
   EPOCHS = 80
   ```

3. **最後用大圖 fine-tune**
   ```python
   IMG_SIZE = 384 或 512
   EPOCHS = 20
   LR = 1e-5  # 較小的學習率
   ```

## 📚 參考資料

- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [Albumentations Documentation](https://albumentations.ai/)
- [PyTorch AMP Tutorial](https://pytorch.org/docs/stable/amp.html)
- [UNet++ Paper](https://arxiv.org/abs/1807.10165)

## 🙏 致謝

本專案整合了以下技術：
- `notebookadfcb42d18.ipynb` 提供的先進架構和訓練技巧
- `brain_tumor_integrated_backup.ipynb` 的完整資料處理流程
- Segmentation Models PyTorch 團隊的優秀工作

---

**作者**: YourName  
**建立日期**: 2025-12-09  
**版本**: 1.0 - SMP 整合版
