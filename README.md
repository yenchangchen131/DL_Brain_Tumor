# 腦腫瘤影像分割 - 快速開始指南

## 📋 目錄
1. [環境設置](#環境設置)
2. [資料準備](#資料準備)
3. [訓練模型](#訓練模型)
4. [使用Jupyter Notebook](#使用jupyter-notebook)
5. [檢視結果](#檢視結果)
6. [常見問題](#常見問題)

---

## 🔧 環境設置

### 方法1: 使用 pip
```bash
# 安裝所有必要套件
pip install -r requirements.txt
```

### 方法2: 使用 conda (推薦)
```bash
# 建立新環境
conda create -n brain_tumor python=3.10
conda activate brain_tumor

# 安裝 PyTorch (根據您的CUDA版本選擇)
# CUDA 11.8
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 或 CUDA 12.1
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# 或 CPU版本
conda install pytorch torchvision cpuonly -c pytorch

# 安裝其他套件
pip install albumentations opencv-python tqdm pandas
```

### 驗證安裝
```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
```

---

## 📁 資料準備

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

## 🚀 訓練模型

### 方法1: 使用Python腳本
```bash
# 執行訓練
python train.py
```

訓練腳本會自動：
- 載入資料
- 建立U-Net模型
- 使用資料增強
- 訓練模型（含early stopping）
- 儲存最佳模型
- 在測試集上評估
- 產生視覺化結果

### 方法2: 修改訓練參數
編輯 `train.py` 中的參數：
```python
BATCH_SIZE = 8          # 根據GPU記憶體調整
NUM_EPOCHS = 100        # 最大訓練輪數
LEARNING_RATE = 1e-4    # 學習率
PATIENCE = 15           # Early stopping patience
NUM_WORKERS = 4         # DataLoader workers
```

---

## 📓 使用Jupyter Notebook

### 1. 開啟Notebook
```bash
jupyter notebook code.ipynb
```

### 2. 逐步執行

#### Step 1: 安裝套件（如需要）
```python
!pip install albumentations opencv-python
```

#### Step 2: 匯入模組
將 `notebook_guide.py` 的內容複製到notebook cells中，或使用：
```python
%run brain_tumor_segmentation.py
```

#### Step 3: 設定參數並執行
按照 `notebook_guide.py` 中的順序執行各個cell

### 3. 可選：只執行特定部分
- **只訓練**: 執行Cell 1-9
- **只評估**: 載入已訓練的模型，執行Cell 11-14
- **只視覺化**: 執行Cell 12

---

## 📊 檢視結果

訓練完成後，結果儲存在 `results/` 目錄：

### 1. 訓練曲線
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

### 4. 完整結果總結
```python
with open('results/final_results.json', 'r') as f:
    results = json.load(f)
    
print(json.dumps(results, indent=2))
```

---

## ❓ 常見問題

### Q1: CUDA out of memory
**解決方案**:
- 減少 `BATCH_SIZE`（例如改為4或2）
- 減少模型的feature channels
- 使用mixed precision training

### Q2: 訓練速度很慢
**解決方案**:
- 確認使用GPU: `device = torch.device('cuda')`
- 增加 `NUM_WORKERS`（但不要超過CPU核心數）
- 確認CUDA和cuDNN已正確安裝

### Q3: Dice Score不高
**可能原因和解決方案**:
- 訓練不夠久：增加epochs或減少patience
- 學習率不適合：嘗試1e-3或1e-5
- 資料增強不足：增加更多augmentation
- 模型容量問題：調整U-Net的feature數量

### Q4: 過擬合 (訓練集好但驗證集差)
**解決方案**:
- 增強資料增強
- 增加weight decay
- 減少模型複雜度
- 使用Dropout

### Q5: 驗證集loss不下降
**檢查事項**:
- 確認資料載入正確
- 檢查learning rate是否太小或太大
- 確認資料增強沒有太激進
- 嘗試不同的loss function weight

---

## 📝 快速測試

### 最小可運行測試
```python
import torch
from brain_tumor_segmentation import UNet

# 建立模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, out_channels=1).to(device)

# 測試前向傳播
test_input = torch.randn(1, 3, 640, 640).to(device)
with torch.no_grad():
    output = model(test_input)

print(f"輸入形狀: {test_input.shape}")
print(f"輸出形狀: {output.shape}")
print("✓ 模型測試成功！")
```

---

## 🎯 預期結果

### 訓練時間（參考）
- GPU (RTX 3080): ~30-60分鐘
- GPU (GTX 1080): ~1-2小時
- CPU: 不建議（太慢）

### 預期指標
- Validation Dice Score: > 0.85
- Test Dice Score: > 0.80
- Test IoU: > 0.70

---

## 📚 進階使用

### 1. 繼續訓練
```python
# 載入checkpoint
checkpoint = torch.load('models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 繼續訓練
history, best_dice = train_model(
    model, train_loader, valid_loader, 
    criterion, optimizer, scheduler, device,
    num_epochs=50  # 再訓練50個epochs
)
```

### 2. 單一影像預測
```python
from brain_tumor_segmentation import BrainTumorDataset, ValidTransform
import matplotlib.pyplot as plt

# 載入測試資料集
test_dataset = BrainTumorDataset(
    'test', 
    'test/_annotations.coco.json',
    transform=ValidTransform()
)

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
axes[0].set_title('Image')
axes[1].imshow(mask.squeeze(), cmap='gray')
axes[1].set_title('Ground Truth')
axes[2].imshow(pred_mask, cmap='gray')
axes[2].set_title('Prediction')
plt.show()
```

### 3. 匯出模型
```python
# 匯出為ONNX格式
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

## 🔗 相關資源

- [PyTorch官方文檔](https://pytorch.org/docs/)
- [U-Net論文](https://arxiv.org/abs/1505.04597)
- [Albumentations文檔](https://albumentations.ai/)

---

## 📧 Support

如有問題，請檢查：
1. 錯誤訊息
2. 本README的常見問題section
3. 確認環境配置正確

---

**最後更新**: 2025-12-05
