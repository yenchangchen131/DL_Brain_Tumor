# 腦腫瘤影像分割 - 專案報告

## 1. 專案簡介

### 1.1 背景
腦腫瘤是一種嚴重的疾病，早期診斷和精確分割對於治療規劃至關重要。本專案使用深度學習技術，特別是U-Net架構，來自動化腦腫瘤MRI影像的分割任務。

### 1.2 目標
- 建立一個自動化的腦腫瘤分割系統
- 達到高精度的分割效果（Dice Score > 0.80）
- 提供視覺化的預測結果以輔助醫學判讀

## 2. 資料集分析

### 2.1 資料集概述
- **來源**: Roboflow TumorSegmentation Dataset
- **格式**: COCO Segmentation
- **影像數量**: 2,146張
- **影像尺寸**: 640×640 pixels
- **標註類型**: Binary segmentation (腫瘤/背景)

### 2.2 資料分布
```
訓練集: [填入實際數量] 張
驗證集: [填入實際數量] 張
測試集: [填入實際數量] 張
```

### 2.3 資料特性
- 已進行自動方向校正
- 已調整至統一尺寸（640×640，拉伸）
- 原始資料未經增強處理

## 3. 方法論

### 3.1 資料預處理

#### 3.1.1 資料增強策略
為了提升模型的泛化能力並防止過擬合，我們對訓練集採用以下增強技術：

1. **幾何變換**
   - 水平翻轉 (p=0.5)
   - 垂直翻轉 (p=0.5)
   - 隨機旋轉 (±15度, p=0.5)
   - 彈性變形 (p=0.2)

2. **顏色變換**
   - 隨機亮度/對比度調整 (p=0.3)
   - 高斯模糊 (p=0.2)

3. **正規化**
   - 使用ImageNet的mean和std進行正規化
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

#### 3.1.2 Mask處理
- 將COCO格式的polygon segmentation轉換為binary mask
- 確保mask與影像的對應關係正確

### 3.2 模型架構

#### 3.2.1 U-Net
選擇U-Net作為主要模型架構，原因如下：
- 專為醫學影像分割設計
- 編碼器-解碼器結構有效捕捉多尺度特徵
- Skip connections保留細節資訊
- 在小資料集上也能達到良好效果

#### 3.2.2 架構細節
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
└─ Conv: 64 → 1 channel (sigmoid activation)
```

**總參數量**: [填入實際數量]

### 3.3 訓練策略

#### 3.3.1 損失函數
使用組合損失函數以達到最佳效果：
- **Dice Loss** (50%): 處理類別不平衡問題
- **Binary Cross Entropy** (50%): 提供逐像素的監督信號

組合公式：
```
Total Loss = 0.5 × Dice Loss + 0.5 × BCE Loss
```

#### 3.3.2 優化器與學習率
- **優化器**: Adam
- **初始學習率**: 1×10⁻⁴
- **權重衰減**: 1×10⁻⁵
- **學習率調整**: ReduceLROnPlateau
  - Patience: 5 epochs
  - Reduction factor: 0.5

#### 3.3.3 訓練配置
- **Batch Size**: 8
- **最大Epochs**: 100
- **Early Stopping**: Patience=15 epochs
- **監控指標**: Validation Dice Score

## 4. 實驗結果

### 4.1 訓練過程

#### 4.1.1 訓練曲線
[插入 training_curves.png]

**觀察**:
- 訓練損失和驗證損失的收斂情況
- 是否有過擬合現象
- 學習率調整的時機和效果

#### 4.1.2 訓練統計
```
訓練輪數: [填入] epochs
最佳Epoch: [填入]
最佳驗證Dice Score: [填入]
是否Early Stop: [是/否]
```

### 4.2 測試集評估

#### 4.2.1 整體指標
| 指標 | 數值 |
|------|------|
| Dice Coefficient | [填入] |
| IoU Score | [填入] |
| Precision | [填入] |
| Recall | [填入] |
| F1 Score | [填入] |

#### 4.2.2 指標分布
[插入 metrics_distribution.png]

**分析**:
- 各指標的平均值和標準差
- 表現最好和最差的樣本分析
- 可能的改進方向

### 4.3 視覺化結果

#### 4.3.1 預測樣本
[插入 predictions.png]

**說明**:
- 左至右: 原始影像 | Ground Truth | 模型預測 | 疊加圖
- 選擇代表性的樣本展示

#### 4.3.2 結果分析
1. **成功案例**: 模型能準確分割的情況
2. **挑戰案例**: 模型表現較差的情況及其可能原因
   - 腫瘤邊界模糊
   - 小尺寸腫瘤
   - 複雜形狀

## 5. 討論與分析

### 5.1 模型表現
- **優點**:
  - [根據實際結果填寫]
  
- **限制**:
  - [根據實際結果填寫]

### 5.2 與其他方法比較
[如果有baseline或其他方法可比較]

### 5.3 失敗案例分析
- 分析模型預測失敗的原因
- 提出可能的改進策略

## 6. 結論

### 6.1 研究成果
本研究成功實作了基於U-Net的腦腫瘤自動分割系統，主要成果包括：
1. 在測試集上達到 Dice Score [填入數值]
2. 建立完整的訓練和評估pipeline
3. 提供可視化工具輔助結果分析

### 6.2 實際應用價值
- 可以輔助醫生進行腫瘤診斷
- 減少人工標註的時間成本
- 提供量化的評估指標

### 6.3 未來工作
1. **模型改進**:
   - 嘗試其他架構（如DeepLabV3+, SegFormer）
   - 引入注意力機制
   - 多尺度預測

2. **資料增強**:
   - 更進階的增強技術
   - 合成資料生成

3. **後處理**:
   - 形態學操作優化邊界
   - 條件隨機場（CRF）後處理

4. **實際部署**:
   - 模型輕量化
   - 推理速度優化
   - 整合到臨床工作流程

## 7. 參考資料

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.

2. Dataset: Roboflow TumorSegmentation Dataset (CC BY 4.0)

3. [其他相關文獻]

## 附錄

### A. 環境配置
- Python: 3.x
- PyTorch: [版本]
- CUDA: [版本]
- 其他主要套件: albumentations, opencv-python, numpy, matplotlib

### B. 程式碼結構
```
DL_Brain_Tumor/
├── brain_tumor_segmentation.py  # 主要功能實作
├── train.py                      # 訓練腳本
├── notebook_guide.py             # Notebook執行指南
├── code.ipynb                    # Jupyter Notebook
├── models/
│   └── best_model.pth           # 最佳模型權重
├── results/
│   ├── training_curves.png
│   ├── predictions.png
│   ├── metrics_distribution.png
│   └── *.json
└── README.md
```

### C. 超參數總結
```json
{
  "batch_size": 8,
  "num_epochs": 100,
  "learning_rate": 1e-4,
  "patience": 15,
  "optimizer": "Adam",
  "weight_decay": 1e-5,
  "loss": "Combined (Dice + BCE)"
}
```

---

**專案完成日期**: 2025-12-05
**作者**: [您的名字]
