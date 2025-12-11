這是一份根據您提供的 `brain_tumor.ipynb` 程式碼內容所撰寫的 README 文件。我參考了您上傳的 `README.md` 風格，使用了 Emoji 與清晰的結構來呈現。

-----

# 🧠 Brain Tumor Detection - 腦腫瘤影像分割專案

本專案提供了一套完整的腦腫瘤 MRI 影像分割流程（Segmentation Pipeline），從資料集的清理、前處理，到使用先進的深度學習模型進行訓練、評估與視覺化。

## 🎯 專案特色

本程式碼整合了資料清理與深度學習模型訓練，主要特色如下：

### 1\. **自動化資料清理 (Data Cleaning)**

  - 自動載入 COCO 格式的標註檔案（`_annotations.coco.json`）。
  - **異常檢測**：識別並移除無標註或重複標註的異常圖片。
  - **自動修正**：將清理後的資料儲存為 `.cleaned.json`，確保訓練資料的品質。

### 2\. **強大的模型架構 (SMP U-Net++)**

  - **架構**：使用 `segmentation_models_pytorch` (SMP) 函式庫中的 **U-Net++**。
  - **Backbone**：採用 **ResNet34** 作為編碼器 (Encoder)，並載入 **ImageNet** 預訓練權重，以加快收斂速度並提升特徵提取能力。

### 3\. **混合精度訓練 (Mixed Precision Training)**

  - 實作 `torch.amp.autocast` 與 `GradScaler`。
  - **優勢**：在保持模型精度的同時，顯著減少 GPU 記憶體使用量並加速訓練過程。

### 4\. **複合損失函數 (Combined Loss)**

結合了兩種損失函數以優化分割效果：

  - **Focal Loss**：專注於難以分類的樣本，解決正負樣本不平衡問題。
  - **Dice Loss**：直接優化分割任務的核心指標（重疊率）。
  - **公式**：`Loss = 0.5 * FocalLoss + 0.5 * DiceLoss`

### 5\. **動態學習率調整**

  - 使用 `ReduceLROnPlateau` 排程器。
  - 當驗證集的 Dice Score 停止提升時，自動降低學習率，幫助模型跳出局部最佳解。

### 6\. **最佳閾值搜尋 (Threshold Tuning)**

  - 訓練完成後，不會直接使用預設的 0.5，而是在驗證集上自動搜尋能讓 Dice Score 最高的**最佳閾值 (Best Threshold)**，進一步提升測試集表現。

-----

## 🛠️ 執行流程 (Pipeline)

### Step 1: 環境與資料準備

程式會自動檢查 GPU 可用性（支援 CUDA），並設定隨機種子以確保結果可重現。

  - **資料增強**：使用 `Albumentations` 進行豐富的圖像增強，包括：
      - `ElasticTransform` (彈性變形)
      - `GaussianBlur` (高斯模糊)
      - `RandomBrightnessContrast` (亮度對比調整)
      - 翻轉與旋轉

### Step 2: 資料清理

呼叫 `clean_and_analyze_data()` 函數：

1.  分析 Train/Valid/Test 資料集。
2.  移除異常 ID（如多重標註或無標註）。
3.  產生 `_annotations.coco.cleaned.json`。

### Step 3: 模型訓練

呼叫 `train_eval_loop()` 函數開始訓練：

  - **Optimizer**: Adam (lr=1e-4)
  - **Epochs**: 預設 100 (包含 Early Stopping 機制，Patience=15)
  - **儲存模型**: 自動儲存驗證集 Dice Score 最高的模型為 `best_model.pth`。

### Step 4: 評估與測試

1.  **載入權重**: 載入 `best_model.pth`。
2.  **尋找閾值**: 使用 `find_best_threshold()` 在驗證集上找出最佳切分點。
3.  **測試集評估**: 使用最佳閾值在測試集上計算最終指標（Dice, IoU, Precision, Recall, F1, Accuracy）。

### Step 5: 視覺化

使用 `visualize_predictions()` 隨機抽取樣本，並排顯示：

  - 原始 MRI 影像
  - 真實遮罩 (Ground Truth)
  - 模型預測 (Prediction)
  - 疊加比較圖 (Overlay)

-----

## 📊 效能表現 (範例)

根據訓練日誌，模型在測試集上的表現如下：

| Metric | Score |
|--------|-------|
| **Mean Dice Score** | **0.8106** |
| Mean IoU Score | 0.7141 |
| Mean Precision | 0.8268 |
| Mean Recall | 0.8447 |
| Mean Accuracy | **98.70%** |

*(數據基於最佳閾值 Thr=0.50)*

-----

## 💻 程式碼片段

### 模型建構

```python
import segmentation_models_pytorch as smp

model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)
```

### 訓練迴圈 (AMP)

```python
scaler = GradScaler()

with autocast():
    outputs = model(imgs)
    loss = combined_loss(outputs, masks)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 📋 需求套件

  - `torch`
  - `segmentation-models-pytorch`
  - `albumentations`
  - `opencv-python`
  - `matplotlib`
  - `tqdm`