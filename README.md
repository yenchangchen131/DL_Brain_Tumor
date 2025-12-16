# Brain Tumor Detection (Segmentation)

本專案以 **腦腫瘤影像分割**為目標，使用 COCO 格式標註資料集，完成：

1. **資料探索與清理**：找出無標註、重複/多重標註等異常資料並清理，輸出 cleaned 標註檔。
2. **模型訓練與評估**：以 `segmentation_models_pytorch` 的 **U-Net++ (ResNet34 encoder)** 進行訓練，並在驗證集上搜尋最佳二值化閾值，最後在測試集輸出 Dice / IoU / Precision / Recall / F1 / Accuracy 等指標並視覺化預測。

---

## 專案結構與資料格式

Notebook 預期資料集以資料夾區分資料切分（Train/Valid/Test），且每個切分資料夾中包含影像與 COCO JSON：

```
./train/
  _annotations.coco.json
  (images...)

./valid/
  _annotations.coco.json
  (images...)

./test/
  _annotations.coco.json
  (images...)
```

清理後會在各 split 目錄下產生：

```
_annotations.coco.cleaned.json
```

---

## 環境需求

Notebook 主要依賴：

* Python 3.x
* `torch`
* `opencv-python` (`cv2`)
* `albumentations` + `albumentations.pytorch (ToTensorV2)`
* `segmentation-models-pytorch`
* `numpy`, `matplotlib`, `tqdm`

可用（示意）方式安裝：

```bash
pip install torch opencv-python albumentations segmentation-models-pytorch matplotlib tqdm numpy
```

---

## Notebook 流程說明（brain_tumor.ipynb）

### Part 1：資料探索與清理

核心流程由 `clean_and_analyze_data()` 完成：

* 載入 `train/valid/test` 的 COCO JSON（`_annotations.coco.json`）
* 分析並識別資料問題：

  * 無標註的圖片
  * 多重/重複標註的圖片（依 notebook 內規則判斷）
* 移除異常圖片與其 annotations
* 輸出清理後標註檔：`_annotations.coco.cleaned.json`
* 隨機抽樣視覺化訓練集影像與遮罩（polygon segmentation 轉 mask 後疊圖檢查）

> 遮罩生成方式：將 COCO `segmentation`（polygon）以 OpenCV `fillPoly` 填入同一張 binary mask。

---

### Part 2：模型訓練與評估

#### 1) Dataset / DataLoader

* Dataset 類別：`BrainTumorDataset(Dataset)`

  * 讀取影像、依 COCO polygon 的 segmentation生成 mask
  * 支援讀取 cleaned 檔案，若 cleaned 不存在會回退讀取原始 JSON
* DataLoader 建立：`get_loaders(batch_size=16, num_workers=4)`

  * 預設 `batch_size=16`
  * 測試集 loader 會使用 batch size = 1（避免不必要的堆疊與便於視覺化）

#### 2) Data Augmentation

* 由 `get_transforms(image_size=640)` 定義
* 使用 `albumentations`，包含 `A.Resize(image_size, image_size)` 與 ToTensorV2 等流程（以 notebook 定義為準）
* 預設影像大小：**640 x 640**

#### 3) 模型

* `build_model()`：建立 **U-Net++**

  * Encoder：`resnet34`
  * Encoder weights：`imagenet`
  * `in_channels=3`, `classes=1`

#### 4) Loss、Optimizer、Scheduler

* Loss：

  * `FocalLoss(mode='binary', alpha=0.8, gamma=2.0)`
  * `DiceLoss(mode='binary', from_logits=True)`
* Optimizer：

  * Adam，`lr=1e-4`
* Scheduler：

  * `ReduceLROnPlateau(optimizer, 'max', patience=5)`（以驗證指標為監控目標）

並使用 AMP 混合精度：

* `torch.cuda.amp.autocast`
* `GradScaler`

#### 5) 訓練迴圈與 Early Stopping

* 主迴圈：`train_eval_loop(model, train_loader, valid_loader, epochs=100, patience=15)`

  * 預設訓練 `epochs=100`
  * Early stopping：`patience=15`
  * 會儲存最佳權重：`best_model.pth`

#### 6) 評估指標

* `calculate_metrics(pred, target, threshold=0.5)` 支援自訂閾值，計算：

  * Dice
  * IoU
  * Precision
  * Recall
  * F1
  * Accuracy（含 TN）

#### 7) 閾值搜尋（Validation）

* `find_best_threshold(model, loader, device)`

  * 搜尋範圍：**0.30 ~ 0.70**
  * step：**0.05**
  * 以驗證集 Dice 最大為準，回傳 `best_threshold`

#### 8) Test Set 評估與視覺化

* `evaluate_test_set(model, test_loader, threshold=best_threshold)`
  輸出測試集平均指標（Dice/IoU/Precision/Recall/F1/Accuracy）
* `visualize_predictions(..., threshold=best_threshold)`
  隨機抽樣顯示影像、GT mask、Pred mask（依 notebook 版本繪製）

---

## 如何使用

1. 將資料集依前述結構放好（`./train ./valid ./test`）
2. 打開並執行 `brain_tumor.ipynb`
3. 依序執行：

   * Part 1：清理與視覺化（會輸出 cleaned JSON）
   * Part 2：訓練、畫曲線、載入最佳權重、threshold tuning、測試集評估與視覺化

輸出重點檔案：

* `best_model.pth`：最佳模型權重
* `*_annotations.coco.cleaned.json`：清理後標註檔（依 split 產生）

---

## 注意事項

* 本專案將 COCO polygon segmentation 轉為 binary mask（腫瘤=1 / 背景=0），並以單通道輸出進行二元分割。
* 若 GPU 可用會優先使用（Notebook 內 `device` 邏輯）。
* 若要加速訓練，可調整 `image_size`、`batch_size`、或減少資料增強強度。