"""
Jupyter Notebook 執行腳本
用於在 code.ipynb 中執行訓練
將此程式碼複製到 notebook cell 中執行
"""

# ============================================================================
# Cell 1: 安裝必要套件
# ============================================================================
"""
!pip install albumentations
!pip install opencv-python
"""

# ============================================================================
# Cell 2: 匯入套件
# ============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import cv2
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import random
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2

print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ============================================================================
# Cell 3: 從 brain_tumor_segmentation.py 匯入所有功能
# ============================================================================
# 如果在 Jupyter 中，可以使用 %run 來執行腳本
# %run brain_tumor_segmentation.py

# 或者直接匯入
from brain_tumor_segmentation import *

# ============================================================================
# Cell 4: 設定參數
# ============================================================================
# 資料路徑
BASE_DIR = Path('.')
TRAIN_DIR = BASE_DIR / 'train'
VALID_DIR = BASE_DIR / 'valid'
TEST_DIR = BASE_DIR / 'test'

TRAIN_ANN = TRAIN_DIR / '_annotations.coco.json'
VALID_ANN = VALID_DIR / '_annotations.coco.json'
TEST_ANN = TEST_DIR / '_annotations.coco.json'

# 訓練參數
CONFIG = {
    'batch_size': 8,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'patience': 15,
    'num_workers': 2,  # Notebook中建議使用較少的workers
}

# 裝置設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用裝置: {device}')

# 建立儲存目錄
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ============================================================================
# Cell 5: 載入並檢查資料
# ============================================================================
print('載入資料...')

train_loader, valid_loader, test_loader = create_dataloaders(
    train_dir=TRAIN_DIR,
    valid_dir=VALID_DIR,
    test_dir=TEST_DIR,
    train_ann=TRAIN_ANN,
    valid_ann=VALID_ANN,
    test_ann=TEST_ANN,
    batch_size=CONFIG['batch_size'],
    num_workers=CONFIG['num_workers']
)

print(f'訓練集樣本數: {len(train_loader.dataset)}')
print(f'驗證集樣本數: {len(valid_loader.dataset)}')
print(f'測試集樣本數: {len(test_loader.dataset)}')

# 檢查一個batch的資料
images, masks = next(iter(train_loader))
print(f'\nBatch形狀:')
print(f'  Images: {images.shape}')
print(f'  Masks: {masks.shape}')

# ============================================================================
# Cell 6: 視覺化一些訓練樣本
# ============================================================================
def show_batch(images, masks, num_samples=4):
    """顯示一個batch中的部分樣本"""
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3*num_samples))
    
    for i in range(min(num_samples, len(images))):
        # 反正規化影像
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        mask = masks[i].cpu().squeeze().numpy()
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Image {i+1}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f'Mask {i+1}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

show_batch(images, masks, num_samples=4)

# ============================================================================
# Cell 7: 建立模型
# ============================================================================
print('建立模型...')

model = UNet(in_channels=3, out_channels=1).to(device)

# 計算參數量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'總參數量: {total_params:,}')
print(f'可訓練參數量: {trainable_params:,}')

# 測試模型
with torch.no_grad():
    test_input = torch.randn(1, 3, 640, 640).to(device)
    test_output = model(test_input)
    print(f'\n模型輸入形狀: {test_input.shape}')
    print(f'模型輸出形狀: {test_output.shape}')

# ============================================================================
# Cell 8: 設定損失函數、優化器
# ============================================================================
criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
optimizer = optim.Adam(
    model.parameters(), 
    lr=CONFIG['learning_rate'], 
    weight_decay=1e-5
)
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='max',
    patience=5,
    factor=0.5,
    verbose=True
)

print('損失函數: Combined Loss (Dice + BCE)')
print(f'優化器: Adam (lr={CONFIG["learning_rate"]})')
print('學習率調整: ReduceLROnPlateau')

# ============================================================================
# Cell 9: 開始訓練
# ============================================================================
print('開始訓練...')
print('='*50)

history, best_dice = train_model(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    num_epochs=CONFIG['num_epochs'],
    patience=CONFIG['patience'],
    save_dir='./models'
)

print(f'\n訓練完成！最佳驗證 Dice Score: {best_dice:.4f}')

# 儲存訓練歷史
with open('results/training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

# ============================================================================
# Cell 10: 繪製訓練曲線
# ============================================================================
plot_training_history(history, save_path='results/training_curves.png')

# ============================================================================
# Cell 11: 載入最佳模型並測試
# ============================================================================
print('載入最佳模型並在測試集上評估...')

# 載入最佳模型
checkpoint = torch.load('models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 測試
avg_metrics, all_metrics = test_model(model, test_loader, device, save_dir='./results')

print('\n測試集平均指標:')
print('-' * 30)
for key, value in avg_metrics.items():
    print(f'{key.capitalize():12s}: {value:.4f}')

# 儲存測試結果
with open('results/test_metrics.json', 'w') as f:
    json.dump({
        'average': avg_metrics,
        'all_samples': all_metrics
    }, f, indent=2)

# ============================================================================
# Cell 12: 視覺化預測結果
# ============================================================================
print('產生預測視覺化...')

test_dataset = BrainTumorDataset(
    TEST_DIR, 
    TEST_ANN,
    transform=ValidTransform()
)

visualize_predictions(
    model, 
    test_dataset, 
    device, 
    num_samples=8,
    save_path='results/predictions.png'
)

# ============================================================================
# Cell 13: 顯示評估指標的統計
# ============================================================================
import pandas as pd

# 建立DataFrame
df_metrics = pd.DataFrame(all_metrics)

print('測試集指標統計:')
print('='*50)
print(df_metrics.describe())

# 繪製指標分布
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

metrics_to_plot = ['dice', 'iou', 'precision', 'recall', 'f1']

for i, metric in enumerate(metrics_to_plot):
    axes[i].hist(df_metrics[metric], bins=20, edgecolor='black', alpha=0.7)
    axes[i].axvline(df_metrics[metric].mean(), color='red', 
                    linestyle='--', label=f'Mean: {df_metrics[metric].mean():.3f}')
    axes[i].set_xlabel(metric.capitalize())
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{metric.capitalize()} Distribution')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

# 移除最後一個空的subplot
fig.delaxes(axes[-1])

plt.tight_layout()
plt.savefig('results/metrics_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# Cell 14: 儲存最終報告用的關鍵資訊
# ============================================================================
final_results = {
    'model': 'U-Net',
    'dataset': {
        'train_samples': len(train_loader.dataset),
        'valid_samples': len(valid_loader.dataset),
        'test_samples': len(test_loader.dataset),
    },
    'training': {
        'epochs_trained': len(history['train_loss']),
        'best_epoch': history['valid_dice'].index(max(history['valid_dice'])) + 1,
        'best_valid_dice': best_dice,
    },
    'test_results': avg_metrics,
    'hyperparameters': CONFIG
}

with open('results/final_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print('='*50)
print('所有結果已儲存至 results/ 目錄')
print('='*50)
print('\n檔案清單:')
print('  - results/training_history.json')
print('  - results/training_curves.png')
print('  - results/test_metrics.json')
print('  - results/predictions.png')
print('  - results/metrics_distribution.png')
print('  - results/final_results.json')
print('  - models/best_model.pth')
