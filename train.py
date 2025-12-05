"""
主訓練腳本 - 腦腫瘤分割
執行此腳本來訓練模型
"""

# 解決 Windows 上的 OpenMP 衝突問題
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from pathlib import Path

# 匯入自訂模組
from brain_tumor_segmentation import (
    create_dataloaders,
    UNet,
    CombinedLoss,
    train_model,
    test_model,
    visualize_predictions,
    plot_training_history,
    BrainTumorDataset,
    ValidTransform
)


def main():
    # =============================================================================
    # 1. 設定參數
    # =============================================================================
    
    # 資料路徑
    BASE_DIR = Path('.')
    TRAIN_DIR = BASE_DIR / 'train'
    VALID_DIR = BASE_DIR / 'valid'
    TEST_DIR = BASE_DIR / 'test'
    
    TRAIN_ANN = TRAIN_DIR / '_annotations.coco.json'
    VALID_ANN = VALID_DIR / '_annotations.coco.json'
    TEST_ANN = TEST_DIR / '_annotations.coco.json'
    
    # 訓練參數
    BATCH_SIZE = 1  # GTX 960 (4GB) 最保守設定，確保穩定訓練
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    PATIENCE = 15
    NUM_WORKERS = 0  # Windows 系統建議設為 0，避免多進程問題
    
    # 裝置設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用裝置: {device}')
    
    # 建立儲存目錄
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # =============================================================================
    # 2. 載入資料
    # =============================================================================
    
    print('\n' + '='*50)
    print('載入資料...')
    print('='*50)
    
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_dir=TRAIN_DIR,
        valid_dir=VALID_DIR,
        test_dir=TEST_DIR,
        train_ann=TRAIN_ANN,
        valid_ann=VALID_ANN,
        test_ann=TEST_ANN,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    print(f'訓練集樣本數: {len(train_loader.dataset)}')
    print(f'驗證集樣本數: {len(valid_loader.dataset)}')
    print(f'測試集樣本數: {len(test_loader.dataset)}')
    
    # =============================================================================
    # 3. 建立模型
    # =============================================================================
    
    print('\n' + '='*50)
    print('建立模型...')
    print('='*50)
    
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'總參數量: {total_params:,}')
    print(f'可訓練參數量: {trainable_params:,}')
    
    # =============================================================================
    # 4. 設定損失函數、優化器、學習率調整器
    # =============================================================================
    
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',
        patience=5,
        factor=0.5,
        verbose=True
    )
    
    # =============================================================================
    # 5. 訓練模型
    # =============================================================================
    
    print('\n' + '='*50)
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
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        save_dir='./models'
    )
    
    print(f'\n訓練完成！最佳驗證 Dice Score: {best_dice:.4f}')
    
    # 儲存訓練歷史
    with open('results/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # 繪製訓練曲線
    plot_training_history(history, save_path='results/training_curves.png')
    
    # =============================================================================
    # 6. 載入最佳模型並在測試集上評估
    # =============================================================================
    
    print('\n' + '='*50)
    print('在測試集上評估...')
    print('='*50)
    
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
    
    # =============================================================================
    # 7. 視覺化預測結果
    # =============================================================================
    
    print('\n' + '='*50)
    print('產生預測視覺化...')
    print('='*50)
    
    test_dataset = BrainTumorDataset(
        TEST_DIR, 
        TEST_ANN,
        transform=ValidTransform()
    )
    
    visualize_predictions(
        model, 
        test_dataset, 
        device, 
        num_samples=5,
        save_path='results/predictions.png'
    )
    
    print('\n所有結果已儲存至 results/ 目錄')
    print('=' * 50)
    print('程式執行完畢！')
    print('=' * 50)


if __name__ == '__main__':
    main()
