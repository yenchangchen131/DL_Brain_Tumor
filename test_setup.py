"""
驗證程式碼完整性的測試腳本
執行此腳本來檢查所有模組是否可以正確載入
"""

import sys
from pathlib import Path

def test_imports():
    """測試所有必要的套件是否可以匯入"""
    print("="*60)
    print("測試套件匯入...")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch 未安裝")
        return False
    
    try:
        import torchvision
        print(f"✓ torchvision {torchvision.__version__}")
    except ImportError:
        print("✗ torchvision 未安裝")
        return False
    
    try:
        import albumentations
        print(f"✓ albumentations {albumentations.__version__}")
    except ImportError:
        print("✗ albumentations 未安裝")
        print("  請執行: pip install albumentations")
        return False
    
    try:
        import cv2
        print(f"✓ opencv-python {cv2.__version__}")
    except ImportError:
        print("✗ opencv-python 未安裝")
        print("  請執行: pip install opencv-python")
        return False
    
    try:
        import numpy as np
        print(f"✓ numpy {np.__version__}")
    except ImportError:
        print("✗ numpy 未安裝")
        return False
    
    try:
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__}")
    except ImportError:
        print("✗ matplotlib 未安裝")
        return False
    
    try:
        from tqdm import tqdm
        print(f"✓ tqdm")
    except ImportError:
        print("✗ tqdm 未安裝")
        print("  請執行: pip install tqdm")
        return False
    
    return True


def test_module_import():
    """測試自訂模組是否可以匯入"""
    print("\n" + "="*60)
    print("測試自訂模組匯入...")
    print("="*60)
    
    try:
        from brain_tumor_segmentation import (
            UNet,
            DiceLoss,
            CombinedLoss,
            BrainTumorDataset,
            TrainTransform,
            ValidTransform,
            EarlyStopping,
            train_one_epoch,
            validate,
            test_model,
            calculate_metrics,
            visualize_predictions,
            plot_training_history
        )
        print("✓ brain_tumor_segmentation.py - 所有功能成功匯入")
        return True
    except Exception as e:
        print(f"✗ brain_tumor_segmentation.py - 匯入失敗: {e}")
        return False


def test_model_creation():
    """測試模型是否可以成功建立"""
    print("\n" + "="*60)
    print("測試模型建立...")
    print("="*60)
    
    try:
        import torch
        from brain_tumor_segmentation import UNet
        
        model = UNet(in_channels=3, out_channels=1)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ U-Net模型建立成功")
        print(f"  總參數量: {total_params:,}")
        
        # 測試前向傳播
        test_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model(test_input)
        print(f"  輸入形狀: {test_input.shape}")
        print(f"  輸出形狀: {output.shape}")
        
        if output.shape == (1, 1, 640, 640):
            print("✓ 模型前向傳播正常")
            return True
        else:
            print("✗ 輸出形狀不正確")
            return False
            
    except Exception as e:
        print(f"✗ 模型建立失敗: {e}")
        return False


def test_loss_functions():
    """測試損失函數是否可以正常運作"""
    print("\n" + "="*60)
    print("測試損失函數...")
    print("="*60)
    
    try:
        import torch
        from brain_tumor_segmentation import DiceLoss, CombinedLoss
        
        # 建立測試資料
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        # 測試Dice Loss
        dice_loss = DiceLoss()
        loss_value = dice_loss(pred, target)
        print(f"✓ Dice Loss: {loss_value.item():.4f}")
        
        # 測試Combined Loss
        combined_loss = CombinedLoss()
        loss_value = combined_loss(pred, target)
        print(f"✓ Combined Loss: {loss_value.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 損失函數測試失敗: {e}")
        return False


def check_data_structure():
    """檢查資料結構是否正確"""
    print("\n" + "="*60)
    print("檢查資料結構...")
    print("="*60)
    
    base_dir = Path('.')
    
    # 檢查資料夾
    folders = ['train', 'valid', 'test']
    all_exist = True
    
    for folder in folders:
        folder_path = base_dir / folder
        if folder_path.exists():
            # 檢查annotation檔案
            ann_file = folder_path / '_annotations.coco.json'
            if ann_file.exists():
                print(f"✓ {folder}/ - 資料夾和annotation檔案存在")
            else:
                print(f"⚠ {folder}/ - 資料夾存在但缺少 _annotations.coco.json")
                all_exist = False
        else:
            print(f"✗ {folder}/ - 資料夾不存在")
            all_exist = False
    
    return all_exist


def main():
    """執行所有測試"""
    print("\n" + "="*60)
    print("腦腫瘤影像分割專案 - 完整性檢查")
    print("="*60)
    
    results = []
    
    # 執行測試
    results.append(("套件匯入", test_imports()))
    results.append(("模組匯入", test_module_import()))
    results.append(("模型建立", test_model_creation()))
    results.append(("損失函數", test_loss_functions()))
    results.append(("資料結構", check_data_structure()))
    
    # 總結
    print("\n" + "="*60)
    print("測試總結")
    print("="*60)
    
    for name, passed in results:
        status = "✓ 通過" if passed else "✗ 失敗"
        print(f"{name:12s}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ 所有測試通過！您可以開始訓練了。")
        print("\n執行以下命令開始訓練:")
        print("  python train.py")
    else:
        print("✗ 部分測試失敗，請檢查上述錯誤訊息。")
        print("\n如果缺少套件，請執行:")
        print("  pip install -r requirements.txt")
    print("="*60)


if __name__ == '__main__':
    main()
