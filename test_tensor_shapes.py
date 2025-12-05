"""
快速測試損失函數和形狀處理
"""
import torch
from brain_tumor_segmentation import CombinedLoss, DiceLoss, dice_coefficient, iou_score, calculate_metrics

# 創建測試數據
# 模擬模型輸出（4D）和 albumentations 產生的遮罩（3D，uint8）
batch_size = 2
height, width = 640, 640

# 模擬模型輸出（4D float）
pred = torch.randn(batch_size, 1, height, width)

# 模擬 albumentations ToTensorV2 產生的遮罩（3D uint8）
target_3d_uint8 = torch.randint(0, 2, (batch_size, height, width), dtype=torch.uint8)

# 模擬標準遮罩（4D float）
target_4d_float = torch.randint(0, 2, (batch_size, 1, height, width), dtype=torch.float32)

print("=" * 60)
print("測試損失函數和指標的形狀/類型處理")
print("=" * 60)

# 測試 DiceLoss
print("\n1. 測試 DiceLoss")
dice_loss = DiceLoss()

try:
    loss1 = dice_loss(pred, target_3d_uint8)
    print(f"   ✓ 3D uint8 target: loss = {loss1.item():.4f}")
except Exception as e:
    print(f"   ✗ 3D uint8 target 失敗: {e}")

try:
    loss2 = dice_loss(pred, target_4d_float)
    print(f"   ✓ 4D float target: loss = {loss2.item():.4f}")
except Exception as e:
    print(f"   ✗ 4D float target 失敗: {e}")

# 測試 CombinedLoss
print("\n2. 測試 CombinedLoss")
combined_loss = CombinedLoss()

try:
    loss1 = combined_loss(pred, target_3d_uint8)
    print(f"   ✓ 3D uint8 target: loss = {loss1.item():.4f}")
except Exception as e:
    print(f"   ✗ 3D uint8 target 失敗: {e}")

try:
    loss2 = combined_loss(pred, target_4d_float)
    print(f"   ✓ 4D float target: loss = {loss2.item():.4f}")
except Exception as e:
    print(f"   ✗ 4D float target 失敗: {e}")

# 測試評估指標
print("\n3. 測試評估指標")

try:
    dice = dice_coefficient(pred, target_3d_uint8)
    print(f"   ✓ dice_coefficient (3D uint8): {dice:.4f}")
except Exception as e:
    print(f"   ✗ dice_coefficient 失敗: {e}")

try:
    iou = iou_score(pred, target_3d_uint8)
    print(f"   ✓ iou_score (3D uint8): {iou:.4f}")
except Exception as e:
    print(f"   ✗ iou_score 失敗: {e}")

try:
    metrics = calculate_metrics(pred, target_3d_uint8)
    print(f"   ✓ calculate_metrics (3D uint8):")
    for key, value in metrics.items():
        print(f"      {key}: {value:.4f}")
except Exception as e:
    print(f"   ✗ calculate_metrics 失敗: {e}")

print("\n" + "=" * 60)
print("所有測試完成！")
print("=" * 60)
