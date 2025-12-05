"""
Brain Tumor Segmentation - Complete Implementation
包含資料增強、DataLoader、模型、訓練、評估等所有功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

import numpy as np
import cv2
from PIL import Image
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2


# =============================================================================
# 1. 資料增強與預處理
# =============================================================================

class TrainTransform:
    """訓練集的資料增強"""
    def __init__(self, image_size=640):
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], is_check_shapes=False)
    
    def __call__(self, image, mask):
        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']


class ValidTransform:
    """驗證集/測試集的資料轉換（不增強）"""
    def __init__(self, image_size=640):
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], is_check_shapes=False)
    
    def __call__(self, image, mask):
        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']


# =============================================================================
# 2. Dataset & DataLoader
# =============================================================================

class BrainTumorDataset(Dataset):
    """腦腫瘤影像分割資料集"""
    
    def __init__(self, image_dir, annotation_file, transform=None):
        """
        Args:
            image_dir: 影像資料夾路徑
            annotation_file: COCO格式的annotation JSON檔案
            transform: 資料增強/轉換函數
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # 載入COCO annotations
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)
        
        # 建立image id到檔名的映射
        self.images = {img['id']: img for img in self.coco_data['images']}
        
        # 建立image id到annotations的映射
        self.image_to_anns = defaultdict(list)
        for ann in self.coco_data['annotations']:
            self.image_to_anns[ann['image_id']].append(ann)
        
        # 取得所有image ids
        self.image_ids = list(self.images.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # 取得image資訊
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        
        # 載入影像
        image_path = self.image_dir / image_info['file_name']
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 建立mask
        height, width = image_info['height'], image_info['width']
        mask = self._create_mask(image_id, height, width)
        
        # 應用轉換
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            # 如果沒有transform，至少轉成tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask
    
    def _create_mask(self, image_id, height, width):
        """從COCO segmentation建立binary mask"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 取得該影像的所有annotations
        annotations = self.image_to_anns[image_id]
        
        for ann in annotations:
            if 'segmentation' in ann:
                # COCO格式的segmentation可能是polygon或RLE
                if isinstance(ann['segmentation'], list):
                    # Polygon格式
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask, [poly], 1)
                # 如果是RLE格式，這裡需要額外處理（略）
        
        return mask


def create_dataloaders(train_dir, valid_dir, test_dir, 
                       train_ann, valid_ann, test_ann,
                       batch_size=8, num_workers=4):
    """建立訓練、驗證、測試的DataLoader"""
    
    # 建立datasets
    train_dataset = BrainTumorDataset(
        train_dir, train_ann, 
        transform=TrainTransform()
    )
    
    valid_dataset = BrainTumorDataset(
        valid_dir, valid_ann,
        transform=ValidTransform()
    )
    
    test_dataset = BrainTumorDataset(
        test_dir, test_ann,
        transform=ValidTransform()
    )
    
    # 建立dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader


# =============================================================================
# 3. 損失函數
# =============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # 確保形狀匹配
        if pred.dim() == 4 and target.dim() == 3:
            target = target.unsqueeze(1)
        
        # 確保類型正確
        target = target.float()
        
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """組合損失: Dice Loss + Binary Cross Entropy"""
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred, target):
        # 確保 target 和 pred 的形狀匹配
        # pred: [B, 1, H, W], target可能是 [B, H, W] 或 [B, 1, H, W]
        if pred.dim() == 4 and target.dim() == 3:
            target = target.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        
        # 確保 target 是 float 類型
        target = target.float()
        
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


# =============================================================================
# 4. U-Net 模型
# =============================================================================

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net架構用於影像分割"""
    
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder (下採樣)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder (上採樣)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))
        
        # 最終輸出層
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            # 處理尺寸不匹配的情況
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        
        return self.final_conv(x)


# =============================================================================
# 5. 評估指標
# =============================================================================

def dice_coefficient(pred, target, threshold=0.5):
    """計算Dice係數"""
    # 確保形狀匹配
    if pred.dim() == 4 and target.dim() == 3:
        target = target.unsqueeze(1)
    
    # 確保類型正確
    target = target.float()
    
    pred = (torch.sigmoid(pred) > threshold).float()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-8)
    
    return dice.item()


def iou_score(pred, target, threshold=0.5):
    """計算IoU (Intersection over Union)"""
    # 確保形狀匹配
    if pred.dim() == 4 and target.dim() == 3:
        target = target.unsqueeze(1)
    
    # 確保類型正確
    target = target.float()
    
    pred = (torch.sigmoid(pred) > threshold).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-8)
    
    return iou.item()


def calculate_metrics(pred, target, threshold=0.5):
    """計算所有評估指標"""
    # 確保形狀匹配
    if pred.dim() == 4 and target.dim() == 3:
        target = target.unsqueeze(1)
    
    # 確保類型正確
    target = target.float()
    
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    
    # True Positives, False Positives, False Negatives
    tp = (pred_binary * target).sum().item()
    fp = (pred_binary * (1 - target)).sum().item()
    fn = ((1 - pred_binary) * target).sum().item()
    
    # Precision, Recall, F1
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Dice & IoU
    dice = dice_coefficient(pred, target, threshold)
    iou = iou_score(pred, target, threshold)
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# =============================================================================
# 6. 訓練與驗證
# =============================================================================

class EarlyStopping:
    """Early Stopping來防止過擬合"""
    
    def __init__(self, patience=15, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        elif self.mode == 'min':
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0


def train_one_epoch(model, loader, criterion, optimizer, device):
    """訓練一個epoch"""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    
    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # 前向傳播
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 計算指標
        dice = dice_coefficient(outputs, masks)
        
        running_loss += loss.item()
        running_dice += dice
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}'
        })
    
    epoch_loss = running_loss / len(loader)
    epoch_dice = running_dice / len(loader)
    
    return epoch_loss, epoch_dice


def validate(model, loader, criterion, device):
    """驗證"""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            dice = dice_coefficient(outputs, masks)
            
            running_loss += loss.item()
            running_dice += dice
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}'
            })
    
    epoch_loss = running_loss / len(loader)
    epoch_dice = running_dice / len(loader)
    
    return epoch_loss, epoch_dice


def train_model(model, train_loader, valid_loader, criterion, optimizer, 
                scheduler, device, num_epochs=100, patience=15, 
                save_dir='./models'):
    """完整的訓練流程"""
    
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    early_stopping = EarlyStopping(patience=patience, mode='max')
    
    history = {
        'train_loss': [],
        'train_dice': [],
        'valid_loss': [],
        'valid_dice': [],
        'lr': []
    }
    
    best_dice = 0.0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 50)
        
        # 訓練
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 驗證
        valid_loss, valid_dice = validate(
            model, valid_loader, criterion, device
        )
        
        # 學習率調整
        scheduler.step(valid_dice)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 記錄
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['valid_loss'].append(valid_loss)
        history['valid_dice'].append(valid_dice)
        history['lr'].append(current_lr)
        
        print(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}, Valid Dice: {valid_dice:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # 儲存最佳模型
        if valid_dice > best_dice:
            best_dice = valid_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, best_model_path)
            print(f'✓ Saved best model (Dice: {best_dice:.4f})')
        
        # Early Stopping
        early_stopping(valid_dice)
        if early_stopping.early_stop:
            print(f'\nEarly stopping triggered at epoch {epoch + 1}')
            break
    
    return history, best_dice


# =============================================================================
# 7. 測試與視覺化
# =============================================================================

def test_model(model, test_loader, device, save_dir='./results'):
    """在測試集上評估模型"""
    model.eval()
    
    all_metrics = []
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc='Testing')):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            metrics = calculate_metrics(outputs, masks)
            all_metrics.append(metrics)
    
    # 計算平均指標
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    return avg_metrics, all_metrics


def visualize_predictions(model, dataset, device, num_samples=5, save_path=None):
    """視覺化預測結果"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    indices = random.sample(range(len(dataset)), num_samples)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            
            # 預測
            image_input = image.unsqueeze(0).to(device)
            output = model(image_input)
            pred_mask = torch.sigmoid(output).cpu().squeeze().numpy()
            
            # 反正規化影像以供顯示
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image_np = np.clip(image_np, 0, 1)
            
            mask_np = mask.cpu().squeeze().numpy()
            
            # 畫圖
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
            
            # Overlay
            axes[i, 3].imshow(image_np)
            axes[i, 3].imshow(pred_mask, cmap='jet', alpha=0.5)
            axes[i, 3].set_title('Overlay')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_history(history, save_path=None):
    """繪製訓練歷史"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss curve
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['valid_loss'], label='Valid Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Dice curve
    axes[1].plot(history['train_dice'], label='Train Dice')
    axes[1].plot(history['valid_dice'], label='Valid Dice')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Training and Validation Dice Score')
    axes[1].legend()
    axes[1].grid(True)
    
    # Learning rate
    axes[2].plot(history['lr'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
