# CUDA 超時錯誤修復指南

## 問題
```
RuntimeError: CUDA error: the launch timed out and was terminated
```

這是 Windows TDR (Timeout Detection and Recovery) 機制導致的。當 GPU 運算超過 2 秒（預設值），Windows 會強制中斷。

---

## 解決方案

### 🔧 方案 1: 修改 Windows 註冊表（推薦，永久生效）

**步驟 1: 備份註冊表**
- 按 `Win + R`，輸入 `regedit`，按 Enter
- 在左側導航樹中找到要修改的位置，右鍵點擊 → 匯出 → 保存備份

**步驟 2: 修改 TDR 設定**

1. 按 `Win + R`，輸入 `regedit`，按 Enter
2. 導航到：
   ```
   HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers
   ```

3. 在右側窗格中，右鍵 → 新增 → DWORD (32位) 值

4. 創建以下兩個值：

   | 名稱 | 類型 | 數值 | 說明 |
   |------|------|------|------|
   | `TdrDelay` | DWORD | `60` (十進位) | GPU 超時延遲（秒）|
   | `TdrLevel` | DWORD | `0` (十進位) | 禁用 TDR (0=禁用) |

   **或者更保守的設定：**
   - `TdrDelay`: `10` (10秒超時)
   - `TdrLevel`: `3` (僅恢復，不重啟驅動)

**步驟 3: 重啟電腦**

修改後**必須重啟電腦**才會生效。

---

### 🔧 方案 2: 使用 PowerShell 腳本（需要管理員權限）

創建一個 PowerShell 腳本來自動設定：

```powershell
# 需要以管理員身份運行
# 修改 TDR 設定
$regPath = "HKLM:\System\CurrentControlSet\Control\GraphicsDrivers"

# 檢查註冊表路徑是否存在
if (!(Test-Path $regPath)) {
    New-Item -Path $regPath -Force
}

# 設定 TDR 延遲為 60 秒
Set-ItemProperty -Path $regPath -Name "TdrDelay" -Value 60 -Type DWord

# 禁用 TDR（慎用！）
Set-ItemProperty -Path $regPath -Name "TdrLevel" -Value 0 -Type DWord

Write-Host "TDR 設定已修改！" -ForegroundColor Green
Write-Host "TdrDelay = 60 秒" -ForegroundColor Yellow
Write-Host "TdrLevel = 0 (禁用)" -ForegroundColor Yellow
Write-Host "" 
Write-Host "請重啟電腦以使設定生效！" -ForegroundColor Red
```

保存為 `fix_tdr.ps1`，然後以管理員身份運行：
```powershell
# 以管理員身份打開 PowerShell，然後運行：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\fix_tdr.ps1
```

---

### 🔧 方案 3: 程式碼層級的解決方案（不修改系統設定）

如果不想修改註冊表，可以：

#### A. 降低影像解析度
修改 notebook 中的圖像大小：

```python
class TrainTransform:
    def __init__(self, image_size=512):  # 從 640 改為 512 或 448
        # ...

class ValidTransform:
    def __init__(self, image_size=512):  # 從 640 改為 512 或 448
        # ...
```

**效果**：
- 640 → 512: 速度提升約 1.5 倍
- 640 → 448: 速度提升約 2 倍
- 可以使用更大的 batch size (2 或 4)

#### B. 使用梯度累積（模擬更大的 batch size）

```python
def train_one_epoch(model, loader, criterion, optimizer, device, accumulation_steps=4):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc='Training')
    for i, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # 前向傳播
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # 正規化 loss（平均多個 batch）
        loss = loss / accumulation_steps
        
        # 反向傳播
        loss.backward()
        
        # 每 accumulation_steps 次才更新參數
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # 計算指標
        dice = dice_coefficient(outputs, masks)
        
        running_loss += loss.item() * accumulation_steps
        running_dice += dice
        
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'dice': f'{dice:.4f}'
        })
    
    epoch_loss = running_loss / len(loader)
    epoch_dice = running_dice / len(loader)
    
    return epoch_loss, epoch_dice
```

#### C. 添加 GPU 同步點

在訓練循環中添加：

```python
# 在每個 batch 後
torch.cuda.synchronize()
```

---

### 🔧 方案 4: 使用 Google Colab / Kaggle（最簡單）

**Google Colab:**
- 免費 T4 GPU (16GB)
- 無需擔心 TDR
- 訓練速度快 5-10 倍

**Kaggle:**
- 免費 P100 GPU (16GB)
- 每週 30 小時 GPU 配額
- 訓練速度快 5-10 倍

---

## 建議的操作順序

### 🎯 推薦方案（依優先順序）：

1. **先嘗試方案 3A**（降低解析度）
   - 最簡單，無需系統更改
   - 修改 `image_size=512` 或 `448`
   
2. **如果還是超時，使用方案 1**（修改 TDR）
   - 永久解決問題
   - 需要重啟電腦

3. **如果不想修改系統，使用方案 4**（Colab/Kaggle）
   - 最快的訓練速度
   - 無需擔心硬體限制

---

## 檢查 TDR 當前設定

執行此 PowerShell 命令查看當前設定：

```powershell
Get-ItemProperty -Path "HKLM:\System\CurrentControlSet\Control\GraphicsDrivers" -Name TdrDelay -ErrorAction SilentlyContinue
Get-ItemProperty -Path "HKLM:\System\CurrentControlSet\Control\GraphicsDrivers" -Name TdrLevel -ErrorAction SilentlyContinue
```

如果顯示「找不到」，表示使用預設值（2秒超時）。

---

## 恢復原始設定

如果想要恢復：

```powershell
# 以管理員身份運行
$regPath = "HKLM:\System\CurrentControlSet\Control\GraphicsDrivers"
Remove-ItemProperty -Path $regPath -Name "TdrDelay" -ErrorAction SilentlyContinue
Remove-ItemProperty -Path $regPath -Name "TdrLevel" -ErrorAction SilentlyContinue

Write-Host "TDR 設定已恢復為預設值！" -ForegroundColor Green
Write-Host "請重啟電腦！" -ForegroundColor Yellow
```

---

## ⚠️ 注意事項

1. **修改 TDR 的風險**：
   - 禁用 TDR 可能導致 GPU 掛起時系統無響應
   - 建議只在訓練時禁用，訓練完成後恢復

2. **備份很重要**：
   - 修改註冊表前請先備份
   - 記錄原始值以便恢復

3. **重啟必要**：
   - 修改註冊表後必須重啟電腦

---

## 相關資訊

- [Microsoft TDR 文檔](https://docs.microsoft.com/en-us/windows-hardware/drivers/display/timeout-detection-and-recovery)
- [NVIDIA CUDA Timeout 問題](https://forums.developer.nvidia.com/t/cuda-kernel-timeout/37732)
