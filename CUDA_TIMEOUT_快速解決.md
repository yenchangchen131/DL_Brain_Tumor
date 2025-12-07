# CUDA Timeout 錯誤 - 快速解決指南

## 錯誤訊息
```
RuntimeError: CUDA error: the launch timed out and was terminated
```

---

## 🚨 問題原因

**Windows TDR (Timeout Detection and Recovery) 機制**
- Windows 預設 GPU 計算超過 **2 秒** 就會強制中斷
- 您的 GTX 960 (4GB) 處理 640x640 圖像時，單個 batch 的計算時間超過這個限制
- 即使 batch size 已經降到 1，仍然會超時

---

## ✅ 解決方案（三選一）

### 方案 1️⃣: 降低圖像解析度（最簡單，強烈推薦）

我已經為您創建了兩個修改版本的 notebook：

#### **選項 A: 使用 448x448 （推薦）**
- 檔案: `brain_tumor_complete_size448.ipynb`
- 速度提升: **約 2 倍**
- 計算量減少: **51%**
- 可以提高 batch size 到 2-4

#### **選項 B: 使用 512x512**
- 檔案: `brain_tumor_complete_size512.ipynb`
- 速度提升: **約 1.6 倍**
- 計算量減少: **36%**
- 可以提高 batch size 到 2

#### 📌 使用步驟：

1. **打開修改後的 notebook**
   ```bash
   jupyter notebook brain_tumor_complete_size448.ipynb
   ```

2. **重新運行所有 cells** (從頭開始)
   - 點擊 `Cell` → `Run All`
   - 或按 `Shift + Enter` 逐個執行

3. **開始訓練**
   - 應該不會再出現 CUDA timeout 錯誤
   - 訓練速度會明顯變快

4. **（可選）調整 batch size**
   
   修改這個 cell：
   ```python
   BATCH_SIZE = 2  # 從 1 改為 2 或 4（448 版本可用 4）
   ```

---

### 方案 2️⃣: 修改 Windows TDR 設定（永久解決）

如果您想保持 640x640 解析度，可以修改 Windows 的 GPU 超時限制。

#### 步驟：

1. **按 `Win + R`**，輸入 `regedit`，按 Enter

2. **導航到**：
   ```
   HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers
   ```

3. **右鍵點擊右側空白處** → 新增 → DWORD (32位) 值

4. **創建兩個值**：

   | 名稱 | 數值資料 (十進位) |
   |------|-----------------|
   | `TdrDelay` | `60` |
   | `TdrLevel` | `0` |

5. **重啟電腦** ⚠️ 必須重啟才會生效

#### ⚠️ 注意：
- 這會禁用 GPU 超時保護
- 如果 GPU 掛起，可能導致系統無響應
- 訓練完成後建議恢復設定

詳細說明請參考: `fix_cuda_timeout.md`

---

### 方案 3️⃣: 使用雲端 GPU（最快速度）

#### Google Colab (免費)
- GPU: Tesla T4 (16GB)
- 速度: 比 GTX 960 快 **5-10 倍**
- 限制: 單次最多 12 小時

#### Kaggle (免費)
- GPU: P100 (16GB)
- 速度: 比 GTX 960 快 **5-10 倍**
- 限制: 每週 30 小時

上傳您的資料集和 notebook，即可開始訓練。

---

## 🎯 我的建議

### **最佳方案組合：**

1. **先試方案 1（降低解析度）**
   - 使用 `brain_tumor_complete_size448.ipynb`
   - 設定 `BATCH_SIZE = 2` 或 `4`
   - 應該能成功訓練且速度快

2. **如果還想要 640 解析度**
   - 使用方案 2（修改 TDR）
   - 或使用方案 3（雲端 GPU）

---

## 📊 性能對比表

| 解析度 | 計算量 | 相對速度 | 推薦 Batch Size | CUDA Timeout 風險 |
|--------|--------|----------|----------------|------------------|
| 640x640 | 409,600 | 1.0x (基準) | 1 | ⚠️ 高 |
| 512x512 | 262,144 | ~1.6x | 2 | ⚠️ 中 |
| **448x448** | 200,704 | **~2.0x** | **2-4** | ✅ **低** |

---

## 🔍 如何驗證已修復

運行訓練後，應該看到：

```
Epoch 1/100
Training:   0%|          | 0/1504 [00:00<?, ?it/s]
Training:   1%|▏         | 15/1504 [03:45<6:14:21, 15.08s/it, loss=0.7542, dice=0.0234]
```

**✅ 成功的標誌：**
- 進度條持續更新
- 沒有出現 CUDA error
- 每個 iteration 時間在 10-20 秒左右（448 版本）

**❌ 如果還是失敗：**
- 檢查是否真的使用了修改後的 notebook
- 確認已重新運行所有 cells
- 考慮使用方案 2 或 3

---

## 📝 快速操作清單

- [ ] 打開 `brain_tumor_complete_size448.ipynb`
- [ ] 重新運行所有 cells (Cell → Run All)
- [ ] 等待訓練開始
- [ ] 檢查是否正常運行（無 CUDA error）
- [ ] （可選）調整 BATCH_SIZE 為 2 或 4
- [ ] 讓訓練持續運行

---

## 📞 如果仍有問題

請提供以下資訊：
1. 使用哪個版本的 notebook (448 還是 512)
2. BATCH_SIZE 設定為多少
3. 完整的錯誤訊息
4. GPU 使用情況 (運行 `nvidia-smi`)

---

## 總結

**最簡單的解決方式：**
```bash
# 1. 打開修改後的 notebook
jupyter notebook brain_tumor_complete_size448.ipynb

# 2. 在 notebook 中運行所有 cells
# 3. 開始訓練！
```

**預期結果：**
- ✅ 不再有 CUDA timeout
- ✅ 訓練速度提升 2 倍
- ✅ 可以使用更大的 batch size
- ⚡ 總訓練時間從 5-15 天減少到 2-7 天

---

**祝訓練成功！** 🎉
