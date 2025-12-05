"""
檢查 GPU 和 CUDA 設定
"""
import torch
import sys

print("=" * 60)
print("PyTorch 和 CUDA 檢查")
print("=" * 60)

# 1. PyTorch 版本
print(f"\n1. PyTorch 版本: {torch.__version__}")

# 2. CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"\n2. CUDA 是否可用: {cuda_available}")

# 3. CUDA 版本
if cuda_available:
    print(f"   CUDA 版本: {torch.version.cuda}")
    print(f"   cuDNN 版本: {torch.backends.cudnn.version()}")
else:
    print(f"   PyTorch 編譯時的 CUDA 版本: {torch.version.cuda}")
    print("   [警告] CUDA 不可用！")

# 4. GPU 設備數量
print(f"\n3. GPU 設備數量: {torch.cuda.device_count()}")

# 5. GPU 設備資訊
if torch.cuda.device_count() > 0:
    for i in range(torch.cuda.device_count()):
        print(f"\n   GPU {i}:")
        print(f"   - 名稱: {torch.cuda.get_device_name(i)}")
        print(f"   - 記憶體總量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"   - 計算能力: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")

# 6. 當前設備
device = torch.device('cuda' if cuda_available else 'cpu')
print(f"\n4. 當前使用的設備: {device}")

# 7. 測試 GPU 運算
print(f"\n5. 測試 GPU 運算:")
try:
    if cuda_available:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("   [成功] GPU 運算測試成功！")
    else:
        print("   [跳過] 無法測試：CUDA 不可用")
except Exception as e:
    print(f"   [失敗] GPU 運算測試失敗: {e}")

# 8. 檢查 nvidia-smi
print(f"\n6. 檢查 NVIDIA 驅動:")
import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("   [成功] NVIDIA 驅動已安裝")
        print("\n" + "="*60)
        print("nvidia-smi 輸出:")
        print("="*60)
        print(result.stdout)
    else:
        print("   [失敗] nvidia-smi 執行失敗")
except FileNotFoundError:
    print("   [失敗] 找不到 nvidia-smi (NVIDIA 驅動可能未安裝)")
except Exception as e:
    print(f"   [錯誤] 執行 nvidia-smi 時發生錯誤: {e}")

print("\n" + "=" * 60)
print("診斷完成")
print("=" * 60)

# 給出建議
print("\n建議:")
if not cuda_available:
    print("[警告] CUDA 不可用，可能的原因：")
    print("   1. PyTorch 安裝的不是 CUDA 版本 (可能是 CPU-only 版本)")
    print("   2. NVIDIA 驅動未正確安裝")
    print("   3. CUDA toolkit 版本不匹配")
    print("\n解決方法：")
    print("   1. 先執行 nvidia-smi 檢查您的 GPU 和 CUDA 版本")
    print("   2. 重新安裝 PyTorch (CUDA 版本)")
    print("   3. 參考: https://pytorch.org/get-started/locally/")
    print("\n重新安裝指令範例 (請根據您的 CUDA 版本調整):")
    print("   # CUDA 11.8")
    print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\n   # CUDA 12.1")  
    print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
else:
    print("[成功] GPU 設定正常，可以使用 CUDA 加速訓練！")
