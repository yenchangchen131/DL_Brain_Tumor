"""
修改 Jupyter Notebook 的圖像大小以解決 CUDA Timeout 問題

這個腳本會將 brain_tumor_complete.ipynb 中的圖像大小從 640 改為 448 或 512
"""

import json
import sys

def modify_notebook_image_size(notebook_path, new_size=448):
    """
    修改 notebook 中的圖像大小
    
    Args:
        notebook_path: notebook 檔案路徑
        new_size: 新的圖像大小 (建議 448 或 512)
    """
    print(f"[INFO] 正在修改 notebook: {notebook_path}")
    print(f"[INFO] 新的圖像大小: {new_size}x{new_size}")
    
    # 讀取 notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    modifications = 0
    
    # 遍歷所有 cells
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            # 檢查並修改 source
            modified = False
            new_source = []
            
            for line in cell['source']:
                original_line = line
                
                # 修改 TrainTransform 和 ValidTransform 的 image_size
                if 'def __init__(self, image_size=' in line and 'image_size=640' in line:
                    line = line.replace('image_size=640', f'image_size={new_size}')
                    modified = True
                    modifications += 1
                    print(f"  [OK] 修改: {original_line.strip()[:50]}...")
                
                new_source.append(line)
            
            if modified:
                cell['source'] = new_source
    
    # 保存修改後的 notebook
    backup_path = notebook_path.replace('.ipynb', f'_size{new_size}.ipynb')
    
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUCCESS] 完成！共修改 {modifications} 處")
    print(f"[INFO] 修改後的檔案: {backup_path}")
    print(f"\n[NEXT] 下一步:")
    print(f"   1. 在 Jupyter 中打開 {backup_path}")
    print(f"   2. 重新執行訓練 cells")
    print(f"   3. 訓練速度應該會快 1.5-2 倍")
    
    return backup_path

if __name__ == "__main__":
    # 可以從命令列參數指定大小，預設為 448
    new_size = 448
    if len(sys.argv) > 1:
        new_size = int(sys.argv[1])
    
    notebook_path = r'c:\DL\DL_Brain_Tumor\brain_tumor_complete.ipynb'
    
    print("="*60)
    print("  CUDA Timeout 修復工具 - 降低圖像解析度")
    print("="*60)
    print()
    
    result = modify_notebook_image_size(notebook_path, new_size)
    
    print("\n" + "="*60)
    print(f"  圖像大小對比:")
    print("="*60)
    print(f"  原始: 640x640  → 計算量: 409,600 像素")
    print(f"  修改: {new_size}x{new_size}  → 計算量: {new_size*new_size:,} 像素")
    print(f"  減少: {((640*640 - new_size*new_size) / (640*640) * 100):.1f}%")
    print("="*60)
