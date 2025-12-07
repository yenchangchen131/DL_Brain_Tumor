"""
修改 brain_tumor_complete.ipynb 以實施方案 A: 解決 CUDA timeout 問題
- 降低影像解析度: 640x640 → 448x448
- 使用較小的 U-Net: features=[64,128,256,512] → [32,64,128,256]
- 提升 batch size: 1 → 4
"""

import json
from pathlib import Path

def modify_notebook_for_solution_a():
    """修改 notebook 實施方案 A"""
    
    notebook_path = Path('c:/DL/DL_Brain_Tumor/brain_tumor_complete.ipynb')
    
    # 讀取 notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    modifications_made = []
    
    # 遍歷所有 cells
    for cell_idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = cell['source']
        modified = False
        
        # 修改 1: TrainTransform 的 image_size
        for i, line in enumerate(source):
            if 'def __init__(self, image_size=640):' in line and 'TrainTransform' in ''.join(source[max(0, i-5):i]):
                source[i] = line.replace('image_size=640', 'image_size=448')
                modifications_made.append(f"Cell {cell_idx}: TrainTransform image_size 640→448")
                modified = True
        
        # 修改 2: ValidTransform 的 image_size
        for i, line in enumerate(source):
            if 'def __init__(self, image_size=640):' in line and 'ValidTransform' in ''.join(source[max(0, i-5):i]):
                source[i] = line.replace('image_size=640', 'image_size=448')
                modifications_made.append(f"Cell {cell_idx}: ValidTransform image_size 640→448")
                modified = True
        
        # 修改 3: UNet 預設 features
        for i, line in enumerate(source):
            if 'def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):' in line:
                source[i] = line.replace('features=[64, 128, 256, 512]', 'features=[32, 64, 128, 256]')
                modifications_made.append(f"Cell {cell_idx}: UNet features [64,128,256,512]→[32,64,128,256]")
                modified = True
        
        # 修改 4: 模型創建時指定較小的 features (測試 cell)
        for i, line in enumerate(source):
            if line.strip() == 'model = UNet(in_channels=3, out_channels=1)\\n':
                source[i] = 'model = UNet(in_channels=3, out_channels=1, features=[32, 64, 128, 256])\\n'
                modifications_made.append(f"Cell {cell_idx}: Model creation with features=[32,64,128,256]")
                modified = True
        
        # 修改 5: 測試輸入大小
        for i, line in enumerate(source):
            if 'test_input = torch.randn(1, 3, 640, 640)' in line:
                source[i] = line.replace('(1, 3, 640, 640)', '(1, 3, 448, 448)')
                modifications_made.append(f"Cell {cell_idx}: Test input 640→448")
                modified = True
        
        # 修改 6: BATCH_SIZE
        for i, line in enumerate(source):
            if line.strip().startswith('BATCH_SIZE = 1'):
                source[i] = 'BATCH_SIZE = 4       # 優化設定：配合較小影像和模型\\n'
                modifications_made.append(f"Cell {cell_idx}: BATCH_SIZE 1→4")
                modified = True
        
        # 修改 7: 訓練時的模型創建
        for i, line in enumerate(source):
            if 'model = UNet(in_channels=3, out_channels=1).to(device)' in line:
                source[i] = line.replace(
                    'UNet(in_channels=3, out_channels=1)',
                    'UNet(in_channels=3, out_channels=1, features=[32, 64, 128, 256])'
                )
                modifications_made.append(f"Cell {cell_idx}: Training model with features=[32,64,128,256]")
                modified = True
        
        if modified:
            cell['source'] = source
    
    # 儲存修改後的 notebook
    backup_path = notebook_path.with_suffix('.ipynb.backup')
    
    # 備份原始檔案
    import shutil
    shutil.copy2(notebook_path, backup_path)
    print(f"✓ 已備份原始檔案到: {backup_path}")
    
    # 寫入修改後的 notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 已成功修改 {notebook_path}")
    print(f"\n進行的修改:")
    for mod in modifications_made:
        print(f"  • {mod}")
    
    print(f"\n修改總數: {len(modifications_made)}")
    
    return len(modifications_made)

if __name__ == '__main__':
    print("=" * 60)
    print("方案 A 實施: 解決 CUDA Timeout 問題")
    print("=" * 60)
    print("\n修改內容:")
    print("  1. 影像解析度: 640x640 → 448x448")
    print("  2. U-Net 特徵: [64,128,256,512] → [32,64,128,256]")
    print("  3. Batch Size: 1 → 4")
    print("\n預期效果:")
    print("  • 訓練速度提升 2-4 倍")
    print("  • 每個 iteration: < 10 秒")
    print("  • 不會發生 CUDA timeout")
    print("\n開始修改...\n")
    
    count = modify_notebook_for_solution_a()
    
    if count > 0:
        print("\n" + "=" * 60)
        print("✓ 修改完成！")
        print("=" * 60)
        print("\n下一步:")
        print("  1. 重新開啟 Jupyter Notebook")
        print("  2. 執行修改後的 cells")
        print("  3. 觀察訓練速度和穩定性")
        print("\n如果需要復原，備份檔案位於:")
        print("  brain_tumor_complete.ipynb.backup")
    else:
        print("\n⚠️ 沒有找到需要修改的內容，請檢查 notebook 結構")
