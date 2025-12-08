import json
import copy
import sys

# 設定 stdout 編碼為 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def merge_notebooks(nb1_path, nb2_path, output_path):
    """
    合併兩個 Jupyter notebooks
    
    策略:
    1. 從 code.ipynb 取得資料探索和清理的部分
    2. 從 brain_tumor_complete.ipynb 取得完整的訓練流程
    3. 組織成一個結構良好的完整 notebook
    """
    
    # 讀取兩個 notebooks
    with open(nb1_path, 'r', encoding='utf-8') as f:
        nb1 = json.load(f)
    
    with open(nb2_path, 'r', encoding='utf-8') as f:
        nb2 = json.load(f)
    
    # 創建新的 notebook 結構 (基於 nb2 的格式)
    merged_nb = copy.deepcopy(nb2)
    merged_nb['cells'] = []
    
    print("開始合併 notebooks...")
    print("="*80)
    
    # ============================================================================
    # 第一部分: 總標題
    # ============================================================================
    title_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 腦腫瘤分割 - 完整流程\n",
            "\n",
            "本 notebook 包含從資料探索、資料清理到模型訓練和評估的完整流程。\n",
            "\n",
            "## 目錄\n",
            "1. **資料探索與清理** (來自 code.ipynb)\n",
            "   - 資料集基本資訊\n",
            "   - 標註分析\n",
            "   - 問題圖片識別\n",
            "   - 資料清理\n",
            "\n",
            "2. **模型訓練流程** (來自 brain_tumor_complete.ipynb)\n",
            "   - 環境設置\n",
            "   - 資料增強\n",
            "   - Dataset 和 DataLoader\n",
            "   - U-Net 模型\n",
            "   - 訓練與驗證\n",
            "   - 測試與視覺化\n"
        ]
    }
    merged_nb['cells'].append(title_cell)
    
    # ============================================================================
    # 第二部分: 資料探索與清理 (從 code.ipynb)
    # ============================================================================
    
    # 添加章節標題
    section1_title = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "# Part 1: 資料探索與清理\n",
            "---\n",
            "\n",
            "在訓練模型之前,我們需要先了解和清理資料集。\n"
        ]
    }
    merged_nb['cells'].append(section1_title)
    
    # 從 code.ipynb 添加 cells (跳過空的 cell 和最後的空 cell)
    for i, cell in enumerate(nb1['cells']):
        source = ''.join(cell.get('source', []))
        
        # 跳過空 cell
        if not source.strip():
            continue
        
        # 跳過最後的 "## 訓練 U-Net模型" 相關的內容,因為我們會用 nb2 的
        if '## 訓練 U-Net模型' in source or 'U-Net' in source:
            continue
        
        merged_nb['cells'].append(cell)
        print(f"✓ 添加 code.ipynb Cell {i}")
    
    # 添加資料清理的新 cell
    data_cleaning_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 資料清理 - 驗證集和測試集\n",
            "\n",
            "將訓練集的清理操作應用到驗證集和測試集。\n"
        ]
    }
    merged_nb['cells'].append(data_cleaning_cell)
    
    # 讀取資料清理程式碼
    try:
        with open('notebook_cell_code.py', 'r', encoding='utf-8') as f:
            cleaning_code = f.read()
        
        cleaning_code_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cleaning_code.split('\n')
        }
        merged_nb['cells'].append(cleaning_code_cell)
        print("✓ 添加資料清理程式碼")
    except:
        print("⚠ 警告: 找不到 notebook_cell_code.py,跳過資料清理 cell")
    
    # ============================================================================
    # 第三部分: 模型訓練流程 (從 brain_tumor_complete.ipynb)
    # ============================================================================
    
    section2_title = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "# Part 2: 模型訓練與評估\n",
            "---\n",
            "\n",
            "使用清理後的資料集進行 U-Net 模型的訓練和評估。\n"
        ]
    }
    merged_nb['cells'].append(section2_title)
    
    # 從 brain_tumor_complete.ipynb 添加所有 cells
    for i, cell in enumerate(nb2['cells']):
        merged_nb['cells'].append(cell)
        print(f"✓ 添加 brain_tumor_complete.ipynb Cell {i}")
    
    # ============================================================================
    # 保存合併後的 notebook
    # ============================================================================
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_nb, f, ensure_ascii=False, indent=1)
    
    print("="*80)
    print(f"✓ 合併完成!")
    print(f"✓ 輸出檔案: {output_path}")
    print(f"✓ 總 cells 數: {len(merged_nb['cells'])}")
    print("="*80)

# 執行合併
merge_notebooks(
    nb1_path="code.ipynb",
    nb2_path="brain_tumor_complete.ipynb",
    output_path="brain_tumor_integrated.ipynb"
)

print("\n建議:")
print("1. 在 Jupyter 中開啟 brain_tumor_integrated.ipynb")
print("2. 依序執行 cells,從資料探索到模型訓練")
print("3. 確認資料路徑使用的是清理後的 .cleaned.json 檔案")
