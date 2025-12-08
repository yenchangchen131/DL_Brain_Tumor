import json

def analyze_notebook(notebook_path):
    """分析 notebook 的結構"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        print(f"\n{'='*80}")
        print(f"Notebook: {notebook_path}")
        print(f"{'='*80}")
        
        total_cells = len(nb.get('cells', []))
        code_cells = sum(1 for cell in nb['cells'] if cell['cell_type'] == 'code')
        markdown_cells = sum(1 for cell in nb['cells'] if cell['cell_type'] == 'markdown')
        
        print(f"總 cells 數: {total_cells}")
        print(f"程式碼 cells: {code_cells}")
        print(f"Markdown cells: {markdown_cells}")
        
        print(f"\n各 Cell 摘要:")
        print(f"{'-'*80}")
        
        for i, cell in enumerate(nb['cells']):
            cell_type = cell['cell_type']
            source = ''.join(cell.get('source', []))
            
            # 取得前 100 個字元作為摘要
            summary = source[:100].replace('\n', ' ').strip()
            if len(source) > 100:
                summary += "..."
            
            # 如果是 markdown cell，取得標題
            if cell_type == 'markdown':
                lines = source.split('\n')
                for line in lines:
                    if line.startswith('#'):
                        summary = line
                        break
            
            print(f"Cell {i:2d} [{cell_type:8s}]: {summary}")
        
        return nb
        
    except Exception as e:
        print(f"錯誤: 無法讀取 {notebook_path}")
        print(f"原因: {e}")
        return None

# 分析兩個 notebooks
print("開始分析 Notebooks...")

nb1 = analyze_notebook("code.ipynb")
nb2 = analyze_notebook("brain_tumor_complete.ipynb")

if nb1 and nb2:
    print(f"\n{'='*80}")
    print("分析完成!")
    print(f"{'='*80}")
