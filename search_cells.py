import json

# 讀取 notebook
with open('code.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 搜尋特定關鍵字
keywords = ['train', 'valid', 'test', 'cleaning', 'remove', 'delete']

# 印出包含關鍵字的 cell
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source'])
        # 檢查是否包含關鍵字
        if any(keyword in source_text.lower() for keyword in keywords):
            print(f"=== Cell {i} ===")
            print(source_text)
            print("\n" + "="*80 + "\n")
