import json

# 讀取 notebook
with open('code.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 印出所有程式碼 cell
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        print(f"=== Cell {i} ===")
        print(''.join(cell['source']))
        print("\n")
