import json

# 驗證合併後的 notebook
with open('brain_tumor_integrated.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=" * 80)
print("驗證 brain_tumor_integrated.ipynb")
print("=" * 80)

total_cells = len(nb['cells'])
print(f"\n總 cells 數: {total_cells}")

# 統計不同類型的 cells
code_cells = sum(1 for cell in nb['cells'] if cell['cell_type'] == 'code')
markdown_cells = sum(1 for cell in nb['cells'] if cell['cell_type'] == 'markdown')

print(f"程式碼 cells: {code_cells}")
print(f"Markdown cells: {markdown_cells}")

print("\n前 15 個 cells 預覽:")
print("-" * 80)
for i, cell in enumerate(nb['cells'][:15]):
    cell_type = cell['cell_type']
    source = ''.join(cell.get('source', []))
    
    # 取第一行作為摘要
    first_line = source.split('\n')[0] if source else "(空)"
    if len(first_line) > 60:
        first_line = first_line[:60] + "..."
    
    print(f"Cell {i:2d} [{cell_type:8s}]: {first_line}")

print("\n" + "=" * 80)
print("✓ Notebook 結構驗證完成!")
print("=" * 80)
