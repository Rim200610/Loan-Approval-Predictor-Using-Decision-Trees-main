import json

with open('Main.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    ct = cell['cell_type']
    src = ''.join(cell['source'])
    print(f"--- Cell {i} ({ct}) ---")
    print(src)
    print()
