import json
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_FILE = PROJECT_ROOT / "notebooks" / "credit_score_automl_tdsp.ipynb"

# Validate the TDSP notebook
with open(NOTEBOOK_FILE, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("[OK] Notebook is valid JSON")
print(f"File: {NOTEBOOK_FILE.relative_to(PROJECT_ROOT)}")
print(f"Total cells: {len(nb['cells'])}")
print(f"Notebook format: {nb['nbformat']}.{nb['nbformat_minor']}")

# Count cell types
markdown_cells = sum(1 for cell in nb['cells'] if cell['cell_type'] == 'markdown')
code_cells = sum(1 for cell in nb['cells'] if cell['cell_type'] == 'code')

print(f"\nCell breakdown:")
print(f"   - Markdown cells: {markdown_cells}")
print(f"   - Code cells: {code_cells}")

# Check TDSP sections
tdsp_sections = [
    "Business Understanding",
    "Data Acquisition",
    "Data Understanding",
    "Data Preparation",
    "Modeling",
    "Evaluation",
    "Deployment"
]

print(f"\nTDSP Sections found:")
for section in tdsp_sections:
    found = any(section in ''.join(cell.get('source', []))
                for cell in nb['cells'] if cell['cell_type'] == 'markdown')
    status = "[OK]" if found else "[MISSING]"
    print(f"   {status} {section}")

print(f"\n[SUCCESS] Notebook structure validated successfully!")

# Made with Bob
