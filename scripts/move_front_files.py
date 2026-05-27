import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
app = ROOT / 'app'
scripts = ROOT / 'scripts'
data = ROOT / 'data'
outputs = ROOT / 'outputs' / 'predictions'

app_files = ['streamlit_app.py','STREAMLIT_README.md']
script_files = ['run_streamlit.bat','run_streamlit.ps1','convert_delta_to_csv.py','setup_check.py']
data_files = ['example_test_data.csv','credit_predictions.db']
prediction_files = ['predictions_improved.parquet','train_silver_predictions.parquet']

moved = []

for f in app_files:
    src = ROOT / f
    if src.exists():
        dst = app / src.name
        shutil.move(str(src), str(dst))
        moved.append((str(src), str(dst)))

for f in script_files:
    src = ROOT / f
    if src.exists():
        dst = scripts / src.name
        shutil.move(str(src), str(dst))
        moved.append((str(src), str(dst)))

for f in data_files:
    src = ROOT / f
    if src.exists():
        dst = data / src.name
        shutil.move(str(src), str(dst))
        moved.append((str(src), str(dst)))

for f in prediction_files:
    src = ROOT / f
    if src.exists():
        outputs.mkdir(parents=True, exist_ok=True)
        dst = outputs / src.name
        shutil.move(str(src), str(dst))
        moved.append((str(src), str(dst)))

print('Moved items:')
for s,d in moved:
    print(f"{s} -> {d}")

print('\nRemaining root items:')
for p in sorted([p.name for p in ROOT.iterdir() if p.is_file()]):
    print(p)
