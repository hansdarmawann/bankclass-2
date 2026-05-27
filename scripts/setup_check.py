"""
Setup script for Streamlit app
Run this once to verify everything is ready
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_ENTRYPOINT = PROJECT_ROOT / "app" / "streamlit_app.py"
NOTEBOOK_FILE = PROJECT_ROOT / "notebooks" / "credit_score_automl_tdsp.ipynb"
DB_FILE = PROJECT_ROOT / "data" / "credit_predictions.db"

def check_packages():
    """Check if all required packages are installed"""
    print("Checking packages...")
    
    required = {
        'streamlit': 'Streamlit',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'mlflow': 'MLflow',
        'sklearn': 'scikit-learn',
        'pyarrow': 'PyArrow',
        'sqlite3': 'SQLite3'
    }
    
    missing = []
    for package, name in required.items():
        try:
            __import__(package)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name}")
            missing.append(package)
    
    return missing

def check_directories():
    """Check if required directories exist"""
    print("\nChecking directories...")

    required = {
        'app': 'Streamlit app source',
        'notebooks': 'Active notebooks',
        'scripts': 'Helper scripts',
        'data\\train_silver': 'Training data',
        'data\\test_silver': 'Test data',
        'outputs\\mlruns': 'MLflow runs',
        'outputs': 'Generated outputs'
    }
    
    missing = []
    for folder, name in required.items():
        path = PROJECT_ROOT / Path(folder)
        if path.exists():
            print(f"  [OK] {name} ({folder})")
        else:
            print(f"  [MISSING] {name} ({folder}) - MISSING")
            missing.append(folder)
    
    return missing

def check_model():
    """Check if model exists in MLflow"""
    print("\nChecking model...")

    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'outputs' / 'mlflow.db').as_posix()}")
        
        client = mlflow.tracking.MlflowClient()
        
        try:
            versions = client.search_model_versions("name='credit_score_classifier'")
            if versions:
                print(f"  [OK] Model found ({len(versions)} versions)")
                latest = max(versions, key=lambda x: x.version)
                print(f"     Latest: v{latest.version}")
            else:
                print(f"  [MISSING] Registered model NOT found")
                print(f"     Run the Jupyter Notebook first to train and register model")
        except:
            print(f"  [ERROR] Could not check registered model")
            print(f"     Make sure Jupyter Notebook completed successfully")
    
    except Exception as e:
            print(f"  [ERROR] Error checking model: {str(e)}")

def check_files():
    """Check key project files and writable runtime locations"""
    print("\nChecking files...")

    required_files = {
        APP_ENTRYPOINT: "Streamlit entrypoint",
        NOTEBOOK_FILE: "Active TDSP notebook",
        PROJECT_ROOT / "scripts" / "requirements.txt": "App requirements",
        PROJECT_ROOT / "requirements-notebook.txt": "Notebook requirements",
    }

    missing = []
    for path, name in required_files.items():
        if path.exists():
            print(f"  [OK] {name} ({path.relative_to(PROJECT_ROOT)})")
        else:
            print(f"  [MISSING] {name} ({path.relative_to(PROJECT_ROOT)})")
            missing.append(str(path.relative_to(PROJECT_ROOT)))

    db_dir = DB_FILE.parent
    if db_dir.exists():
        print(f"  [OK] Database directory ({db_dir.relative_to(PROJECT_ROOT)})")
    else:
        print(f"  [MISSING] Database directory ({db_dir.relative_to(PROJECT_ROOT)})")
        missing.append(str(db_dir.relative_to(PROJECT_ROOT)))

    return missing

def main():
    print("="*50)
    print("Credit Score Classifier - Setup Check")
    print("="*50)
    
    # Check packages
    missing_packages = check_packages()
    
    # Check directories
    missing_dirs = check_directories()

    # Check files
    missing_files = check_files()
    
    # Check model
    check_model()
    
    # Summary
    print("\n" + "="*50)
    if not missing_packages and not missing_dirs and not missing_files:
        print("[OK] Everything looks good! Ready to use.")
        print("\nTo start the app, run:")
        print("   Windows: scripts\\run_streamlit.bat")
        print("   PowerShell: .\\scripts\\run_streamlit.ps1")
        print("   Direct: python -m streamlit run app\\streamlit_app.py")
        return 0
    else:
        print("[ISSUES] Some issues found:")
        if missing_packages:
            print(f"\n  Missing packages ({len(missing_packages)}):")
            for pkg in missing_packages:
                print(f"    - {pkg}")
            print(f"\n  Install with:")
            print(f"    pip install {' '.join(missing_packages)}")
        
        if missing_dirs:
            print(f"\n  Missing directories ({len(missing_dirs)}):")
            for folder in missing_dirs:
                print(f"    - {folder}")
            print(f"\n  Please ensure these folders exist in the project root")

        if missing_files:
            print(f"\n  Missing files ({len(missing_files)}):")
            for file_path in missing_files:
                print(f"    - {file_path}")
        
        print("\n  Then run setup again to verify.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
