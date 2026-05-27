"""
Script untuk convert Delta Lake format dari test_silver menjadi CSV
"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Try reading with Delta Lake first, fallback to direct parquet reading
def convert_delta_to_csv():
    test_silver_path = PROJECT_ROOT / 'data' / 'test_silver'
    
    print(f"📁 Reading from: {test_silver_path}")
    
    try:
        # Try using delta lake
        try:
            import delta
            df = pd.read_parquet(str(test_silver_path))
        except:
            # Fallback: read parquet files directly from delta directory
            parquet_files = list(test_silver_path.glob('*.parquet'))
            
            if not parquet_files:
                print("❌ No parquet files found in test_silver")
                return
            
            print(f"✅ Found {len(parquet_files)} parquet file(s)")
            df = pd.read_parquet(str(parquet_files[0]))
        
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"\n📋 Columns: {list(df.columns)}")
        print(f"\n🔍 First few rows:")
        print(df.head())
        
        # Save as CSV
        csv_path = test_silver_path / 'test_data.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n💾 CSV saved to: {csv_path}")
        print(f"✅ Conversion complete!")
        
        return csv_path
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    convert_delta_to_csv()
