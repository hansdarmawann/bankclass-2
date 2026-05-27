import json
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from pathlib import Path
import mlflow
import mlflow.pyfunc
import logging
import pyarrow.parquet as pq

# Page config
st.set_page_config(
    page_title="Credit Score Classifier",
    page_icon="ðŸ’³",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_FILE = PROJECT_ROOT / "data" / "credit_predictions.db"
MODEL_NAME = "credit_score_classifier"
TRAIN_DIR = PROJECT_ROOT / 'train_silver'
DERIVED_FEATURES = {
    'debt_to_income_ratio',
    'utilization_per_card',
    'income_to_emi_ratio',
    'age_group'
}

def init_database():
    """Initialize SQLite database"""
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            input_method TEXT,
            actual_credit_score TEXT,
            predicted_credit_score TEXT,
            prob_poor REAL,
            prob_standard REAL,
            prob_good REAL,
            confidence REAL,
            input_data TEXT
        )
    ''')
    
    conn.commit()
    conn.close()


def save_prediction(input_method, actual_score, predicted_score, probabilities, input_data_json):
    """Save prediction to database"""
    if not isinstance(input_data_json, str):
        input_data_json = json.dumps(input_data_json, default=str)

    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO predictions 
        (input_method, actual_credit_score, predicted_credit_score, prob_poor, prob_standard, prob_good, confidence, input_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        input_method,
        actual_score,
        predicted_score,
        float(probabilities.get('prob_poor', 0)),
        float(probabilities.get('prob_standard', 0)),
        float(probabilities.get('prob_good', 0)),
        float(probabilities.get('confidence', 0)),
        input_data_json
    ))
    
    conn.commit()
    conn.close()


def parse_registered_model_meta(meta_file: Path):
    """Read MLflow registered_model_meta without requiring a YAML dependency."""
    metadata = {}
    for line in meta_file.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip("'\"")
    return metadata


def find_model_artifact():
    """Find the intended local MLflow model artifact."""
    possible_roots = [
        PROJECT_ROOT / 'mlruns' / '1' / 'models',
        PROJECT_ROOT / 'archive' / 'mlruns' / '1' / 'models'
    ]

    named_candidates = []
    fallback_candidates = []
    for model_root in possible_roots:
        if not model_root.exists():
            continue
        for model_dir in model_root.iterdir():
            artifact_dir = model_dir / 'artifacts'
            if not artifact_dir.exists():
                continue
            fallback_candidates.append((artifact_dir, model_dir.name, 0))

            meta_file = artifact_dir / 'registered_model_meta'
            if not meta_file.exists():
                continue
            try:
                metadata = parse_registered_model_meta(meta_file)
                if metadata.get('model_name') != MODEL_NAME:
                    continue
                version = int(metadata.get('model_version', '0'))
                named_candidates.append((artifact_dir, f"{MODEL_NAME} v{version}", version))
            except Exception as exc:
                logger.warning(f"Failed to parse {meta_file}: {exc}")

    candidates = named_candidates or fallback_candidates
    if not candidates:
        return None, None

    if named_candidates:
        artifact_dir, display_name, _ = max(candidates, key=lambda item: item[2])
    else:
        artifact_dir, display_name, _ = max(candidates, key=lambda item: item[0].stat().st_mtime)

    return artifact_dir, display_name


@st.cache_resource
def load_best_model():
    """Load latest model from mlruns/1/models"""
    try:
        mlflow.set_tracking_uri(f"file:///{PROJECT_ROOT / 'mlruns'}")
        artifact_path, model_id = find_model_artifact()
        if artifact_path is None:
            st.error("âŒ Model MLflow terbaru tidak ditemukan di mlruns/1/models.")
            return None, None

        model = mlflow.pyfunc.load_model(str(artifact_path))
        st.success(f"Loaded model: {model_id}")
        return model, model_id

    except Exception as e:
        st.error(f"âŒ Error loading model: {str(e)}")
        return None, None


def get_model_input_spec(model):
    """Extract input columns and types from model signature"""
    schema = model.metadata.get_input_schema()
    input_columns = [col.name for col in schema.inputs]
    column_types = {}
    for col in schema.inputs:
        col_type = str(col.type).lower()
        if 'string' in col_type:
            column_types[col.name] = 'string'
        else:
            column_types[col.name] = 'double'
    return input_columns, column_types


@st.cache_resource
def load_training_sample(sample_size: int = 100):
    """Load a sample of the training dataset for form defaults"""
    try:
        parquet_files = sorted(TRAIN_DIR.glob('*.parquet'))
        if not parquet_files:
            return pd.DataFrame()
        df = pq.read_table(str(parquet_files[0])).to_pandas()
        return df.head(sample_size)
    except Exception as e:
        logger.warning(f"Failed to load training sample: {e}")
        return pd.DataFrame()


def load_label_mapping(sample_df: pd.DataFrame):
    """Generate mapping from numeric label to human-readable label"""
    default_mapping = {0: 'Good', 1: 'Poor', 2: 'Standard'}
    labels = []

    if 'credit_score' in sample_df.columns:
        labels = sorted(sample_df['credit_score'].dropna().unique().tolist())

    if len(labels) < 3:
        try:
            parquet_files = sorted(TRAIN_DIR.glob('*.parquet'))
            for parquet_file in parquet_files:
                table = pq.read_table(str(parquet_file), columns=['credit_score'])
                values = table.to_pandas()['credit_score'].dropna().unique().tolist()
                labels.extend([str(x) for x in values if x not in labels])
                if len(labels) >= 3:
                    break
            labels = sorted(set(labels))
        except Exception:
            pass

    if labels:
        return {i: label for i, label in enumerate(labels)}
    return default_mapping


def compute_derived_features(df: pd.DataFrame):
    """Compute derived features expected by the model"""
    df = df.copy()

    if 'outstanding_debt' in df.columns and 'annual_income' in df.columns:
        df['debt_to_income_ratio'] = df['outstanding_debt'] / (df['annual_income'] + 1)
    else:
        df['debt_to_income_ratio'] = 0.0

    if 'credit_utilization_ratio' in df.columns and 'num_credit_card' in df.columns:
        df['utilization_per_card'] = df['credit_utilization_ratio'] / (df['num_credit_card'] + 1)
    else:
        df['utilization_per_card'] = 0.0

    if 'monthly_inhand_salary' in df.columns and 'total_emi_per_month' in df.columns:
        df['income_to_emi_ratio'] = df['monthly_inhand_salary'] / (df['total_emi_per_month'] + 1)
    else:
        df['income_to_emi_ratio'] = 0.0

    if 'age' in df.columns:
        df['age_group'] = pd.cut(
            df['age'].astype(float),
            bins=[0, 25, 35, 45, 55, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '55+']
        ).astype(str).fillna('Unknown')
    else:
        df['age_group'] = 'Unknown'

    return df


def preprocess_input(df_input: pd.DataFrame, required_columns, column_types):
    """Prepare input DataFrame for model scoring"""
    df = compute_derived_features(df_input.copy())

    for column in required_columns:
        if column not in df.columns:
            df[column] = 'Unknown' if column_types.get(column) == 'string' else 0.0

    df = df[required_columns].copy()

    for column in required_columns:
        if column_types.get(column) == 'string':
            df[column] = df[column].astype(str).fillna('Unknown')
        else:
            df[column] = pd.to_numeric(df[column], errors='coerce').astype(float).fillna(0.0)

    return df


def get_model_predict_proba(model, processed_data):
    """Try to get probability scores from the model."""
    try:
        return model.predict_proba(processed_data)
    except AttributeError:
        pass
    except Exception as exc:
        logger.warning(f"predict_proba failed on model: {exc}")

    for candidate_name in ('_model_impl', '_model', 'model', 'sklearn_model'):
        candidate = getattr(model, candidate_name, None)
        if candidate is not None:
            try:
                return candidate.predict_proba(processed_data)
            except AttributeError:
                continue
            except Exception as exc:
                logger.warning(f"predict_proba failed on {candidate_name}: {exc}")

    return None


def normalize_credit_label(label, label_mapping):
    """Convert encoded model labels to display labels."""
    if isinstance(label, (np.integer, int)):
        return label_mapping.get(int(label), str(label))
    label_text = str(label)
    if label_text.isdigit():
        return label_mapping.get(int(label_text), label_text)
    return label_text


def make_prediction(model, input_data, required_columns, column_types, label_mapping):
    """Make prediction using the model"""
    try:
        processed_data = preprocess_input(input_data, required_columns, column_types)

        predictions = model.predict(processed_data)
        probabilities = get_model_predict_proba(model, processed_data)

        predicted_value = predictions[0]
        predicted_label = normalize_credit_label(predicted_value, label_mapping)

        prob_dict = {
            'prob_poor': 0.0,
            'prob_standard': 0.0,
            'prob_good': 0.0,
            'confidence': 0.0
        }

        if probabilities is not None:
            prob_vector = probabilities[0]
            classes = None
            if hasattr(model, '_model_impl') and hasattr(model._model_impl, 'classes_'):
                classes = list(model._model_impl.classes_)
            elif hasattr(model, '_model') and hasattr(model._model, 'classes_'):
                classes = list(model._model.classes_)
            elif hasattr(model, 'sklearn_model') and hasattr(model.sklearn_model, 'classes_'):
                classes = list(model.sklearn_model.classes_)

            if classes is None:
                ordered_labels = [label_mapping[i] for i in sorted(label_mapping.keys())]
            else:
                ordered_labels = [normalize_credit_label(label, label_mapping) for label in classes]

            for idx, label in enumerate(ordered_labels):
                prob_dict[f"prob_{str(label).lower()}"] = float(prob_vector[idx]) if idx < len(prob_vector) else 0.0
            prob_dict['confidence'] = float(np.max(prob_vector))

        return predicted_label, prob_dict
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# ==================== STREAMLIT UI ====================
def main():
    st.title("ðŸ’³ Credit Score Classification")
    st.markdown("---")
    
    # Initialize database
    init_database()
    
    # Load model
    model, model_version = load_best_model()
    if model is None:
        st.error("âŒ Model tidak dapat dimuat. Pastikan model sudah dilatih.")
        return
    
    # Load model input schema and sample data
    model_input_columns, model_input_types = get_model_input_spec(model)
    sample_df = load_training_sample()
    label_mapping = load_label_mapping(sample_df)
    
    # Sidebar
    st.sidebar.title("Opsi Input")
    input_method = st.sidebar.radio(
        "Pilih metode input data:",
        ["Upload CSV", "Form Input"]
    )
    
    # ==================== CSV UPLOAD ====================
    if input_method == "Upload CSV":
        st.header("Upload CSV File")
        
        uploaded_file = st.file_uploader(
            "Pilih file CSV",
            type=['csv'],
            help="Format: CSV dengan kolom sesuai data training"
        )
        
        if uploaded_file:
            try:
                # Read CSV with auto-detection of delimiter
                # Try different delimiters if first one fails
                delimiters = [',', ';', '\t', '|', ' ']
                df = None
                
                for delimiter in delimiters:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        df = pd.read_csv(uploaded_file, delimiter=delimiter)
                        if len(df.columns) > 1:  # Successfully parsed with multiple columns
                            st.info(f"âœ… CSV berhasil dibaca dengan delimiter: '{delimiter}'")
                            break
                    except:
                        continue
                
                if df is None:
                    # Fallback: try reading without specifying delimiter
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file)
                
                # Clean column names (remove extra whitespace)
                df.columns = df.columns.str.strip()
                
                st.write(f"ðŸ“Š Data dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
                
                # Show preview
                with st.expander("ðŸ‘€ Preview Data", expanded=False):
                    st.dataframe(df.head(10))
                
                # Process predictions
                if st.button("ðŸš€ Prediksi", key="csv_predict"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results_list = []
                    
                    with st.spinner("â³ Processing predictions..."):
                        for idx, row in df.iterrows():
                            try:
                                # Prepare input
                                input_data = pd.DataFrame([row])
                                
                                # Make prediction
                                pred_class, prob_dict = make_prediction(
                                    model,
                                    input_data,
                                    model_input_columns,
                                    model_input_types,
                                    label_mapping
                                )
                                
                                if pred_class is not None:
                                    results_list.append({
                                        'row_index': idx,
                                        'prediction': pred_class,
                                        'confidence': prob_dict['confidence'],
                                        'probabilities': prob_dict
                                    })
                                    
                                    # Save to database
                                    actual_score = row.get('credit_score', 'unknown')
                                    save_prediction(
                                        input_method='CSV',
                                        actual_score=str(actual_score),
                                        predicted_score=str(pred_class),
                                        probabilities=prob_dict,
                                        input_data_json=row.to_json()
                                    )
                                
                                # Update progress
                                progress = (idx + 1) / len(df)
                                progress_bar.progress(progress)
                                status_text.text(f"âœ… Diproses: {idx + 1}/{len(df)}")
                            
                            except Exception as e:
                                logger.error(f"Error on row {idx}: {str(e)}")
                                continue
                    
                    # Display results
                    if results_list:
                        st.success(f"âœ… Prediksi selesai! {len(results_list)} data berhasil diproses")
                        
                        results_df = pd.DataFrame(results_list)
                        
                        # Show results
                        st.subheader("ðŸ“ˆ Hasil Prediksi")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Prediksi", len(results_df))
                        with col2:
                            st.metric("Avg Confidence", f"{results_df['confidence'].mean():.2%}")
                        with col3:
                            st.metric("Model Version", model_version or "Latest")
                        
                        # Results table
                        display_results = results_df[['row_index', 'prediction', 'confidence']].copy()
                        display_results.columns = ['Row', 'Prediction', 'Confidence']
                        display_results['Confidence'] = display_results['Confidence'].apply(lambda x: f"{x:.2%}")
                        
                        st.dataframe(display_results, use_container_width=True)
                        
                        # Download results
                        results_export = results_df.copy()
                        results_export['probabilities'] = results_export['probabilities'].apply(lambda x: str(x))
                        csv_export = results_export.to_csv(index=False)
                        
                        st.download_button(
                            label="ðŸ“¥ Download Hasil Prediksi (CSV)",
                            data=csv_export,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("âŒ Tidak ada prediksi yang berhasil")
            
            except Exception as e:
                st.error(f"âŒ Error membaca CSV: {str(e)}")
    
    # ==================== FORM INPUT ====================
    else:
        st.header("Input Data Manual")

        input_columns = [col for col in model_input_columns if col not in DERIVED_FEATURES]
        numeric_columns = [col for col in input_columns if model_input_types.get(col) == 'double']
        string_columns = [col for col in input_columns if model_input_types.get(col) == 'string']

        form_data = {}
        
        # Numeric fields
        st.subheader("ðŸ”¢ Numeric Input")
        col1, col2 = st.columns(2)
        first_half = numeric_columns[: len(numeric_columns) // 2]
        second_half = numeric_columns[len(numeric_columns) // 2 :]

        def default_numeric(col_name):
            if not sample_df.empty and col_name in sample_df.columns:
                return float(sample_df[col_name].median())
            return 0.0

        with col1:
            for col in first_half:
                if 'age' == col:
                    form_data[col] = st.number_input(
                        f"ðŸ”¢ {col.replace('_', ' ').title()}",
                        value=int(default_numeric(col) or 30),
                        min_value=0,
                        max_value=120,
                        step=1
                    )
                elif any(x in col.lower() for x in ['count', 'num', 'number', 'loan', 'inquiries']):
                    form_data[col] = st.number_input(
                        f"ðŸ”¢ {col.replace('_', ' ').title()}",
                        value=int(default_numeric(col)),
                        min_value=0,
                        step=1
                    )
                else:
                    form_data[col] = st.number_input(
                        f"ðŸ’° {col.replace('_', ' ').title()}",
                        value=float(default_numeric(col)),
                        min_value=0.0,
                        step=0.01
                    )

        with col2:
            for col in second_half:
                if 'age' == col:
                    form_data[col] = st.number_input(
                        f"ðŸ”¢ {col.replace('_', ' ').title()}",
                        value=int(default_numeric(col) or 30),
                        min_value=0,
                        max_value=120,
                        step=1
                    )
                elif any(x in col.lower() for x in ['count', 'num', 'number', 'loan', 'inquiries']):
                    form_data[col] = st.number_input(
                        f"ðŸ”¢ {col.replace('_', ' ').title()}",
                        value=int(default_numeric(col)),
                        min_value=0,
                        step=1
                    )
                else:
                    form_data[col] = st.number_input(
                        f"ðŸ’° {col.replace('_', ' ').title()}",
                        value=float(default_numeric(col)),
                        min_value=0.0,
                        step=0.01
                    )

        # Categorical fields
        st.subheader("ðŸ§¾ Categorical Input")
        for col in string_columns:
            default_value = 'Unknown'
            options = []
            if not sample_df.empty and col in sample_df.columns:
                options = sorted(sample_df[col].dropna().astype(str).unique().tolist())
                if options:
                    default_value = options[0]

            if options and 1 < len(options) <= 30:
                form_data[col] = st.selectbox(
                    f"{col.replace('_', ' ').title()}",
                    options,
                    index=0
                )
            else:
                form_data[col] = st.text_input(
                    f"{col.replace('_', ' ').title()}",
                    value=default_value
                )

        # Optional: actual credit score
        with st.expander("ðŸ“‹ Informasi Tambahan (Opsional)"):
            actual_score = st.selectbox(
                "Kredit score aktual (jika tersedia):",
                ["Unknown", "Poor", "Standard", "Good"]
            )

        if st.button("ðŸš€ Prediksi", key="form_predict", use_container_width=True):
            try:
                input_df = pd.DataFrame([form_data])

                with st.spinner("â³ Processing prediction..."):
                    pred_class, prob_dict = make_prediction(
                        model,
                        input_df,
                        model_input_columns,
                        model_input_types,
                        label_mapping
                    )

                if pred_class is not None:
                    save_prediction(
                        input_method='Form',
                        actual_score=actual_score,
                        predicted_score=str(pred_class),
                        probabilities=prob_dict,
                        input_data_json=input_df.to_json()
                    )

                    st.success("âœ… Prediksi selesai!")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prediksi", pred_class, delta=None)
                    with col2:
                        st.metric("Confidence", f"{prob_dict['confidence']:.2%}")
                    with col3:
                        st.metric("Model Version", model_version or "Latest")

                    st.subheader("ðŸ“Š Probabilitas per Kategori")
                    prob_data = {
                        'Kategori': ['Poor', 'Standard', 'Good'],
                        'Probabilitas': [
                            prob_dict.get('prob_poor', 0),
                            prob_dict.get('prob_standard', 0),
                            prob_dict.get('prob_good', 0)
                        ]
                    }
                    prob_df = pd.DataFrame(prob_data)
                    prob_df['Probabilitas %'] = (prob_df['Probabilitas'] * 100).round(2)

                    st.bar_chart(prob_df.set_index('Kategori')['Probabilitas %'])
                    st.dataframe(prob_df, use_container_width=True)

            except Exception as e:
                st.error(f"âŒ Error: {str(e)}")
    
    # ==================== HISTORY & DATABASE ====================
    st.markdown("---")
    
    with st.expander("ðŸ“œ Riwayat Prediksi"):
        try:
            conn = sqlite3.connect(str(DB_FILE))
            history_df = pd.read_sql_query(
                "SELECT timestamp, input_method, actual_credit_score, predicted_credit_score, confidence FROM predictions ORDER BY timestamp DESC LIMIT 50",
                conn
            )
            conn.close()
            
            if len(history_df) > 0:
                st.write(f"ðŸ“Š Total prediksi tercatat: {len(history_df)}")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("CSV Uploads", len(history_df[history_df['input_method'] == 'CSV']))
                with col2:
                    st.metric("Form Inputs", len(history_df[history_df['input_method'] == 'Form']))
                with col3:
                    st.metric("Avg Confidence", f"{history_df['confidence'].mean():.2%}")
                
                st.dataframe(history_df, use_container_width=True)
                
                # Download all data
                csv_data = history_df.to_csv(index=False)
                st.download_button(
                    label="ðŸ“¥ Download Semua Riwayat (CSV)",
                    data=csv_data,
                    file_name=f"all_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("â„¹ï¸ Belum ada prediksi yang tersimpan")
        
        except Exception as e:
            st.warning(f"âš ï¸ Tidak dapat membaca riwayat: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
        ðŸ’¡ Tips: Upload CSV dengan format yang sama dengan data training, atau gunakan form untuk input manual.
        <br>
        ðŸ“‚ Database: {db_path}
        <br>
        ðŸ¤– Model Version: {version}
    </div>
    """.format(
        db_path=DB_FILE,
        version=model_version or "Latest"
    ), unsafe_allow_html=True)

if __name__ == "__main__":
    main()

