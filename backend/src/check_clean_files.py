import os
import pandas as pd

# 1. Resolve path to your main dataset relative to /backend
SRC_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
BACKEND_DIR = os.path.dirname(SRC_DIR)

master_ledger = os.path.join(BACKEND_DIR, "data", "processed", "scaled_siat_fraud_dataset.parquet")

print(f"🗃️ Reading main data ledger from: {master_ledger}")

try:
    # 2. Read the parquet dataset
    df = pd.read_parquet(master_ledger)
    
    # 3. Filter out rows where is_anomaly_ip is 0 (Non-anomalous / Clean transactions)
    # Adjust column name casing if your dataset layout uses 'is_anomaly_ip' or 'IS_ANOMALY_IP'
    anomaly_col = 'is_anomaly_ip' if 'is_anomaly_ip' in df.columns else 'IS_ANOMALY_IP'
    
    clean_transactions = df[df[anomaly_col] == 0.0]
    
    print(f"\n📊 Dataset Analysis Summary:")
    print(f" • Total transactions in ledger: {len(df)}")
    print(f" • Clean transactions (No Anomaly): {len(clean_transactions)}")
    print(f" • Flagged anomalies: {len(df) - len(clean_transactions)}")
    print("=" * 70)
    
    # 4. Display the top 10 clean files to test in your UI
    print("\n💡 TOP 10 CLEAN TRANSACTIONS (Use these in Streamlit to test compliant pathways):")
    columns_to_show = ['file_id', 'numero_dossier', 'emetteur', 'valeur_fob', anomaly_col]
    # Check what columns exist to prevent KeyError
    existing_cols = [c for c in columns_to_show if c in df.columns]
    
    print(clean_transactions[existing_cols].head(10).to_string(index=False))
    print("=" * 70)

except Exception as e:
    print(f"❌ Error loading dataset matrix: {str(e)}")