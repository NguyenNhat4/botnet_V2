import pandas as pd
import numpy as np
import pickle
import os
from tqdm.auto import tqdm

def quick_classify(label):
    """
    Classifies a label string into 'Botnet', 'C&C', or 'Normal'.
    """
    if not isinstance(label, str):
        return 'Normal'
    label_lower = label.lower()
    if 'c&c' in label_lower or 'cc' in label_lower:
        return 'C&C'
    elif 'botnet' in label_lower:
        return 'Botnet'
    else:
        return 'Normal'

def calculate_global_frequencies(csv_paths):
    """
    Calculates global frequencies for IPs and Ports across all CSV files.
    Returns a dictionary of dictionaries.
    """
    freqs = {
        'SrcAddr': {},
        'DstAddr': {},
        'Sport': {},
        'Dport': {}
    }
    
    print("Scanning CSVs for frequencies...")
    for path in tqdm(csv_paths, desc="Global Freqs"):
        try:
            # Read specific columns to save memory
            # Note: low_memory=False helps with mixed types
            df = pd.read_csv(path, usecols=['SrcAddr', 'DstAddr', 'Sport', 'Dport'], low_memory=False)
            
            for col in freqs:
                if col in df.columns:
                    # Value counts
                    vc = df[col].value_counts().to_dict()
                    for k, v in vc.items():
                        # We use simple types for keys to ensure matches later
                        # Convert key to string if needed?
                        # Ports might be int or str. IPs are str.
                        # Let's keep original types but be careful in mapping.
                        # Actually, better to normalize to string for consistency?
                        # But process_batch converts to numeric for ports.
                        # Let's stick to raw values here, assume map handles it or we align later.
                        # But for safety, maybe convert to string?
                        # The notebook output says "Unique Source Ports: 61941".
                        
                        freqs[col][k] = freqs[col].get(k, 0) + v
        except Exception as e:
            print(f"Warning: Could not process {path} for frequencies: {e}")
            
    return freqs

def process_batch_fast_v2(chunk, freq_dicts, expected_columns=None):
    """
    Processes a batch (DataFrame chunk).
    1. Cleans data.
    2. Encodes features (Frequencies, One-Hot).
    3. Aligns columns.
    Returns (X_values, y_values, columns_list).
    """
    df = chunk.copy()
    
    # 1. Label Processing
    if 'Label' in df.columns:
        y = df['Label'].apply(quick_classify)
        df = df.drop(columns=['Label'])
    else:
        y = None 

    # 2. Basic Cleaning
    if 'StartTime' in df.columns:
        df = df.drop(columns=['StartTime'])
        
    # 3. Numeric Conversions
    # Columns that should be numeric
    numeric_cols = ['Dur', 'TotPkts', 'TotBytes', 'SrcBytes']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # 4. Feature Engineering: Frequencies
    for col in ['SrcAddr', 'DstAddr', 'Sport', 'Dport']:
        freq_col_name = f"{col}_freq"
        if col in df.columns and col in freq_dicts:
            # Map values. Using map with a dict. 
            # Note: If types don't match (e.g. int vs str), map produces NaN.
            # We assume consistent types from read_csv.
            df[freq_col_name] = df[col].map(freq_dicts.get(col, {})).fillna(0)
        else:
            df[freq_col_name] = 0

    # 5. Split IP thành 4 octet (4 feature riêng) thay vì một integer lớn
    for ip_col in ['SrcAddr', 'DstAddr']:
        if ip_col in df.columns:
            # Chuyển sang string để split theo '.'
            ip_str = df[ip_col].astype(str)
            parts = ip_str.str.split('.', expand=True)

            # Đảm bảo có đủ 4 cột (nếu IP không chuẩn, phần thiếu sẽ là NaN)
            for i in range(4):
                if i < parts.shape[1]:
                    df[f'{ip_col}_octet_{i+1}'] = pd.to_numeric(parts[i], errors='coerce').fillna(0)
                else:
                    # Nếu thiếu cột, tạo cột toàn 0
                    df[f'{ip_col}_octet_{i+1}'] = 0

    # 6. Handle Port Columns (Keep as numeric features)
    for col in ['Sport', 'Dport']:
        if col in df.columns:
            # Force numeric (handle hex or strings)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 7. Drop raw IP addresses (đã có tần suất + 4 octet)
    df = df.drop(columns=['SrcAddr', 'DstAddr'], errors='ignore')
    
    # 8. One-Hot Encoding for 'Proto'
    if 'Proto' in df.columns:
        # Common protos: TCP, UDP, ICMP
        # Limit cardinality if needed? Usually low.
        dummies = pd.get_dummies(df['Proto'], prefix='Proto')
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=['Proto'])
        
    # 8. One-Hot Encoding for 'State' (không giới hạn top state)
    if 'State' in df.columns:
        dummies = pd.get_dummies(df['State'], prefix='State')
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=['State'])
        
    # 9. Handle 'Dir'
    if 'Dir' in df.columns:
         dummies = pd.get_dummies(df['Dir'], prefix='Dir')
         df = pd.concat([df, dummies], axis=1)
         df = df.drop(columns=['Dir'])
         
    # 10. Align columns with expected_columns
    if expected_columns is not None:
        # Add missing columns with 0
        # Use reindex for efficiency and handling both missing and extra
        df = df.reindex(columns=expected_columns, fill_value=0)
    else:
        # First time detection: Fill NaNs just in case
        df = df.fillna(0)
        
    # Return values
    X_vals = df.values.astype(np.float32) # Ensure float32 for PyTorch
    y_vals = y.values if y is not None else None
    
    return X_vals, y_vals, df.columns.tolist()

def save_global_stats(global_stats, filepath='global_stats.pkl'):
    """Saves global statistics to a pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(global_stats, f)
    print(f"Global stats saved to {filepath}")

def load_global_stats(filepath='global_stats.pkl'):
    """Loads global statistics from a pickle file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found.")
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_scaler(scaler, filepath='scaler.pkl'):
    """Saves the scaler to a pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {filepath}")

def load_scaler(filepath='scaler.pkl'):
    """Loads the scaler from a pickle file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found.")
    with open(filepath, 'rb') as f:
        return pickle.load(f)
