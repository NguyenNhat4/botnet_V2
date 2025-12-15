import pandas as pd
import numpy as np
from tqdm.auto import tqdm

def quick_classify(label):
    """
    Classifies a label string into 'Botnet', 'C&C', or 'Normal'.
    """
    if not isinstance(label, str):
        return 'Normal'
    label_lower = label.lower()
    if 'botnet' in label_lower:
        return 'Botnet'
    elif 'c&c' in label_lower or 'cc' in label_lower:
        return 'C&C'
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

def process_batch_fast_v2(chunk, top_states, freq_dicts, expected_columns=None):
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

    # 5. Handle Port Columns (Keep as numeric features)
    for col in ['Sport', 'Dport']:
        if col in df.columns:
            # Force numeric (handle hex or strings)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 6. Drop IP addresses (High cardinality, replaced by freq)
    df = df.drop(columns=['SrcAddr', 'DstAddr'], errors='ignore')
    
    # 7. One-Hot Encoding for 'Proto'
    if 'Proto' in df.columns:
        # Common protos: TCP, UDP, ICMP
        # Limit cardinality if needed? Usually low.
        dummies = pd.get_dummies(df['Proto'], prefix='Proto')
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=['Proto'])
        
    # 8. One-Hot Encoding for 'State' using top_states
    if 'State' in df.columns:
        # Keep only top states, others -> 'Other'
        # Ensure top_states is valid list
        if top_states:
            df['State'] = df['State'].apply(lambda x: x if x in top_states else 'Other')
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
