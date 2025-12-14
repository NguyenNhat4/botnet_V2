
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def quick_classify(label):
    """Fast label classification"""
    label_str = str(label)
    if 'Botnet' in label_str and 'CC' in label_str:
        return 'C&C'
    elif 'Botnet' in label_str:
        return 'Botnet'
    return 'Normal'

def calculate_global_frequencies(target_csvs, sample_size=100000):
    """Pre-calculate global frequencies for IPs and Ports"""
    print("Calculating global IP/Port frequencies...")

    all_src_ips = []
    all_dst_ips = []
    all_sports = []
    all_dports = []

    sample_limit = min(5, len(target_csvs))
    for csv_file in target_csvs[:sample_limit]:
        try:
            df_sample = pd.read_csv(csv_file, nrows=sample_size)
            all_src_ips.extend(df_sample['SrcAddr'].astype(str).tolist())
            all_dst_ips.extend(df_sample['DstAddr'].astype(str).tolist())
            all_sports.extend(df_sample['Sport'].fillna(0).tolist())
            all_dports.extend(df_sample['Dport'].fillna(0).tolist())
        except Exception as e:
            print(f"  Skipping {csv_file} in freq calc: {e}")

    src_ip_freq = pd.Series(all_src_ips).value_counts(normalize=True).to_dict()
    dst_ip_freq = pd.Series(all_dst_ips).value_counts(normalize=True).to_dict()
    sport_freq = pd.Series(all_sports).value_counts(normalize=True).to_dict()
    dport_freq = pd.Series(all_dports).value_counts(normalize=True).to_dict()

    print(f"  Unique Source IPs: {len(src_ip_freq)}")
    print(f"  Unique Dest IPs: {len(dst_ip_freq)}")
    print(f"  Unique Source Ports: {len(sport_freq)}")
    print(f"  Unique Dest Ports: {len(dport_freq)}")

    return src_ip_freq, dst_ip_freq, sport_freq, dport_freq

def process_batch_fast_v2(df_batch, top_states, freq_dicts, expected_columns=None):
    """Improved version with frequency encoding and column standardization"""
    try:
        df_batch = df_batch.dropna(subset=['Label', 'SrcAddr', 'DstAddr', 'Sport', 'Dport'])
        if 'StartTime' in df_batch.columns:
            df_batch = df_batch.drop('StartTime', axis=1)

        df_batch['Label'] = df_batch['Label'].apply(quick_classify)

        src_ip_freq, dst_ip_freq, sport_freq, dport_freq = freq_dicts

        df_batch['Src_freq'] = df_batch['SrcAddr'].astype(str).map(src_ip_freq).fillna(0.0001)
        df_batch['Dst_freq'] = df_batch['DstAddr'].astype(str).map(dst_ip_freq).fillna(0.0001)
        df_batch['Sport_freq'] = df_batch['Sport'].map(sport_freq).fillna(0.0001)
        df_batch['Dport_freq'] = df_batch['Dport'].map(dport_freq).fillna(0.0001)

        df_batch = df_batch.drop(['SrcAddr', 'DstAddr', 'Sport', 'Dport'], axis=1)

        df_batch['State'] = df_batch['State'].apply(lambda x: x if x in top_states else 'Other')

        for col in ['Proto', 'Dir', 'State']:
            if col in df_batch.columns:
                dummies = pd.get_dummies(df_batch[col], prefix=col, drop_first=False)
                df_batch = pd.concat([df_batch, dummies], axis=1)
                df_batch = df_batch.drop(col, axis=1)

        df_features = df_batch.drop('Label', axis=1)

        if expected_columns is not None:
            df_features = df_features.reindex(columns=expected_columns, fill_value=0)

        X = df_features.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
        y = df_batch['Label'].values

        return X, y, df_features.columns.tolist()
    except Exception as e:
        return np.array([]), np.array([]), []

def worker_scan_chunk(df_chunk, top_states, freq_dicts, expected_columns=None):
    """Worker for calculating Min/Max with column standardization"""
    try:
        X_batch, _, cols = process_batch_fast_v2(df_chunk, top_states, freq_dicts, expected_columns)
        if len(X_batch) == 0:
            return None, None, 0, []
        return np.min(X_batch, axis=0), np.max(X_batch, axis=0), len(X_batch), cols
    except Exception as e:
        return None, None, 0, []
