import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from preprocessing_utils import process_batch_fast_v2
from config import CLASS_TO_IDX

class FastBotnetDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = torch.from_numpy(X_data).float()
        self.y_data = torch.from_numpy(y_data).long()

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]

def load_data_from_csvs(csv_list, global_stats, desc="Loading", is_train=True, scaler=None):
    """
    Loads data from a list of CSVs, processes it, and returns X and y.
    """
    X_list = []
    y_list = []

    # Get stats
    top_states = global_stats['top_states']
    freq_dicts = global_stats['freq_dicts']
    expected_cols = global_stats['expected_columns']

    for csv_file in tqdm(csv_list, desc=desc):
        try:
            for chunk in pd.read_csv(csv_file, chunksize=100000, low_memory=False):
                X_batch, y_batch, _ = process_batch_fast_v2(
                    chunk, top_states, freq_dicts, expected_cols
                )
                if len(X_batch) > 0:
                    X_list.append(X_batch)
                    # Convert labels to indices
                    # y_batch is a Series or numpy array of strings
                    if y_batch is not None:
                         # Ensure y_batch contains valid keys from CLASS_TO_IDX
                        y_indices = np.array([CLASS_TO_IDX.get(label, 2) for label in y_batch]) # Default to Normal (2) if not found
                        y_list.append(y_indices)
        except Exception as e:
            print(f"  Error loading {csv_file}: {e}")

    if not X_list:
        return np.array([]), np.array([])

    X_data = np.vstack(X_list).astype(np.float32)
    y_data = np.concatenate(y_list).astype(np.int64)

    # Normalize (Log1p + Scaler)
    X_data = np.log1p(X_data)

    if scaler:
        if is_train:
            X_data = scaler.fit_transform(X_data)
        else:
            X_data = scaler.transform(X_data)

    return X_data, y_data
