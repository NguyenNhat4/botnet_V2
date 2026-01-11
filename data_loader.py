import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from preprocessing_utils import process_batch_fast_v2
from config import CLASS_TO_IDX, IMAGE_SIZE


class FastBotnetDataset(Dataset):
    """
    Dataset:
      - Nhận X_data dạng [N, n_features]
      - Mỗi dòng được pad / cắt về 32x32 và reshape thành ảnh 1x32x32
    """

    def __init__(self, X_data, y_data):
        self.X_data = torch.from_numpy(X_data).float()
        self.y_data = torch.from_numpy(y_data).long()

        self.image_size = IMAGE_SIZE
        self.num_pixels = self.image_size * self.image_size

        # Pre-compute Hilbert curve coordinates for mapping 1D features -> 2D ảnh
        self.hilbert_coords = self._hilbert_curve(self.image_size)

    def __len__(self):
        return len(self.X_data)

    def _hilbert_curve(self, n):
        """
        Sinh danh sách tọa độ (row, col) theo đường cong Hilbert cho lưới n x n.
        n phải là lũy thừa của 2 (ở đây n = 32).
        Thuật toán đơn giản, ưu tiên tính dễ đọc hơn là tối ưu tuyệt đối.
        """
        if n & (n - 1) != 0:
            raise ValueError("IMAGE_SIZE must be a power of 2 for Hilbert curve.")

        def hilbert_index_to_xy(d, order):
            """
            Chuyển chỉ số 1D d trên Hilbert order 'order' (2^order x 2^order)
            sang tọa độ (x, y).
            """
            x = y = 0
            t = d
            s = 1
            while s < (1 << order):
                rx = 1 & (t // 2)
                ry = 1 & (t ^ rx)
                # xoay / phản chiếu
                if ry == 0:
                    if rx == 1:
                        x, y = s - 1 - x, s - 1 - y
                    x, y = y, x
                x += s * rx
                y += s * ry
                t //= 4
                s <<= 1
            return x, y

        order = int(np.log2(n))
        coords = []
        for d in range(n * n):
            x, y = hilbert_index_to_xy(d, order)
            coords.append((x, y))
        return coords

    def __getitem__(self, idx):
        # vector đặc trưng 1D: [n_features]
        feature_vector = self.X_data[idx]

        # Khởi tạo ảnh 0 (0..255), sẽ điền theo đường cong Hilbert
        image = torch.zeros((1, self.image_size, self.image_size), dtype=feature_vector.dtype)

        # Số feature thực sự có (không bỏ dòng nào, chỉ giới hạn số pixel)
        length = min(feature_vector.numel(), self.num_pixels)

        # Map từng feature lên pixel theo Hilbert curve để giữ locality
        for i in range(length):
            x, y = self.hilbert_coords[i]
            image[0, x, y] = feature_vector[i]

        # Scale per-sample về khoảng [0, 255]
        vals = image.view(-1)
        vals = torch.clamp(vals, min=0)
        min_val = vals.min()
        max_val = vals.max()
        if max_val > min_val:
            vals = (vals - min_val) / (max_val - min_val)
        else:
            vals = vals * 0.0
        vals = vals * 255.0
        image = vals.view(1, self.image_size, self.image_size)

        return image, self.y_data[idx]

def load_data_from_csvs(csv_list, global_stats, desc="Loading", is_train=True, scaler=None):
    """
    Loads data from a list of CSVs, processes it, and returns X and y.
    """
    X_list = []
    y_list = []

    # Get stats
    freq_dicts = global_stats['freq_dicts']
    expected_cols = global_stats['expected_columns']

    for csv_file in tqdm(csv_list, desc=desc):
        try:
            for chunk in pd.read_csv(csv_file, chunksize=100000, low_memory=False):
                X_batch, y_batch, _ = process_batch_fast_v2(
                    chunk, freq_dicts, expected_cols
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
    # 1. Sanitize input: Replace existing NaNs/Infs with 0
    X_data = np.nan_to_num(X_data, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Clip negative values to 0. 
    # This prevents np.log1p(-1) from becoming -inf.
    X_data = np.maximum(X_data, 0)

    # Normalize (Log1p + Scaler)
    X_data = np.log1p(X_data)

    if scaler:
        if is_train:
            X_data = scaler.fit_transform(X_data)
        else:
            X_data = scaler.transform(X_data)

    return X_data, y_data
