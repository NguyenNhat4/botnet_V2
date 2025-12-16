import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import gc

from config import (
    WORKING_DIR, BATCH_SIZE, TRAIN_SCENARIOS, TEST_SCENARIOS, 
    CLASS_TO_IDX, IDX_TO_CLASS, STATE_TOP_N, device
)
from utils import get_csv_paths, create_directory, check_csv_in_folder, download_file, rename
from preprocessing_utils import (
    quick_classify, calculate_global_frequencies, process_batch_fast_v2,
    save_global_stats, load_global_stats, save_scaler, load_scaler
)
from model import BotnetClassifier
from data_loader import FastBotnetDataset, load_data_from_csvs

def compute_stats_from_train(main_dir):
    """
    Recomputes global stats and scaler from training scenarios.
    """
    print("Recomputing statistics from training data...")
    train_csvs = get_csv_paths(main_dir, TRAIN_SCENARIOS)
    
    if not train_csvs:
        print("No training files found! Please check TRAIN_SCENARIOS and dataset.")
        return None, None

    # --- Step 1: Calculate Global Frequencies ---
    print("\n[1/3] Calculating global IP/Port frequencies...")
    freq_dicts = calculate_global_frequencies(train_csvs)

    # --- Step 2: Detect Top States ---
    print("\n[2/3] Detecting top states...")
    top_states = []
    try:
        sample_df = pd.read_csv(train_csvs[0], nrows=100000, low_memory=False)
        sample_df['Label'] = sample_df['Label'].apply(quick_classify)
        top_states = sample_df['State'].value_counts().nlargest(STATE_TOP_N).index.tolist()
        print(f"  Top {STATE_TOP_N} states: {top_states}")
        del sample_df
        gc.collect()
    except Exception as e:
        print(f"Error detecting top states: {e}")

    # --- Step 3: Detect Column Schema ---
    print("\n[3/3] Detecting column schema...")
    expected_columns = None
    cols_samples = []

    for csv_file in train_csvs[:5]:
        try:
            chunk = pd.read_csv(csv_file, nrows=5000, low_memory=False)
            X_s, y_s, cols_s = process_batch_fast_v2(chunk, top_states, freq_dicts, expected_columns=None)
            if cols_s:
                cols_samples.extend(cols_s)
        except Exception as e:
            continue

    if cols_samples:
        expected_columns = list(dict.fromkeys(cols_samples))
        print(f"  Detected {len(expected_columns)} feature columns")
    else:
        print("  WARNING: Could not detect column schema!")
        return None, None

    global_stats = {
        'freq_dicts': freq_dicts,
        'top_states': top_states,
        'expected_columns': expected_columns,
        'n_features': len(expected_columns)
    }

    # --- Step 4: Fit Scaler ---
    print("\nFitting Scaler on Training Data...")
    scaler = RobustScaler()
    # We need to load training data to fit scaler. 
    # This might be heavy, but necessary for correct evaluation if stats are missing.
    X_train, _ = load_data_from_csvs(train_csvs, global_stats, desc="Fitting Scaler", is_train=True, scaler=scaler)
    
    # Save for future
    save_global_stats(global_stats)
    save_scaler(scaler)
    
    # Clean up RAM
    del X_train
    gc.collect()

    return global_stats, scaler

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Evaluating"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix_vietnamese(cm, classes, save_path):
    plt.figure(figsize=(10, 8))
    
    # Map class names to Vietnamese if desired, or keep English but title in VN
    # Common mappings: Normal -> Bình thường, Botnet -> Botnet, C&C -> C&C
    # Let's keep class names technical but title/labels in VN.
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Ma trận nhầm lẫn (Confusion Matrix)', fontsize=16)
    plt.ylabel('Nhãn thực tế (True Label)', fontsize=12)
    plt.xlabel('Nhãn dự đoán (Predicted Label)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Botnet Detection Model")
    parser.add_argument('--recompute-stats', action='store_true', help="Recompute global stats and scaler from training data instead of loading.")
    args = parser.parse_args()

    print(f"Device: {device}")
    
    # Check/Download Data (Reuse logic if needed, assuming data exists in CTU-13-Dataset)
    main_dir = './CTU-13-Dataset/'
    if not os.path.exists(main_dir):
        print("Dataset directory not found. Please run train.py to download data first.")
        return

    # 1. Load Stats and Scaler
    if args.recompute_stats or not os.path.exists('global_stats.pkl') or not os.path.exists('scaler.pkl'):
        global_stats, scaler = compute_stats_from_train(main_dir)
        if global_stats is None:
            return
    else:
        print("Loading existing statistics and scaler...")
        global_stats = load_global_stats()
        scaler = load_scaler()
    
    if global_stats is None:
        print("Failed to obtain global statistics.")
        return

    # 2. Load Test Data
    test_csvs = get_csv_paths(main_dir, TEST_SCENARIOS)
    print(f"Found {len(test_csvs)} testing files for evaluation.")
    
    # Handle Feature Selection (match train.py logic)
    drop_cols = ['Src_freq', 'Dst_freq', 'Sport_freq', 'Dport_freq']
    all_cols = global_stats['expected_columns']
    keep_indices = [i for i, col in enumerate(all_cols) if col not in drop_cols]
    
    # Load data
    X_test, y_test = load_data_from_csvs(test_csvs, global_stats, desc="Loading Test Data", is_train=False, scaler=scaler)
    
    if len(keep_indices) < len(all_cols):
        X_test = X_test[:, keep_indices]
        n_features = len(keep_indices)
    else:
        n_features = global_stats['n_features']
        
    print(f"Test Data Shape: {X_test.shape}")

    # 3. Load Model
    model_path = 'best_model.pth'
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return
        
    model = BotnetClassifier(base_model=None, n_features=n_features, n_classes=len(CLASS_TO_IDX))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print("Model loaded successfully.")

    # 4. Evaluate
    test_ds = FastBotnetDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # 0 workers for safety
    
    y_true, y_pred = evaluate_model(model, test_loader, device)

    # Quick sanity check: đếm số mẫu theo lớp trong ground-truth và dự đoán
    label_indices = list(range(len(CLASS_TO_IDX)))
    true_counts = {IDX_TO_CLASS[i]: int((y_true == i).sum()) for i in label_indices}
    pred_counts = {IDX_TO_CLASS[i]: int((y_pred == i).sum()) for i in label_indices}
    print("\nPhân bố nhãn (ground-truth):", true_counts)
    print("Phân bố nhãn (dự đoán):     ", pred_counts)

    # 5. Metrics & Reporting (Vietnamese)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Use a fixed label order based on CLASS_TO_IDX so that
    # confusion matrix and classification report are consistent
    label_indices = list(range(len(CLASS_TO_IDX)))
    cm = confusion_matrix(y_true, y_pred, labels=label_indices)
    
    # Prepare output directory
    metrics_dir = 'metrics'
    create_directory(metrics_dir)
    
    report_path = os.path.join(metrics_dir, 'evaluation_report.txt')
    cm_path = os.path.join(metrics_dir, 'confusion_matrix.png')
    
    report_content = [
        "BÁO CÁO ĐÁNH GIÁ MÔ HÌNH (MODEL EVALUATION REPORT)",
        "="*50,
        f"Kịch bản kiểm thử (Test Scenarios): {TEST_SCENARIOS}",
        f"Tổng số mẫu (Total Samples): {len(y_true)}",
        "-"*50,
        f"Độ chính xác (Accuracy): {accuracy:.4f}",
        f"Độ chính xác (Precision - Weighted): {precision:.4f}",
        f"Độ nhạy (Recall - Weighted): {recall:.4f}",
        f"Điểm F1 (F1-Score - Weighted): {f1:.4f}",
        "="*50,
        "Chi tiết theo lớp (Class-wise Details):"
    ]
    
    # Class-wise metrics
    # Re-map indices to class names (must match label_indices order)
    classes = [IDX_TO_CLASS[i] for i in label_indices]
    
    # Calculate per-class precision/recall/f1.
    # Explicitly pass labels so that number of labels matches target_names,
    # even if some classes do not appear in y_true/y_pred.
    from sklearn.metrics import classification_report
    cls_report = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    
    for cls in classes:
        metrics = cls_report.get(cls, {})
        report_content.append(f"\nLớp: {cls}")
        report_content.append(f"  Precision: {metrics.get('precision', 0):.4f}")
        report_content.append(f"  Recall:    {metrics.get('recall', 0):.4f}")
        report_content.append(f"  F1-Score:  {metrics.get('f1-score', 0):.4f}")
        
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
        
    print(f"\nReport saved to {report_path}")
    print('\n'.join(report_content))

    # Plot Confusion Matrix
    plot_confusion_matrix_vietnamese(cm, classes, cm_path)
    print(f"Confusion Matrix saved to {cm_path}")

if __name__ == "__main__":
    main()
