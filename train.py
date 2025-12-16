import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils import class_weight
from tqdm.auto import tqdm
from loss import FocalLoss
try:
    from torchinfo import summary
except ImportError:
    summary = None

from config import (
    WORKING_DIR, BATCH_SIZE, N_WORKERS, N_EPOCHS, LEARNING_RATE, 
    STATE_TOP_N, IMAGE_SIZE, TRAIN_SCENARIOS, TEST_SCENARIOS, 
    CLASS_TO_IDX, device
)
from utils import (
    create_directory, download_file, check_csv_in_folder, 
    rename, get_csv_paths, plot_and_save_loss
)
from preprocessing_utils import (
    quick_classify, calculate_global_frequencies, process_batch_fast_v2,
    save_global_stats, save_scaler
)
from model import BotnetClassifier
from data_loader import FastBotnetDataset, load_data_from_csvs

def main():
    print(f"Working directory: {os.getcwd()}")
    print("="*60)
    print("SYSTEM RESOURCES")
    print("="*60)
    
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
    print(f"\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {N_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  DataLoader Workers: {N_WORKERS}")
    print("="*60)

    # =============================================================================
    # 1. DOWNLOAD DATASET
    # =============================================================================
    main_dir = './CTU-13-Dataset/'
    create_directory(main_dir)

    for i in range(1, 14):
        create_directory(os.path.join(main_dir, str(i)))

    datasets = [
        (1, 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/detailed-bidirectional-flow-labels/capture20110810.binetflow'),
        (2, 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-43/detailed-bidirectional-flow-labels/capture20110811.binetflow'),
        (3, 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-44/detailed-bidirectional-flow-labels/capture20110812.binetflow'),
        (4, 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-45/detailed-bidirectional-flow-labels/capture20110815.binetflow'),
        (5, 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-46/detailed-bidirectional-flow-labels/capture20110815-2.binetflow'),
        (6, 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-47/detailed-bidirectional-flow-labels/capture20110816.binetflow'),
        (7, 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-48/detailed-bidirectional-flow-labels/capture20110816-2.binetflow'),
        (8, 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-49/detailed-bidirectional-flow-labels/capture20110816-3.binetflow'),
        (9, 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-50/detailed-bidirectional-flow-labels/capture20110817.binetflow'),
        (10, 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-51/detailed-bidirectional-flow-labels/capture20110818.binetflow'),
        (11, 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-52/detailed-bidirectional-flow-labels/capture20110818-2.binetflow'),
        (12, 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-53/detailed-bidirectional-flow-labels/capture20110819.binetflow'),
        (13, 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-54/detailed-bidirectional-flow-labels/capture20110815-3.binetflow')
    ]

    print(f"\nStarting download for {len(datasets)} datasets...")
    for idx, url in datasets:
        filename = url.split('/')[-1]
        destination = os.path.join(main_dir, str(idx), filename)
        folder_path = os.path.join(main_dir, str(idx))

        print(f"\n[{idx}/13] Dataset {idx}:")
        if check_csv_in_folder(folder_path):
            print(f"  [SKIP] CSV already exists.")
            continue
        download_file(url, destination)
    print("\nDownload complete!")

    # =============================================================================
    # 2. CONVERT BINETFLOW TO CSV
    # =============================================================================
    listDir = os.listdir(main_dir)
    listCSV = []
    print("Checking/Converting files...")

    for subDir in sorted(listDir, key=lambda x: int(x) if x.isdigit() else 0):
        path_subDir = os.path.join(main_dir, subDir)
        if not os.path.isdir(path_subDir): continue

        # Check if CSV exists
        csv_files = [f for f in os.listdir(path_subDir) if f.endswith('.csv')]
        if csv_files:
            listCSV.append(os.path.join(path_subDir, csv_files[0]))
            continue

        # If no CSV, look for binetflow to rename
        binetflow_files = [f for f in os.listdir(path_subDir) if 'binetflow' in f]
        if binetflow_files:
            binetflow_file = os.path.join(path_subDir, binetflow_files[0])
            new_name = subDir + '.csv'
            rename(binetflow_file, new_name)
            listCSV.append(os.path.join(path_subDir, new_name))
            print(f"  Converted {binetflow_files[0]} -> {new_name}")

    print(f"\nFound {len(listCSV)} CSV files:")
    for csv in listCSV:
        print(f"  {csv}")

    # =============================================================================
    # 3. IDENTIFY SCENARIOS
    # =============================================================================
    train_csvs = get_csv_paths(main_dir, TRAIN_SCENARIOS)
    test_csvs = get_csv_paths(main_dir, TEST_SCENARIOS)
    
    print(f"Found {len(train_csvs)} training files.")
    print(f"Found {len(test_csvs)} testing files.")

    if not train_csvs:
        print("No training files found! Exiting.")
        return

    # =============================================================================
    # 4. PRE-COMPUTE GLOBAL STATISTICS
    # =============================================================================
    print("="*70)
    print("PRE-COMPUTING GLOBAL STATISTICS")
    print("="*70)
    
    target_csvs = train_csvs 

    # --- Step 1: Calculate Global Frequencies ---
    print("\n[1/4] Calculating global IP/Port frequencies...")
    freq_dicts = calculate_global_frequencies(target_csvs)

    # --- Step 2: Detect Top States ---
    print("\n[2/4] Detecting top states...")
    try:
        sample_df = pd.read_csv(target_csvs[0], nrows=100000, low_memory=False)
        sample_df['Label'] = sample_df['Label'].apply(quick_classify)
        top_states = sample_df['State'].value_counts().nlargest(STATE_TOP_N).index.tolist()
        print(f"  Top {STATE_TOP_N} states: {top_states}")
        del sample_df
        gc.collect()
    except Exception as e:
        print(f"Error detecting top states: {e}")
        top_states = []

    # --- Step 3: Detect Column Schema ---
    print("\n[3/4] Detecting column schema...")
    expected_columns = None
    cols_samples = []

    for csv_file in target_csvs[:5]:
        try:
            chunk = pd.read_csv(csv_file, nrows=5000, low_memory=False)
            X_s, y_s, cols_s = process_batch_fast_v2(chunk, top_states, freq_dicts, expected_columns=None)
            if cols_s:
                cols_samples.extend(cols_s)
        except Exception as e:
            continue

    if cols_samples:
        expected_columns = list(dict.fromkeys(cols_samples))  # Preserve order, remove duplicates
        print(f"  Detected {len(expected_columns)} feature columns")
    else:
        print("  WARNING: Could not detect column schema!")
        return

    # --- Step 4: Scan and Count (Skipped Min/Max per original code being complex) ---
    # We rely on RobustScaler later.
    
    n_features = len(expected_columns)
    
    global_stats = {
        'freq_dicts': freq_dicts,
        'top_states': top_states,
        'expected_columns': expected_columns,
        'n_features': n_features
    }

    # =============================================================================
    # 5. LOAD DATA INTO RAM
    # =============================================================================
    scaler = RobustScaler()
    
    print("\nLoading TRAINING Data...")
    X_train, y_train = load_data_from_csvs(train_csvs, global_stats, desc="Train Data", is_train=True, scaler=scaler)
    
    # Save statistics and scaler after processing training data
    print("\nSaving Global Statistics and Scaler...")
    save_global_stats(global_stats)
    save_scaler(scaler)

    print("\nLoading TESTING Data...")
    X_test, y_test = load_data_from_csvs(test_csvs, global_stats, desc="Test Data", is_train=False, scaler=scaler)

    # Feature Selection: Remove IP frequencies? 
    # Logic from notebook:
    drop_cols = ['Src_freq', 'Dst_freq', 'Sport_freq', 'Dport_freq']
    all_cols = global_stats['expected_columns']
    
    keep_indices = [i for i, col in enumerate(all_cols) if col not in drop_cols]
    keep_cols = [col for i, col in enumerate(all_cols) if i in keep_indices]

    if len(keep_indices) < len(all_cols):
        print(f"  Dropping {len(drop_cols)} columns: {drop_cols}")
        X_train = X_train[:, keep_indices]
        X_test = X_test[:, keep_indices]
        n_features = len(keep_indices)
        print(f"  Features remaining: {n_features}")
    else:
        print("  No columns dropped.")

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")

    # =============================================================================
    # 6. MODEL SET UP
    # =============================================================================
    
    # # Class Weights
   
    # Validation Split
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    train_ds = FastBotnetDataset(X_train_final, y_train_final)
    
    print("Using focal loss")
   
   
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0 # Avoid multiprocessing issues in some envs
    )

    valid_ds = FastBotnetDataset(X_val, y_val)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    test_ds = FastBotnetDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    print("Initializing 1D CNN Model...")
    model = BotnetClassifier(base_model=None, n_features=n_features, n_classes=len(CLASS_TO_IDX))
    model = model.to(device)

    if summary:
        summary(model, input_size=(BATCH_SIZE, n_features))
    else:
        print(model)

    # Compute class weights to handle class imbalance (sklearn >= 1.4 requires 'class_weight' arg)
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    print("#" * 50,"class_weights")
    print(class_weights)
    weights_tensor = torch.tensor(class_weights,dtype=torch.float32).to(device)
    criterion = FocalLoss(weight=weights_tensor,gamma=4.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # =============================================================================
    # 7. TRAINING LOOP
    # =============================================================================
    train_losses = []
    valid_losses = []
    
    best_val_loss = float('inf')

    print(f"Starting training on {device}...")
    
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        # Training
        for X_batch, y_batch in tqdm(train_loader, desc=f"Ep {epoch}/{N_EPOCHS} [Train]", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in tqdm(valid_loader, desc=f"Ep {epoch}/{N_EPOCHS} [Val]", leave=False):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
        epoch_val_loss = val_loss / len(valid_loader.dataset)
        valid_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch}: Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        # Checkpoint
        if epoch_val_loss < best_val_loss:
            print(f"  Val loss decreased ({best_val_loss:.4f} -> {epoch_val_loss:.4f}). Saving model...")
            torch.save(model.state_dict(), 'best_model.pth')
            best_val_loss = epoch_val_loss
        plot_and_save_loss(train_losses, valid_losses, f'training_history_loss_{N_EPOCHS}.png')
    # =============================================================================
    # 8. RESULTS
    # =============================================================================
    print("Training Complete.")

if __name__ == "__main__":
    main()
