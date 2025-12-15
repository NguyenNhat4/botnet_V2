import os
import ssl
import urllib.request
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

def create_directory(path):
    os.makedirs(path, exist_ok=True)
    print(f"  Created/Checked: {path}")

def download_file(url, destination):
    try:
        if os.path.exists(destination):
            print(f"  [SKIP] File exists: {os.path.basename(destination)}")
            return True

        print(f"  Downloading: {os.path.basename(destination)}")
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="  Progress") as t:
            def reporthook(blocknum, blocksize, totalsize):
                t.total = totalsize
                t.update(blocknum * blocksize - t.n)
            urllib.request.urlretrieve(url, destination, reporthook=reporthook)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def check_csv_in_folder(folder_path):
    if not os.path.exists(folder_path): return False
    for file in os.listdir(folder_path):
        if file.endswith('.csv'): return True
    return False

def rename(path_file, new_name):
    dir_path = os.path.dirname(path_file)
    path_new_name = os.path.join(dir_path, new_name)
    os.rename(path_file, path_new_name)

def get_csv_paths(main_dir, scenario_ids):
    csv_paths = []
    for sid in scenario_ids:
        path = os.path.join(main_dir, sid)
        # Find csv file in folder
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith('.csv'):
                    csv_paths.append(os.path.join(path, file))
                    break
    return csv_paths

def plot_and_save_loss(train_losses, valid_losses, save_path):
    """
    Plots Loss chart and saves to file.
    """
    plt.figure(figsize=(10, 6))

    # Plot Train Loss
    plt.plot(train_losses, label='Train Loss', color='blue', marker='o', markersize=4)

    # Plot Valid Loss
    plt.plot(valid_losses, label='Valid Loss', color='orange', marker='o', markersize=4)

    # Decorate
    plt.title('Training vs Validation Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save image
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[INFO] Saved Loss plot at: {save_path}")

    # Show (optional in script mode, but harmless)
    # plt.show() 
    plt.close()
