import os
import torch

# =============================================================================
# SET WORKING DIRECTORY
# =============================================================================
WORKING_DIR = '.' 
os.makedirs(WORKING_DIR, exist_ok=True)

# =============================================================================
# SYSTEM RESOURCES & DEVICE
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# CONFIGURATION
# =============================================================================
BATCH_SIZE = 256
N_WORKERS = 8 # DataLoader Workers

# Check RAM (optional logic can be here or main)
import psutil
ram_gb = psutil.virtual_memory().total / 1e9

# Training config
N_EPOCHS = 6
LEARNING_RATE = 0.0001
STATE_TOP_N = 15
IMAGE_SIZE = 32

# =============================================================================
# DEFINING SCENARIOS
# =============================================================================
# Train scenarios: 
TRAIN_SCENARIOS = ['3','4', '11']

# Test scenario: Rbot
TEST_SCENARIOS = ['10']

# =============================================================================
# LABEL MAPPING
# =============================================================================
CLASS_TO_IDX = {
    'Botnet': 0,
    'C&C': 1,
    'Normal': 2
}

# Inverse mapping
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
