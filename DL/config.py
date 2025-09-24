import torch

# --- Training & Data Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = 'Joined_data_final/clock_bias_correction_combined_1_7_jan_2024.csv'

# --- 🎯 TARGET COLUMN ---
# This is the column the model will learn to predict.
TARGET = 'clock_bias_correction'


# --- ⭐ FEATURE COLUMNS ---
# This is the "Type 1" input feature set from the paper[cite: 291].
FEATURES = [
    'broadcast_clock_bias_scaled',
    'PRN_encoded'  # We create this from the 'PRN' column
]

# --- ⭐ ADVANCED FEATURE COLUMNS (TYPE 2) ---
# To use these, you must first generate and add the 'shadow_status'
# and 'sun_gravity_effect_scaled' columns to your CSV file.
# The 'shadow_status' column should contain categorical data (e.g., 'InShadow', 'OutShadow').
# The data_loader will one-hot encode it into the three binary columns below.
# This aligns with the "Type 2" inputs that gave the researchers a 15-20% accuracy boost[cite: 573].
FEATURES_TYPE_2 = [
    'broadcast_clock_bias_scaled',
    'SunGravity',
    'MoonGravity',
    'InShadow_no'
]


# --- Train/Val/Test split ratio ---
TEST_SIZE = 0.2
VAL_SIZE = 0.15

# --- Sequence lengths based on the paper (12 hours lookback, 2 hours forecast) ---
# Assuming 30-second intervals: 12 hours = 1440 steps, 2 hours = 240 steps[cite: 294, 372].
INPUT_SEQ_LEN = 1440
OUTPUT_SEQ_LEN = 240

# --- Model Hyperparameters (Tunable via command line or tune.py) ---
# These are starting points. The paper found optimal values through extensive search[cite: 412, 414].
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 256
DROPOUT = 0.1

# --- Training Hyperparameters (Tunable via command line or tune.py) ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 10