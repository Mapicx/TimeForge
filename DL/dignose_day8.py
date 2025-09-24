import pandas as pd

# --- Configuration (Hardcoded for this specific analysis) ---
# Removed the need for a separate config.py file

# 1. Set the path to the specific data file you want to analyze
DATA_PATH = 'data/clock_bias_correction_day_8_jan_2024.csv'

# 2. Set the name of the column you want to get statistics for
TARGET_VARIABLE = 'clock_bias_correction' # <-- THIS LINE HAS BEEN CHANGED

# 3. Define the train/validation/test split ratios
#    (e.g., 0.20 means 20% of the data)
TEST_SIZE = 0.20
VAL_SIZE = 0.10


def analyze_variance():
    """
    Loads the specified dataset, splits it into train/val/test sets,
    and prints key statistics for the target variable.
    """
    print(f"📄 Analyzing data from: {DATA_PATH}")
    print(f"🎯 Target variable: {TARGET_VARIABLE}\n")

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ ERROR: Data file not found at '{DATA_PATH}'.")
        print("Please ensure the path is correct and the file exists.")
        return
    except KeyError:
        print(f"❌ ERROR: Target column '{TARGET_VARIABLE}' not found in the CSV.")
        print(f"   Available columns are: {df.columns.tolist()}")
        return


    # --- Replicate data splitting logic ---
    n = len(df)
    test_split_index = int(n * (1 - TEST_SIZE))
    # The validation split is a percentage of the *remaining* (non-test) data
    val_split_index = int(test_split_index * (1 - VAL_SIZE / (1 - TEST_SIZE)))

    train_df = df[:val_split_index]
    val_df = df[val_split_index:test_split_index]
    test_df = df[test_split_index:]

    splits = [
        ('Training Set', train_df),
        ('Validation Set', val_df),
        ('Test Set', test_df)
    ]

    # --- Calculate and Print Statistics for each split ---
    for name, split_df in splits:
        print(f"--- 📊 Statistics for {name} ---")
        
        if split_df.empty:
            print("   - This data split is empty.\n")
            continue
            
        target_series = split_df[TARGET_VARIABLE]
        
        # Calculate statistics
        variance = target_series.var()
        std_dev = target_series.std()
        mean = target_series.mean()
        min_val = target_series.min()
        max_val = target_series.max()
        count = len(target_series)
        
        # Print in a readable format
        print(f"   - Record Count:   {count:,}")
        print(f"   - Mean:           {mean:.6f}")
        print(f"   - Variance:       {variance:.6f}")
        print(f"   - Std Deviation:  {std_dev:.6f}")
        print(f"   - Min Value:      {min_val:.6f}")
        print(f"   - Max Value:      {max_val:.6f}\n")

if __name__ == '__main__':
    analyze_variance()