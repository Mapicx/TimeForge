import pandas as pd
import glob
import os

# Define the date range (1st to 7th January 2024)
dates = [(f"{i}_jan_2024", f"{i:02d}") for i in range(1, 8)]  # (filename_part, day_number)

# Create list to store individual DataFrames
dataframes = []

for date_str, day_num in dates:
    filename = f"data/clock_bias_correction_day_{date_str}.csv"
    
    if os.path.exists(filename):
        # Read CSV file
        df = pd.read_csv(filename)
        
        # Add a column to track original date (optional)
        df['source_date'] = f"2024-01-{day_num}"
        
        dataframes.append(df)
        print(f"Successfully read {filename}")
    else:
        print(f"Warning: {filename} not found")

# Combine all DataFrames
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Sort by date if you have a timestamp column (adjust column name as needed)
    # If your files don't have a timestamp, they'll stay in 1st-7th order
    if 'timestamp' in combined_df.columns:
        combined_df.sort_values('timestamp', inplace=True)
    
    # Save combined file
    output_filename = "clock_bias_correction_combined_1_7_jan_2024.csv"
    combined_df.to_csv(output_filename, index=False)
    print(f"\nCombined file saved as {output_filename}")
    print(f"Total rows: {len(combined_df)}")
else:
    print("No files were found to combine")