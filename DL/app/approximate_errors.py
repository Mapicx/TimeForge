import pandas as pd
import numpy as np

# --- 1. Load Your Data ---
# Define the file path for your data.
file_path = 'data/GNSS_cleaned.csv' # Make sure this file is in the same directory as the script

# Load the CSV into a pandas DataFrame.
try:
    df = pd.read_csv(file_path)
    print("File loaded successfully!")
    print("Original DataFrame columns:", df.columns.tolist())
    print("-" * 30)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file path.")
    exit()

# --- 2. Define Functions for Approximate Errors ---

def approximate_ephemeris_error(pr_error):
    """
    Approximates ephemeris error as a noisy fraction of the total Pr_Error.

    This is a prototyping function that assumes ephemeris error contributes
    about 30% (+/- 10%) to the total pseudorange error.
    """
    # Use the absolute value of pr_error to avoid negative error components
    base_error = abs(pr_error)
    
    # Calculate as 30% of the error, with some randomness (e.g., from 20% to 40%)
    random_fraction = np.random.uniform(0.20, 0.40)
    return base_error * random_fraction

def approximate_clock_error(pr_error):
    """
    Approximates clock error as a noisy fraction of the total Pr_Error.

    This is a prototyping function that assumes clock error contributes
    about 20% (+/- 10%) to the total pseudorange error.
    """
    # Use the absolute value of pr_error to avoid negative error components
    base_error = abs(pr_error)
    
    # Calculate as 20% of the error, with some randomness (e.g., from 10% to 30%)
    random_fraction = np.random.uniform(0.10, 0.30)
    return base_error * random_fraction

# --- 3. Apply the Functions to Separate the Errors ---

# Check if 'Pr_Error' column exists before proceeding.
if 'Pr_Error' in df.columns:
    print("Approximating errors... This may take a moment for large files.")

    # Create new columns by applying the approximation functions.
    # We use a lambda function to pass the 'Pr_Error' of each row to our functions.
    df['ephemeris_error_approx_m'] = df['Pr_Error'].apply(approximate_ephemeris_error)
    df['clock_error_approx_m'] = df['Pr_Error'].apply(approximate_clock_error)

    # Calculate the remaining error after subtracting the approximated components.
    # Note: We subtract from the original Pr_Error, which can be negative.
    df['remaining_error_m'] = df['Pr_Error'] - df['ephemeris_error_approx_m'] - df['clock_error_approx_m']

    # --- 4. Display the Results ---
    print("\nProcessing complete!")
    print("Approximated error columns have been added.")

    # Display the first few rows with the relevant columns to show the result.
    print(df[['Pr_Error', 'ephemeris_error_approx_m', 'clock_error_approx_m', 'remaining_error_m']].head())

    # --- 5. Save the New DataFrame (Optional) ---
    output_filename = 'GNSS_with_approximated_errors.csv'
    df.to_csv(output_filename, index=False)
    print(f"\nDataFrame with new error columns saved to '{output_filename}'")

else:
    print("Error: 'Pr_Error' column not found in the DataFrame.")