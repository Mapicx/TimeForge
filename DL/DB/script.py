import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
CSV_FILE_PATH = 'DL/DB/clock_bias_correction_combined_1_7_jan_2024.csv' # Corrected path syntax
TABLE_NAME = 'satellite_telemetry'
BATCH_SIZE = 500

# --- Script ---
print("Connecting to Supabase...")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print(f"Reading CSV file: {CSV_FILE_PATH}")
df = pd.read_csv(CSV_FILE_PATH)

# --- FIX IS HERE: Add this line ---
print("Converting all column names to lowercase to match database...")
df.columns = df.columns.str.lower()
# ------------------------------------

# Now the script will work correctly
records = df.to_dict(orient='records')
total_records = len(records)
print(f"Found {total_records} records to insert.")

for i in range(0, total_records, BATCH_SIZE):
    batch = records[i:i + BATCH_SIZE]
    print(f"Inserting batch {i//BATCH_SIZE + 1}...")
    try:
        # The keys in 'batch' will now be lowercase and match your Supabase table
        response = supabase.table(TABLE_NAME).insert(batch).execute()
    except Exception as e:
        print(f"An error occurred during batch insert: {e}")
        break # Stop the loop if an error occurs

print("✅ Data import finished.")