# pipeline_fixed.py
# GNSS pipeline with JPL Horizons API ephemeris fetcher (replaces astroquery)
# Requires: pandas, numpy, scikit-learn, georinex, requests

import io
import re
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import georinex as gr
import requests

# ---------------------
# JPL Horizons API Fetcher (replaces astroquery)
# ---------------------
def get_jpl_ephemeris(target_body, start_time, stop_time, step):
    """
    Fetches ephemeris data for a specific celestial body from NASA JPL Horizons.
    """
    api_url = "https://ssd.jpl.nasa.gov/api/horizons.api"

    params = {
        'format': 'text',
        'COMMAND': f"'{target_body}'",
        'OBJ_DATA': 'NO',
        'MAKE_EPHEM': 'YES',
        'EPHEM_TYPE': 'VECTORS',
        'CENTER': "'@399'",
        'START_TIME': f"'{start_time}'",
        'STOP_TIME': f"'{stop_time}'",
        'STEP_SIZE': f"'{step}'",
        'VEC_TABLE': '3',
        'REF_PLANE': 'ECLIPTIC',
        'REF_SYSTEM': 'J2000',
        'OUT_UNITS': 'KM-S',
        'CSV_FORMAT': 'YES',
    }

    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()

        text_content = response.text
        csv_start_index = text_content.find('$$SOE')
        csv_end_index = text_content.find('$$EOE')

        if csv_start_index == -1 or csv_end_index == -1:
            print(f"Error: Could not find CSV data markers for target {target_body}.")
            return None

        csv_data = text_content[csv_start_index + len('$$SOE\n'):csv_end_index].strip()

        column_names = [
            'JDTDB', 'CalendarDate', 'X', 'Y', 'Z',
            'VX', 'VY', 'VZ', 'LT', 'RG', 'RR'
        ]

        df = pd.read_csv(
            io.StringIO(csv_data),
            names=column_names,
            dtype=str,
            header=None,
            sep=r'\s*,\s*',
            engine='python',
            index_col=False
        )

        df.columns = df.columns.str.strip()
        print(f"Columns for {target_body}: {df.columns.tolist()}")

        return df

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the API request: {e}")
        return None

# ---------------------
# Helper: Simplified shadow model
# ---------------------
def calculate_satellite_shadow(datetime_series):
    """Return a simple one-hot shadow classification (InShadow, OutShadow, Undetermined)."""
    hours = pd.to_datetime(datetime_series).dt.hour
    shadow_conditions = []
    for hour in hours:
        if 18 <= hour or hour < 6:
            shadow_conditions.append([1, 0, 0])  # InShadow
        else:
            shadow_conditions.append([0, 1, 0])  # OutShadow
    shadow_df = pd.DataFrame(shadow_conditions, columns=['InShadow', 'OutShadow', 'UndeterminedShadow'])
    return shadow_df

# ---------------------
# Ephemeris resampling function
# ---------------------
def _resample_ephemeris_to_30s(ephem_df, start_date, stop_date):
    """Resample ephemeris to 30-second epochs by time interpolation."""
    df = ephem_df.copy()
    if 'DateTime' not in df.columns:
        raise RuntimeError("ephem_df must contain 'DateTime' column to resample")
    df = df.set_index('DateTime').sort_index()

    start = pd.to_datetime(start_date)
    stop = pd.to_datetime(stop_date)
    full_idx = pd.date_range(start=start, end=stop, freq='30s')

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_reindexed = df.reindex(full_idx)
    if numeric_cols:
        df_reindexed[numeric_cols] = df_reindexed[numeric_cols].interpolate(method='time', limit_direction='both')
    df_reindexed = df_reindexed.reset_index().rename(columns={'index': 'DateTime'})
    return df_reindexed

# ---------------------
# Ephemeris fetcher using JPL API
# ---------------------
def fetch_and_process_ephemeris_data(start_date, stop_date, time_step='30s'):
    """
    Fetch Sun & Moon ephemeris using JPL Horizons API (replaces astroquery)
    """
    print("Fetching Sun and Moon ephemeris via JPL Horizons API...")
    
    try:
        # Convert dates to include time for JPL API
        start_str = f"{start_date} 00:00:00"
        stop_str = f"{stop_date} 23:59:59"
        
        # Try different step sizes
        steps_to_try = ['1h', '30m', '15m']  # JPL API works better with larger steps
        
        for step_try in steps_to_try:
            print(f"Attempting ephemeris fetch with step='{step_try}'")
            
            sun_df = get_jpl_ephemeris('10', start_str, stop_str, step_try)
            moon_df = get_jpl_ephemeris('301', start_str, stop_str, step_try)
            
            if sun_df is None or moon_df is None:
                print(f"Failed to fetch data with step {step_try}, trying next...")
                continue
            
            # Process the data
            date_format = "%Y-%b-%d %H:%M:%S.%f"
            
            for df in [sun_df, moon_df]:
                # Convert numeric columns
                for col in ['X', 'Y', 'Z', 'RG']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Clean and parse dates
                clean_date_series = df['CalendarDate'].astype(str).str.replace('A.D. ', '').str.strip()
                df['DateTime'] = pd.to_datetime(clean_date_series, format=date_format, errors='coerce')
                
                # Drop rows with invalid dates
                df.dropna(subset=['DateTime'], inplace=True)
            
            # Ensure we have data for the full range
            start_dt = pd.to_datetime(start_date)
            stop_dt = pd.to_datetime(stop_date)
            
            sun_df = sun_df[(sun_df['DateTime'] >= start_dt) & (sun_df['DateTime'] <= stop_dt)].copy()
            moon_df = moon_df[(moon_df['DateTime'] >= start_dt) & (moon_df['DateTime'] <= stop_dt)].copy()
            
            if sun_df.empty or moon_df.empty:
                print(f"No data in range with step {step_try}, trying next...")
                continue
            
            # Calculate gravity fields
            sun_df['SunGravity'] = 1 / (sun_df['RG']**2)
            moon_df['MoonGravity'] = 1 / (moon_df['RG']**2)
            
            # Prepare the dataframes
            sun_data = sun_df[['DateTime', 'X', 'Y', 'Z', 'RG', 'SunGravity']].rename(
                columns={'X': 'Sun_X_km', 'Y': 'Sun_Y_km', 'Z': 'Sun_Z_km', 'RG': 'Sun_Dist_km'}
            )
            moon_data = moon_df[['DateTime', 'X', 'Y', 'Z', 'RG', 'MoonGravity']].rename(
                columns={'X': 'Moon_X_km', 'Y': 'Moon_Y_km', 'Z': 'Moon_Z_km', 'RG': 'Moon_Dist_km'}
            )
            
            # Merge and ensure we have data for the entire range
            ephemeris_df = pd.merge(sun_data, moon_data, on='DateTime', how='outer')
            ephemeris_df = ephemeris_df.sort_values('DateTime').reset_index(drop=True)
            
            # Resample to 30-second intervals
            print(f"Resampling ephemeris from {step_try} -> 30s")
            ephemeris_df = _resample_ephemeris_to_30s(ephemeris_df, start_date, stop_date)
            
            # Define features for scaling
            ephemeris_features = [
                'Sun_X_km', 'Sun_Y_km', 'Sun_Z_km', 'Sun_Dist_km', 'SunGravity',
                'Moon_X_km', 'Moon_Y_km', 'Moon_Z_km', 'Moon_Dist_km', 'MoonGravity'
            ]
            
            # Scale the features
            scaler = StandardScaler()
            scaled_ephemeris = pd.DataFrame(
                scaler.fit_transform(ephemeris_df[ephemeris_features]),
                columns=ephemeris_features
            )
            
            # Add shadow classification
            shadow_df = calculate_satellite_shadow(ephemeris_df['DateTime'])
            
            # Combine everything
            ephemeris_df_scaled = pd.concat([
                ephemeris_df[['DateTime']].reset_index(drop=True),
                shadow_df.reset_index(drop=True),
                scaled_ephemeris.reset_index(drop=True)
            ], axis=1)
            
            print(f"Ephemeris data fetched: {len(ephemeris_df_scaled)} rows (step used: {step_try})")
            return ephemeris_df_scaled
        
        print("All ephemeris fetch attempts failed. Ephemeris features will be filled with placeholders.")
        return None
        
    except Exception as e:
        print(f"JPL API ephemeris fetch failed: {repr(e)}")
        return None

# ---------------------
# GNSS parsing & pipeline (unchanged logic)
# ---------------------
def parse_broadcast_rinex(rinex_file):
    print("Parsing broadcast RINEX file...")
    nav = gr.load(rinex_file)
    df_nav = nav.to_dataframe().reset_index()  # type: ignore
    keep_cols = [c for c in ['time', 'sv', 'SVclockBias', 'SVclockDrift', 'SVclockDriftRate'] if c in df_nav.columns]
    df_nav = df_nav[keep_cols]
    # map safely
    col_map = {}
    if 'time' in df_nav.columns:
        col_map['time'] = 'datetime'
    if 'sv' in df_nav.columns:
        col_map['sv'] = 'PRN'
    if 'SVclockBias' in df_nav.columns:
        col_map['SVclockBias'] = 'bias_af0'
    if 'SVclockDrift' in df_nav.columns:
        col_map['SVclockDrift'] = 'bias_af1'
    if 'SVclockDriftRate' in df_nav.columns:
        col_map['SVclockDriftRate'] = 'bias_af2'
    df_nav = df_nav.rename(columns=col_map)
    df_nav['datetime'] = pd.to_datetime(df_nav['datetime'])
    print(f"Broadcast data: {len(df_nav)} rows")
    return df_nav

def parse_precise_clk(clk_file):
    print("Parsing precise clock file...")
    rows = []
    with open(clk_file, 'r') as f:
        for line in f:
            if line.startswith('AS'):
                parts = line.split()
                try:
                    prn = parts[1]
                    year, month, day, hour, minute = map(int, parts[2:7])
                    sec = float(parts[7])
                    bias = float(parts[8])
                    drift = float(parts[9])
                except Exception as e:
                    print(f"Warning: failed to parse AS line -> {e}")
                    continue
                dt = f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{sec:06.3f}"
                rows.append([dt, prn, bias, drift])

    df_clk = pd.DataFrame(rows, columns=['datetime', 'PRN', 'precise_bias', 'drift'])
    df_clk['datetime'] = pd.to_datetime(df_clk['datetime'])
    print(f"Precise clock data: {len(df_clk)} rows")
    return df_clk

def preprocess_gnss_data_pipeline(broadcast_df, precise_df, start_date, stop_date):
    print("Starting GNSS data preprocessing pipeline...")

    print("Step 1: Merging broadcast and precise data...")
    df = pd.merge(broadcast_df, precise_df, on=['datetime', 'PRN'], how='inner')

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(by=['PRN', 'datetime']).set_index('datetime')

    print(f"Merged data: {len(df)} rows")

    print("Step 2: Resampling to 30-second intervals...")
    df_resampled = pd.DataFrame()
    for prn in df['PRN'].unique():
        prn_data = df[df['PRN'] == prn].copy()
        prn_resampled = prn_data.resample('30s').asfreq()
        for col in ['bias_af0', 'bias_af1', 'bias_af2', 'precise_bias', 'drift']:
            if col in prn_resampled.columns:
                prn_resampled[col] = prn_resampled[col].interpolate(method='time')
        prn_resampled['PRN'] = prn
        df_resampled = pd.concat([df_resampled, prn_resampled])

    df = df_resampled.reset_index()

    print("Step 3: Removing outliers...")
    for col in ['bias_af0', 'bias_af1', 'bias_af2', 'precise_bias', 'drift']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower_bound, upper_bound)

    print("Step 4: Feature engineering...")
    df['clock_bias_correction'] = df['bias_af0'] - df['precise_bias']

    print("Step 5: Adding external factors (Type 2 inputs)...")
    ephemeris_df = fetch_and_process_ephemeris_data(start_date, stop_date, '30s')

    if ephemeris_df is not None:
        df = df.sort_values('datetime')
        ephemeris_df = ephemeris_df.sort_values('DateTime')
        df = pd.merge_asof(df, ephemeris_df, left_on='datetime', right_on='DateTime', direction='nearest')
        df = df.drop('DateTime', axis=1, errors='ignore')
    else:
        ephemeris_columns = [
            'InShadow', 'OutShadow', 'UndeterminedShadow',
            'Sun_X_km', 'Sun_Y_km', 'Sun_Z_km', 'Sun_Dist_km', 'SunGravity',
            'Moon_X_km', 'Moon_Y_km', 'Moon_Z_km', 'Moon_Dist_km', 'MoonGravity'
        ]
        for col in ephemeris_columns:
            df[col] = 0.0

    print("Step 6: Feature scaling...")
    features_to_scale = ['bias_af0', 'bias_af1', 'bias_af2', 'precise_bias', 'drift']
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(
        scaler.fit_transform(df[features_to_scale]),
        columns=[f"{col}_scaled" for col in features_to_scale]
    )

    processed_df = pd.concat([df.reset_index(drop=True), scaled_features], axis=1)

    print("Step 7: Creating PRN index...")
    prn_mapping = {prn: idx for idx, prn in enumerate(processed_df['PRN'].unique())}
    processed_df['prn_index'] = processed_df['PRN'].map(prn_mapping)

    final_columns = [
        'datetime', 'PRN',
        'bias_af0_scaled', 'bias_af1_scaled', 'bias_af2_scaled',
        'precise_bias_scaled', 'drift_scaled', 'clock_bias_correction',
        'Sun_X_km', 'Sun_Y_km', 'Sun_Z_km', 'Sun_Dist_km', 'SunGravity',
        'Moon_X_km', 'Moon_Y_km', 'Moon_Z_km', 'Moon_Dist_km', 'MoonGravity',
        'InShadow', 'prn_index'
    ]

    available_columns = [col for col in final_columns if col in processed_df.columns]
    processed_df_final = processed_df[available_columns]

    column_mapping = {
        'bias_af0_scaled': 'broadcast_clock_bias_scaled',
        'bias_af1_scaled': 'broadcast_eph_bias1_scaled',
        'bias_af2_scaled': 'broadcast_eph_bias2_scaled',
        'precise_bias_scaled': 'precise_clock_bias_scaled',
        'drift_scaled': 'precise_clock_drift_scaled',
        'InShadow': 'InShadow_no'
    }

    processed_df_final = processed_df_final.rename(columns=column_mapping)
    processed_df_final = processed_df_final.sort_values(['PRN', 'datetime'])
    processed_df_final = processed_df_final.drop_duplicates(subset=['datetime', 'PRN'])
    processed_df_final = processed_df_final.ffill()

    print(f"Final processed data: {len(processed_df_final)} rows at 30-second intervals")
    return processed_df_final

def main_pipeline(broadcast_file, clk_file, output_file, start_date='2024-01-01', stop_date='2024-01-02'):
    print("=== GNSS Clock Bias Correction Pipeline (Research Paper Implementation) ===")
    print(f"Start date: {start_date}")
    print(f"End date: {stop_date}")
    print(f"Broadcast file: {broadcast_file}")
    print(f"CLK file: {clk_file}")
    print(f"Output file: {output_file}")
    print(f"Time interval: 30 seconds (as per research paper)")
    print("=" * 70)

    broadcast_df = parse_broadcast_rinex(broadcast_file)
    precise_df = parse_precise_clk(clk_file)

    final_df = preprocess_gnss_data_pipeline(broadcast_df, precise_df, start_date, stop_date)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_file, index=False)

    print(f"\n✅ Pipeline completed successfully!")
    print(f"✅ Output saved to: {output_file}")
    print(f"✅ Total rows: {len(final_df)}")
    print(f"✅ PRNs processed: {final_df['PRN'].unique()}")
    print(f"✅ Time range: {final_df['datetime'].min()} to {final_df['datetime'].max()}")
    print(f"✅ Time interval: 30 seconds")

    print("\nSample of output data:")
    print(final_df.head(10))

    return final_df

if __name__ == '__main__':
    broadcast_file = Path('broadcast_data/brdc0080.24n')
    clk_file = Path(r'precise_data\COD0OPSRAP_20240080000_01D_30S_CLK.CLK')
    output_file = Path('data/clock_bias_correction_day_8_jan_2024.csv')

    start_date = '2024-01-08'
    stop_date = '2024-01-09'

    result = main_pipeline(broadcast_file, clk_file, output_file, start_date, stop_date)