import pandas as pd

df = pd.read_csv('data/GNSS_raw_train.csv')

df_reliable = df[df['Label'] == 0].copy()

selected_features = ["GPS_Time(s)", "Satelite_Code", "Code_L1", "Phase_L1", "Cnr_L1", "Pr_Error"]

df_selected = df_reliable[selected_features]

cols = [c for c in df_selected.columns if c != "Pr_Error"] + ["Pr_Error"]
df_selected = df_selected[cols]

df_selected.to_csv("GNSS_cleaned.csv", index=False)

print("Cleaned dataset with only reliable data saved as GNSS_cleaned.csv")
print(df_selected.head())