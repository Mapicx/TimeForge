import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


file_path = 'GNSS_cleaned.csv'
df = pd.read_csv(file_path)

print("--- Original DataFrame Columns ---")
print(df.columns.tolist())
print(df.head())

categorical_features = ['Satelite_Code']

preprocessor = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)


transformed_data = preprocessor.fit_transform(df)

encoded_feature_names = preprocessor.named_transformers_['encoder'].get_feature_names_out(categorical_features)

remaining_feature_names = [col for col in df.columns if col not in categorical_features]

new_column_names = list(encoded_feature_names) + remaining_feature_names

df_encoded = pd.DataFrame(transformed_data, columns=new_column_names)


df_encoded.to_csv(file_path, index=False)

print(f"\n\n--- Encoded DataFrame saved to {file_path} ---")
print("New Columns:")
print(df_encoded.columns.tolist())
print(df_encoded.head())