import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class TimeSeriesDataset(Dataset):
    """
    Custom Dataset for time-series sequencing.
    Separates PRN index for embedding and handles continuous features.
    """
    def __init__(self, features_df, targets, input_seq_len, output_seq_len, prn_col_name='prn_index'):
        # Ensure the PRN column exists before trying to access it
        if prn_col_name not in features_df.columns:
            raise ValueError(f"'{prn_col_name}' not found in the provided features DataFrame.")
            
        prn_idx = list(features_df.columns).index(prn_col_name)
        
        # Store PRN indices as a separate tensor
        self.prn_indices = torch.LongTensor(features_df.iloc[:, prn_idx].values)
        
        # Store remaining continuous features as a tensor
        self.features = torch.FloatTensor(features_df.drop(columns=[prn_col_name]).values)
        
        self.targets = torch.FloatTensor(targets)
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

    def __len__(self):
        # The total length is based on the number of possible sequences we can create
        return len(self.features) - self.input_seq_len - self.output_seq_len + 1

    def __getitem__(self, idx):
        # Get the sequence of continuous features
        input_features_seq = self.features[idx : idx + self.input_seq_len]
        
        # Get the single PRN index for this sequence. We assume it's constant.
        # It must be a LongTensor for the embedding layer.
        prn_val = self.prn_indices[idx].unsqueeze(0) # Shape: [1]
        
        # Get the target sequence
        target_seq = self.targets[idx + self.input_seq_len : idx + self.input_seq_len + self.output_seq_len]
        
        return input_features_seq, prn_val, target_seq

def create_dataloaders(data_path, features_list, target_col, test_size, val_size, input_seq_len, output_seq_len, batch_size):
    """Reads your data, preprocesses it, and creates PyTorch DataLoaders."""
    df = pd.read_csv(data_path)

    # --- Preprocessing ---
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(by='datetime').reset_index(drop=True)

    # The user's data sample already has 'prn_index', so we'll assume it exists.
    # If not, this is where you would create it.
    if 'prn_index' not in df.columns:
        print("Creating 'prn_index' from 'PRN' column.")
        le = LabelEncoder()
        df['prn_index'] = le.fit_transform(df['PRN'])
    
    # --- Splitting Data ---
    n = len(df)
    test_split = int(n * (1 - test_size))
    val_split = int(test_split * (1 - val_size / (1 - test_size)))

    train_df = df[:val_split]
    val_df = df[val_split:test_split]
    test_df = df[test_split:]

    # --- Creating Datasets ---
    # We now pass the DataFrame directly to the dataset class
    # The list of features passed should now include 'prn_index'
    full_feature_list = features_list + ['prn_index']
    
    train_dataset = TimeSeriesDataset(
        train_df[full_feature_list],
        train_df[target_col].values,
        input_seq_len,
        output_seq_len
    )
    val_dataset = TimeSeriesDataset(
        val_df[full_feature_list],
        val_df[target_col].values,
        input_seq_len,
        output_seq_len
    )
    test_dataset = TimeSeriesDataset(
        test_df[full_feature_list],
        test_df[target_col].values,
        input_seq_len,
        output_seq_len
    )

    # --- Creating DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    print(f"Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader