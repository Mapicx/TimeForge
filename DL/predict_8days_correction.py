import torch
import torch.nn as nn
import math
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import argparse

# --- Model Classes (Must be identical to your training setup) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoderModel(nn.Module):
    def __init__(self, num_continuous_features, num_satellites, embedding_dim, d_model, n_heads, n_layers, d_ff, dropout, output_seq_len, input_seq_len):
        super(TransformerEncoderModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=num_satellites, embedding_dim=embedding_dim)
        self.encoder = nn.Linear(num_continuous_features + embedding_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.Linear(d_model * input_seq_len, output_seq_len)

    def forward(self, continuous_src, prn_src):
        prn_embedding = self.embedding(prn_src).repeat(1, continuous_src.size(1), 1)
        src = torch.cat([continuous_src, prn_embedding], dim=2)
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.reshape(output.size(0), -1)
        output = self.decoder(output)
        return output

# --- Custom Dataset for Prediction (Now with Evaluation Capability) ---
class PredictionDataset(Dataset):
    # Updated the default target column name
    def __init__(self, features_df, input_seq_len, output_seq_len=None, evaluate_mode=False, prn_col_name='prn_index', target_col_name='clock_bias_correction'):
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.evaluate_mode = evaluate_mode

        prn_idx_pos = features_df.columns.get_loc(prn_col_name)
        self.prn_indices = torch.LongTensor(features_df.iloc[:, prn_idx_pos].values)
        
        self.features = torch.FloatTensor(features_df.drop(columns=[prn_col_name]).values)

        if self.evaluate_mode:
            if target_col_name not in features_df.columns:
                 raise ValueError(f"Target column '{target_col_name}' not found for evaluation.")
            if output_seq_len is None:
                raise ValueError("output_seq_len must be provided in evaluate_mode.")
            target_idx_pos = features_df.drop(columns=[prn_col_name]).columns.get_loc(target_col_name)
            self.targets = self.features[:, target_idx_pos]


    def __len__(self):
        if self.evaluate_mode:
            return len(self.features) - self.input_seq_len - self.output_seq_len + 1
        else:
            return len(self.features) - self.input_seq_len + 1

    def __getitem__(self, idx):
        input_features_seq = self.features[idx : idx + self.input_seq_len]
        prn_val = self.prn_indices[idx].unsqueeze(0)
        
        if self.evaluate_mode:
            target_seq = self.targets[idx + self.input_seq_len : idx + self.input_seq_len + self.output_seq_len]
            return input_features_seq, prn_val, target_seq
        else:
            return input_features_seq, prn_val

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict GNSS clock bias using a trained Transformer model.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file for prediction.')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model state dictionary (.pth file).')
    parser.add_argument('--out', type=str, required=True, help='Path to save the output predictions CSV file.')
    parser.add_argument('--evaluate', action='store_true', help='Enable evaluation against ground truth if available in the input file.')
    args = parser.parse_args()

    # --- Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    INPUT_SEQ_LEN = 1440
    OUTPUT_SEQ_LEN = 240
    
    # --- CHANGE #1: Updated the feature list to include the new target variable ---
    FEATURES_TYPE_2 = [
        'clock_bias_correction',
        'SunGravity',
        'MoonGravity',
        'InShadow_no'
    ]
    
    D_MODEL = 32
    N_HEADS = 8
    N_LAYERS = 1
    D_FF = 128
    DROPOUT = 0.1
    NUM_SATELLITES = 32
    EMBEDDING_DIM = 10

    # --- 1. Load and Preprocess Data ---
    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {args.input}")
        exit()
        
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.sort_values(by='datetime', inplace=True)
    
    missing_cols = [col for col in FEATURES_TYPE_2 if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: Your input file is missing the following required feature columns:")
        print(missing_cols)
        exit()

    le = LabelEncoder()
    df['prn_index'] = le.fit_transform(df['PRN'])
    
    full_feature_list = FEATURES_TYPE_2 + ['prn_index']
    
    # --- 2. Load Model ---
    print(f"Loading model from {args.model}...")
    model = TransformerEncoderModel(
        num_continuous_features=len(FEATURES_TYPE_2),
        num_satellites=NUM_SATELLITES,
        embedding_dim=EMBEDDING_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        output_seq_len=OUTPUT_SEQ_LEN,
        input_seq_len=INPUT_SEQ_LEN
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(args.model, map_location=DEVICE, weights_only=True))
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {args.model}")
        exit()
    except Exception as e:
        print(f"❌ Error loading model state dictionary: {e}")
        exit()
        
    model.eval()

    # --- 3. Run Prediction & Optional Evaluation ---
    print("Preparing dataset...")
    pred_dataset = PredictionDataset(
        features_df=df[full_feature_list], 
        input_seq_len=INPUT_SEQ_LEN,
        output_seq_len=OUTPUT_SEQ_LEN,
        evaluate_mode=args.evaluate,
        # --- CHANGE #2: Explicitly set the target column for evaluation ---
        target_col_name='clock_bias_correction'
    )
    pred_loader = DataLoader(pred_dataset, batch_size=32, shuffle=False)

    print("Running predictions...")
    all_predictions = []
    batch_losses = []
    loss_fn = nn.MSELoss() if args.evaluate else None

    with torch.no_grad():
        if args.evaluate:
            print("Evaluation mode enabled. Calculating error metrics...")
            for continuous_inputs, prn_inputs, targets in pred_loader:
                continuous_inputs = continuous_inputs.to(DEVICE)
                prn_inputs = prn_inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                
                outputs = model(continuous_inputs, prn_inputs)
                all_predictions.extend(outputs.cpu().numpy())

                loss = loss_fn(outputs, targets)
                batch_losses.append(loss.item())
        else:
            for continuous_inputs, prn_inputs in pred_loader:
                continuous_inputs = continuous_inputs.to(DEVICE)
                prn_inputs = prn_inputs.to(DEVICE)
                
                outputs = model(continuous_inputs, prn_inputs)
                all_predictions.extend(outputs.cpu().numpy())

    # --- 4. Print Evaluation Results if Enabled ---
    if args.evaluate:
        if batch_losses:
            avg_mse = np.mean(batch_losses)
            rmse = np.sqrt(avg_mse)
            print("\n" + "="*30)
            print("📊 Evaluation Results")
            print("="*30)
            print(f"  Mean Squared Error (MSE): {avg_mse:.6f}")
            print(f"  Root Mean Squared Error (RMSE): {rmse:.6f}")
            print("="*30)
        else:
            print("\n⚠️ Evaluation was enabled, but no data was processed. The input file might be too short for the required input/output sequence lengths.")

    # --- 5. Save Results ---
    if not all_predictions:
        print("\n⚠️ No predictions were generated. The input file might be too short.")
        exit()

    pred_df = pd.DataFrame(np.array(all_predictions))
    
    start_index = INPUT_SEQ_LEN 
    end_index = start_index + len(pred_df)
    
    pred_info = df[['datetime', 'PRN']].iloc[start_index:end_index].reset_index(drop=True)
    
    final_df = pd.concat([pred_info, pred_df], axis=1)
    final_df.columns = ['datetime', 'PRN'] + [f'pred_{i+1}' for i in range(OUTPUT_SEQ_LEN)]
    
    final_df.to_csv(args.out, index=False)
    
    print(f"\n✅ Predictions saved successfully to {args.out}")
    print(f"   - Total input rows: {len(df)}")
    print(f"   - Total predictions generated: {len(final_df)}")