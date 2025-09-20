import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import argparse
import os
import sys
from tqdm import tqdm  # Import tqdm for progress bars

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.encoder import GRUEncoder
from model.decoder import GRUDecoder
from model.loung_attention import LuongAttention
from model.wrapper import Seq2Seq


class GNSSPredictor:
    """
    A comprehensive GNSS error prediction system using Seq2Seq with Attention.
    Supports prediction of clock error, ephemeris error, or remaining error.
    """

    def __init__(self, config):
        """
        Initialize the GNSS predictor.
        """
        self.config = config
        # Force GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_features = None
        self.model = None
        self.teacher_forcing_ratio = 1.0  # Start with full teacher forcing

        # Scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def initialize_model(self):
        """Initialize the model after we know the input features."""
        if self.input_features is None:
            raise ValueError("Input features not set. Load data first.")

        self.encoder = GRUEncoder(
            input_size=self.input_features,
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        ).to(self.device)

        self.decoder = GRUDecoder(
            output_size=1,
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        ).to(self.device)

        self.model = Seq2Seq(self.encoder, self.decoder, self.device).to(self.device)

        # Optimizer and loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config['learning_rate'], 
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        self.criterion = nn.SmoothL1Loss()

    def prepare_data(self, data_path, sequence_length=672, forecast_horizon=96):
        """Prepare GNSS dataset for training and validation."""
        df = pd.read_csv(data_path)

        target_column = self._get_target_column()
        feature_columns = self._get_feature_columns(df.columns)

        features = df[feature_columns].values
        target = df[target_column].values.reshape(-1, 1) # type: ignore

        # Clip outliers in target (1st and 99th percentiles)
        lower_bound = np.percentile(target, 1)
        upper_bound = np.percentile(target, 99)
        target = np.clip(target, lower_bound, upper_bound)

        # Set input features for model
        self.input_features = features.shape[1]
        self.initialize_model()

        # Scale
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.target_scaler.fit_transform(target)

        # Sequences
        X, y = self._create_sequences(features_scaled, target_scaled,
                                      sequence_length, forecast_horizon)

        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        return X_train, X_val, y_train, y_val

    def _get_target_column(self):
        target_mapping = {
            'clock': 'clock_error_approx_m',
            'ephemeris': 'ephemeris_error_approx_m',
            'remaining': 'remaining_error_m'
        }
        return target_mapping[self.config['target_type']]

    def _get_feature_columns(self, all_columns):
        satellite_columns = [col for col in all_columns if col.startswith('Satelite_Code_')]
        measurement_columns = ['GPS_Time(s)', 'Code_L1', 'Phase_L1', 'Cnr_L1', 'Pr_Error']
        return satellite_columns + measurement_columns

    def _create_sequences(self, features, target, seq_length, horizon):
        X, y = [], []
        for i in range(len(features) - seq_length - horizon + 1):
            X.append(features[i:i+seq_length])
            y.append(target[i+seq_length:i+seq_length+horizon])
        return np.array(X), np.array(y)

    def update_teacher_forcing_ratio(self, epoch, total_epochs):
        """Gradually reduce teacher forcing ratio during training"""
        # Linear decay from 1.0 to 0.0
        self.teacher_forcing_ratio = max(0.0, 1.0 - (epoch / total_epochs))

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model with loss + accuracy logging and scheduled sampling."""
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        history = {'train_loss': [], 'val_loss': [], 'val_r2': [], 'val_mae': []}

        for epoch in range(epochs):
            # Update teacher forcing ratio
            self.update_teacher_forcing_ratio(epoch, epochs)
            
            self.model.train() # type: ignore
            epoch_loss = 0
            
            # Create progress bar for batches
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch')
            
            for batch_X, batch_y in pbar:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward with current teacher forcing ratio
                predictions = self.model(
                    batch_X, 
                    batch_y, 
                    teacher_forcing_ratio=self.teacher_forcing_ratio
                ) # type: ignore

                # Loss
                loss = self.criterion(predictions, batch_y)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # type: ignore
                self.optimizer.step()

                epoch_loss += loss.item()
                
                # Update progress bar with current loss
                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'teacher_forcing': f'{self.teacher_forcing_ratio:.3f}'
                })

            # Validation
            val_loss, val_r2, val_mae = self.validate(X_val, y_val)

            history['train_loss'].append(epoch_loss / len(train_loader))
            history['val_loss'].append(val_loss)
            history['val_r2'].append(val_r2)
            history['val_mae'].append(val_mae)

            print(
                f"\nEpoch {epoch+1}/{epochs} | "
                f"Train Loss: {epoch_loss/len(train_loader):.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val R²: {val_r2:.4f} | "
                f"Val MAE: {val_mae:.6f} | "
                f"Teacher Forcing: {self.teacher_forcing_ratio:.3f}"
            )

        return history

    def validate(self, X_val, y_val, batch_size=32):
        """Calculate validation metrics."""
        self.model.eval() # type: ignore
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        total_loss = 0
        preds, trues = [], []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                predictions = self.model(batch_X, batch_y, teacher_forcing_ratio=0.0) # type: ignore
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()

                preds.append(predictions.cpu().numpy())
                trues.append(batch_y.cpu().numpy())

        preds = np.concatenate(preds, axis=0).reshape(-1)
        trues = np.concatenate(trues, axis=0).reshape(-1)

        r2 = r2_score(trues, preds)
        mae = mean_absolute_error(trues, preds)

        return total_loss / len(val_loader), r2, mae

    def predict(self, X):
        """Predict new sequences."""
        self.model.eval() # type: ignore
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor, None, teacher_forcing_ratio=0.0) # type: ignore

        predictions = predictions.cpu().numpy()
        predictions = self.target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).reshape(predictions.shape)

        return predictions

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(), # type: ignore
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'config': self.config,
            'input_features': self.input_features
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.input_features = checkpoint['input_features']
        self.initialize_model()
        self.model.load_state_dict(checkpoint['model_state_dict']) # type: ignore
        self.feature_scaler = checkpoint['feature_scaler']
        self.target_scaler = checkpoint['target_scaler']


def main():
    parser = argparse.ArgumentParser(description='GNSS Error Prediction using Seq2Seq with Attention')
    parser.add_argument('--target', type=str, required=True,
                        choices=['clock', 'ephemeris', 'remaining'],
                        help='Type of error to predict: clock, ephemeris, or remaining')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the GNSS data CSV file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='Hidden size for GRU layers')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GRU layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--save_path', type=str, default='gnss_model.pth',
                        help='Path to save the trained model')

    args = parser.parse_args()

    config = {
        'target_type': args.target,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    predictor = GNSSPredictor(config)

    if not os.path.exists(args.data_path):
        print(f"Error: Data file {args.data_path} not found.")
        return

    print("Preparing data...")
    X_train, X_val, y_train, y_val = predictor.prepare_data(args.data_path)

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    print("Training model...")
    history = predictor.train(X_train, y_train, X_val, y_val,
                              epochs=args.epochs, batch_size=args.batch_size)

    predictor.save_model(args.save_path)
    print(f"Model saved to {args.save_path}")

    # Plot history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.plot(history['val_r2'], label='Validation R²')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.title(f'Training History - {args.target} Error Prediction')
    plt.savefig('training_history.png')
    plt.show()


if __name__ == "__main__":
    main()