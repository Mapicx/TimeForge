import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import argparse

# Local imports
import config
from model.transformer import TransformerEncoderModel
from utils.data_loader import create_dataloaders
from utils.metrics import calculate_metrics

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        # The dataloader now yields three items
        for continuous_inputs, prn_inputs, targets in dataloader:
            continuous_inputs = continuous_inputs.to(device)
            prn_inputs = prn_inputs.to(device)
            targets = targets.to(device)
            
            # Pass both inputs to the model
            outputs = model(continuous_inputs, prn_inputs)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_targets, all_preds)
    return avg_loss, metrics

def main(args):
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # IMPORTANT: Ensure config.FEATURES does NOT include 'prn_index'
    # [cite_start]The 'Type 2' features are recommended by the paper for better performance [cite: 573]
    features_to_use = config.FEATURES_TYPE_2 

    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=config.DATA_PATH,
        features_list=features_to_use, 
        target_col=config.TARGET,
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        input_seq_len=config.INPUT_SEQ_LEN,
        output_seq_len=config.OUTPUT_SEQ_LEN,
        batch_size=args.batch_size
    )

    # --- Model Instantiation with Embedding Layer ---
    # Number of continuous features is the length of your feature list
    num_continuous_features = len(features_to_use)
    # Number of unique satellites (e.g., GPS has 32)
    num_satellites = 32 # Adjust if using a different constellation
    # Dimension for the learned satellite embeddings
    embedding_dim = 10 # This is a new hyperparameter you can tune
    
    model = TransformerEncoderModel(
        num_continuous_features=num_continuous_features,
        num_satellites=num_satellites,
        embedding_dim=embedding_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        output_seq_len=config.OUTPUT_SEQ_LEN,
        input_seq_len=config.INPUT_SEQ_LEN
    ).to(device)

    print("\n--- Model Architecture ---\n", model, "\n--------------------------\n")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        # The dataloader now yields a tuple of three tensors
        for continuous_inputs, prn_inputs, targets in progress_bar:
            continuous_inputs = continuous_inputs.to(device)
            prn_inputs = prn_inputs.to(device)
            targets = targets.to(device)
            
            # Pass both sets of inputs to the model
            outputs = model(continuous_inputs, prn_inputs)
            
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Validation Metrics -> RMSE: {val_metrics['rmse']:.4f} | MAE: {val_metrics['mae']:.4f} | R2: {val_metrics['r2']:.4f}\n")

    print("--- Running Final Evaluation on Test Set ---")
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Metrics -> RMSE: {test_metrics['rmse']:.4f} | MAE: {test_metrics['mae']:.4f} | R2: {test_metrics['r2']:.4f}")

    model_path = "gnss_transformer_model_with_embeddings.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Model saved to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Transformer model for GNSS clock-bias prediction.")
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='Number of epochs')
    parser.add_argument('--d_model', type=int, default=config.D_MODEL, help='Model embedding dimension')
    parser.add_argument('--n_heads', type=int, default=config.N_HEADS, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=config.N_LAYERS, help='Number of encoder layers')
    parser.add_argument('--d_ff', type=int, default=config.D_FF, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=config.DROPOUT, help='Dropout rate')
    args = parser.parse_args()
    main(args)