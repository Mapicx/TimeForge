import torch
import argparse
from tqdm import tqdm
import numpy as np

# Local imports
import config
from model.transformer import TransformerEncoderModel
from utils.data_loader import create_dataloaders
from utils.metrics import calculate_metrics

def evaluate_model(model_path):
    """
    Loads a trained model and evaluates its performance on the test set.
    """
    print(f"--- 🧪 Starting Evaluation ---")
    print(f"Loading model from: {model_path}")

    device = torch.device(config.DEVICE)
    features_to_use = config.FEATURES_TYPE_2

    # 1. Load the test data using the same settings as in training
    print("Loading test dataset...")
    _, _, test_loader = create_dataloaders(
        data_path=config.DATA_PATH,
        features_list=features_to_use, 
        target_col=config.TARGET,
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        input_seq_len=config.INPUT_SEQ_LEN,
        output_seq_len=config.OUTPUT_SEQ_LEN,
        batch_size=config.BATCH_SIZE # Can be larger for evaluation if GPU memory allows
    )

    # 2. Re-create the model architecture with the SAME hyperparameters used for training
    num_continuous_features = len(features_to_use)
    num_satellites = 32
    embedding_dim = 10
    
    model = TransformerEncoderModel(
        num_continuous_features=num_continuous_features,
        num_satellites=num_satellites,
        embedding_dim=embedding_dim,
        d_model=32,      # From your command
        n_heads=2,       # From your command
        n_layers=1,      # From your command
        d_ff=128,        # From your command
        dropout=config.DROPOUT,
        output_seq_len=config.OUTPUT_SEQ_LEN,
        input_seq_len=config.INPUT_SEQ_LEN
    )

    # 3. Load the saved weights into the model instance
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"❌ ERROR: Model file not found at '{model_path}'.")
        return
    except Exception as e:
        print(f"❌ ERROR: Failed to load model weights. Ensure architecture matches. Details: {e}")
        return
        
    model.to(device)
    model.eval() # Set the model to evaluation mode

    # 4. Run predictions on the test set
    print("Running predictions on the test set...")
    all_preds = []
    all_targets = []

    with torch.no_grad(): # Disable gradient calculations for efficiency
        progress_bar = tqdm(test_loader, desc="Evaluating")
        for continuous_inputs, prn_inputs, targets in progress_bar:
            continuous_inputs = continuous_inputs.to(device)
            prn_inputs = prn_inputs.to(device)
            
            outputs = model(continuous_inputs, prn_inputs)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # 5. Calculate and print the final metrics
    print("\n--- ✅ Evaluation Complete ---")
    final_metrics = calculate_metrics(all_targets, all_preds)
    
    print("\n--- Final Performance on Test Set ---")
    print(f"  - 📏 RMSE (Root Mean Squared Error): {final_metrics['rmse']:.6f}")
    print(f"  - 📐 MAE (Mean Absolute Error):     {final_metrics['mae']:.6f}")
    # --- New line to print MAPE ---
    print(f"  - 🎯 MAPE (Mean Absolute Percentage Error): {final_metrics['mape']:.6f} %")
    print(f"  - 📈 R² Score:                       {final_metrics['r2']:.6f}")
    print("-------------------------------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained Transformer model.")
    parser.add_argument(
        '--model_path', 
        type=str, 
        default="gnss_transformer_model_with_embeddings.pth", 
        help='Path to the saved model file (.pth)'
    )
    args = parser.parse_args()
    evaluate_model(args.model_path)