import torch
import torch.nn as nn
from functools import partial

# Ray Tune imports
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

# Local imports
import config
from model.transformer import TransformerEncoderModel
from utils.data_loader import create_dataloaders

def train_for_tune(tune_config, data_loaders):
    """Training function compatible with Ray Tune."""
    train_loader, val_loader = data_loaders['train'], data_loaders['val']
    device = torch.device(config.DEVICE)
    
    # --- Model ---
    num_features = len(config.FEATURES)
    model = TransformerEncoderModel(
        num_features=num_features,
        d_model=tune_config["d_model"],
        n_heads=tune_config["n_heads"],
        n_layers=tune_config["n_layers"],
        d_ff=tune_config["d_ff"],
        dropout=tune_config["dropout"],
        output_seq_len=config.OUTPUT_SEQ_LEN,
        input_seq_len=config.INPUT_SEQ_LEN
    ).to(device)

    # --- Optimizer and Loss ---
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=tune_config["lr"])

    # --- Training Loop ---
    for epoch in range(config.EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Report metrics to Ray Tune
        train.report({"val_loss": avg_val_loss})

def main():
    print("Loading data for Ray Tune...")
    # Load data once to be shared across trials
    train_loader, val_loader, _ = create_dataloaders(
        data_path=config.DATA_PATH,
        features_list=config.FEATURES,
        target_col=config.TARGET,
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        input_seq_len=config.INPUT_SEQ_LEN,
        output_seq_len=config.OUTPUT_SEQ_LEN,
        batch_size=config.BATCH_SIZE
    )
    data_loaders = {"train": train_loader, "val": val_loader}

    # --- Define Search Space ---
    search_space = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "d_model": tune.choice([32, 64, 128]),
        "n_heads": tune.choice([2, 4, 8]),
        "n_layers": tune.choice([1, 2, 3]),
        "d_ff": tune.choice([128, 256, 512]),
        "dropout": tune.uniform(0.1, 0.3)
    }

    # --- Define Scheduler ---
    # ASHA stops unpromising trials early
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        grace_period=1,
        reduction_factor=2,
    )

    # --- Run Tuner ---
    tuner = tune.Tuner(
        partial(train_for_tune, data_loaders=data_loaders),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=20, # Number of different hyperparameter combinations to try
            scheduler=scheduler
        ),
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("val_loss", "min")
    print("\n--- Best Trial Found ---")
    print(f"Validation loss: {best_result.metrics['val_loss']:.4f}")
    print("Best hyperparameters:")
    for key, value in best_result.config.items():
        print(f"  - {key}: {value}")


if __name__ == '__main__':
    # You might need to install these first:
    # pip install "ray[tune]"
    main()