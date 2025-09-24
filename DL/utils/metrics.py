import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    """Calculates and returns RMSE, MAE, R2 Score, and MAPE."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # --- New MAPE Calculation ---
    # A small epsilon is added to the denominator to prevent division by zero.
    epsilon = 1e-8 
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Add MAPE to the returned dictionary
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}