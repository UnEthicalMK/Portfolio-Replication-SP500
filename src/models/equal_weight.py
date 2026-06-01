import pandas as pd
import numpy as np

def train_equal_weight(X_train: pd.DataFrame, y_train: pd.Series, 
                       X_val: pd.DataFrame, y_val: pd.Series, top_k: int) -> pd.Series:
    """
    Trains an independent Equal-Weight Baseline portfolio.
    Strictly uses training data to rank assets by correlation to the target,
    then allocates capital equally among the top_k assets.
    """
    print(f"Training Model A: Equal-Weight Baseline (Constrained to top {top_k} assets)...")
    
    # Calculate training correlations to rank assets (Prevents validation look-ahead bias)
    correlations = X_train.corrwith(y_train).sort_values(ascending=False)
    
    # Force the model to only use the exact number of assets Lasso chose
    top_assets = correlations.head(top_k).index
    
    # Initialize a zero-weight vector for all available assets
    w = pd.Series(0.0, index=X_train.columns)
    
    # Allocate capital equally across the selected top_k assets
    w[top_assets] = 1.0 / top_k
    
    return w