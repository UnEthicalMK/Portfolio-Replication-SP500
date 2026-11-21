"""
src/features.py
---------------
Strict chronological temporal splitting and causal feature standardization.
Guarantees zero data leakage from future market regimes into past parameters.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

def chronological_split(
    log_returns: pd.DataFrame,
    target_col: str = "^SP500TR",
    train_end: str = "2020-12-31",
    val_end: str = "2021-12-31",
    test_end: str = "2024-12-31"
):
    """
    Slices the timeline into Train (2014-2020), Validation (2021), and Test (2022-2024) eras.
    
    Parameters:
        log_returns (pd.DataFrame): Stationary log returns.
        target_col (str): The column name of the benchmark index.
        train_end (str): End of training era.
        val_end (str): End of validation era.
        test_end (str): End of out-of-sample test vault.
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    if target_col not in log_returns.columns:
        raise KeyError(f"Target column '{target_col}' not found in log returns. Available: {list(log_returns.columns[:5])}...")
        
    # Isolate Features (X) and Benchmark Target (y)
    X = log_returns.drop(columns=[target_col])
    y = log_returns[target_col]
    
    # 1. Training Era (2014-01-01 to 2020-12-31): 7 years capturing steady state & COVID shock
    X_train = X.loc[:train_end]
    y_train = y.loc[:train_end]
    
    # 2. Validation Era (2021-01-01 to 2021-12-31): 1 year post-COVID reopening inflation shock
    val_start = pd.to_datetime(train_end) + pd.Timedelta(days=1)
    X_val = X.loc[val_start:val_end]
    y_val = y.loc[val_start:val_end]
    
    # 3. Test Era Vault (2022-01-01 to 2024-12-31): 3-year multi-wave stress test
    test_start = pd.to_datetime(val_end) + pd.Timedelta(days=1)
    X_test = X.loc[test_start:test_end]
    y_test = y.loc[test_start:test_end]
    
    print("\n[+] Executed Strict Chronological Temporal Split:")
    print(f"    Train Era (2014-2020):      {len(X_train):>5} days ({X_train.index[0].strftime('%Y-%m-%d')} -> {X_train.index[-1].strftime('%Y-%m-%d')})")
    print(f"    Validation Era (2021):      {len(X_val):>5} days ({X_val.index[0].strftime('%Y-%m-%d')} -> {X_val.index[-1].strftime('%Y-%m-%d')})")
    print(f"    Test Vault Era (2022-2024): {len(X_test):>5} days ({X_test.index[0].strftime('%Y-%m-%d')} -> {X_test.index[-1].strftime('%Y-%m-%d')})")
    
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("One of the temporal splits is empty! Check dataset date coverage.")
        
    return X_train, y_train, X_val, y_val, X_test, y_test

def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    """
    Standardizes features to zero mean and unit variance.
    Strict Causal Rule: Parameters are learned ONLY on X_train.
    Validation and Test data are transformed using frozen historical parameters.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        index=X_val.index,
        columns=X_val.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
