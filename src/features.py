import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist  # FIX: Correct module import
from src.config import (
    DATA_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    TARGET,
    TEST_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
)

plt.style.use('seaborn-v0_8-whitegrid')

def ensure_directories():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    print("Loading log returns...")
    return pd.read_csv(
        DATA_DIR / 'log_returns.csv',
        index_col=0,
        parse_dates=True
    )


def partition_data(df):

    if TARGET not in df.columns:
        raise ValueError(f"Return data is missing the benchmark column: {TARGET}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Return data must use a DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Return data index must be sorted in ascending order.")

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    X_train = X.loc[:TRAIN_END]
    y_train = y.loc[:TRAIN_END]

    X_val = X.loc[VAL_START:VAL_END]
    y_val = y.loc[VAL_START:VAL_END]

    X_test = X.loc[TEST_START:]
    y_test = y.loc[TEST_START:]

    for name, split in {
        "train": X_train,
        "validation": X_val,
        "test": X_test,
    }.items():
        if split.empty:
            raise ValueError(f"{name.capitalize()} split is empty.")

    print(f"  Train: {X_train.shape[0]} rows")
    print(f"  Val:   {X_val.shape[0]} rows")
    print(f"  Test:  {X_test.shape[0]} rows")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


def plot_correlation_matrix(X_train):
    print("Generating Plot 3: Hierarchical Correlation Matrix...")
    corr = X_train.corr()

    # FIX: Use pdist on transposed features to natively calculate condensed distances.
    # Completely avoids AttributeError and floating-point ValueError traps.
    dist_array = pdist(X_train.T, metric='correlation')
    
    linkage = hc.linkage(
        dist_array,
        method='average'
    )

    order = hc.leaves_list(linkage)

    sorted_cols = corr.columns[order]
    sorted_corr = X_train[sorted_cols].corr()

    mask = np.triu(
        np.ones_like(sorted_corr, dtype=bool)
    )

    plt.figure(figsize=(14, 12))

    sns.heatmap(
        sorted_corr,
        mask=mask,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        annot=False
    )

    plt.title(
        "Training Correlation Matrix (2014–2021)",
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "03_correlation_matrix.png",
        dpi=150
    )
    plt.close()


def scale_features(
    X_train,
    X_val,
    X_test
):
    print("Fitting Scalers and transforming data...")
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
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

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    )


def run_feature_engineering():
    print("\n--- Starting Milestone 2: Feature Engineering & Scaling ---")
    ensure_directories()

    df = load_data()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = partition_data(df)

    plot_correlation_matrix(X_train)

    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    ) = scale_features(
        X_train,
        X_val,
        X_test
    )

    # =====================================================
    # Save scaled datasets
    # =====================================================
    X_train_scaled.to_csv(DATA_DIR / 'X_train_scaled.csv')
    X_val_scaled.to_csv(DATA_DIR / 'X_val_scaled.csv')
    X_test_scaled.to_csv(DATA_DIR / 'X_test_scaled.csv')

    # =====================================================
    # Save unscaled datasets
    # =====================================================
    X_train.to_csv(DATA_DIR / 'X_train_unscaled.csv')
    X_val.to_csv(DATA_DIR / 'X_val_unscaled.csv')
    X_test.to_csv(DATA_DIR / 'X_test_unscaled.csv')

    y_train.to_csv(DATA_DIR / 'y_train.csv')
    y_val.to_csv(DATA_DIR / 'y_val.csv')
    y_test.to_csv(DATA_DIR / 'y_test.csv')

    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')

    print("Milestone 2 Complete. Datasets partitioned, scaled, and saved.\n")