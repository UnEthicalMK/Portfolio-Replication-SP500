import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist  # FIX: Correct module import

plt.style.use('seaborn-v0_8-whitegrid')

TARGET = '^SP500TR'

# =========================================================
# SINGLE SOURCE OF TRUTH FOR ALL SPLITS
# =========================================================

TRAIN_END = '2021-12-31'
VAL_START = '2022-01-01'
VAL_END = '2022-12-31'
TEST_START = '2023-01-01'


def ensure_directories():
    os.makedirs('data', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)


def load_data():
    print("Loading log returns...")
    return pd.read_csv(
        'data/log_returns.csv',
        index_col=0,
        parse_dates=True
    )


def partition_data(df):

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    X_train = X.loc[:TRAIN_END]
    y_train = y.loc[:TRAIN_END]

    X_val = X.loc[VAL_START:VAL_END]
    y_val = y.loc[VAL_START:VAL_END]

    X_test = X.loc[TEST_START:]
    y_test = y.loc[TEST_START:]

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
        "plots/03_correlation_matrix.png",
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
    X_train_scaled.to_csv('data/X_train_scaled.csv')
    X_val_scaled.to_csv('data/X_val_scaled.csv')
    X_test_scaled.to_csv('data/X_test_scaled.csv')

    # =====================================================
    # Save unscaled datasets
    # =====================================================
    X_train.to_csv('data/X_train_unscaled.csv')
    X_val.to_csv('data/X_val_unscaled.csv')
    X_test.to_csv('data/X_test_unscaled.csv')

    y_train.to_csv('data/y_train.csv')
    y_val.to_csv('data/y_val.csv')
    y_test.to_csv('data/y_test.csv')

    joblib.dump(scaler, 'models/scaler.pkl')

    print("Milestone 2 Complete. Datasets partitioned, scaled, and saved.\n")