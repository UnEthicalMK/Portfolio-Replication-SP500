import os
import pandas as pd
import matplotlib.pyplot as plt

from src.models.equal_weight import train_equal_weight
from src.models.simplex_optimizer import (
    train_ridge_tracker,
    plot_ridge_sweep
)
from src.models.sparse_autoencoder import train_sae

plt.style.use('seaborn-v0_8-whitegrid')

TARGET = '^GSPC'


def ensure_directories():
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)


def load_preprocessed_data():
    print("[INFO] Loading preprocessed datasets...")

    X_train_scaled = pd.read_csv(
        "data/X_train_scaled.csv",
        index_col=0,
        parse_dates=True
    )
    X_val_scaled = pd.read_csv(
        "data/X_val_scaled.csv",
        index_col=0,
        parse_dates=True
    )

    X_train_unscaled = pd.read_csv(
        "data/X_train_unscaled.csv",
        index_col=0,
        parse_dates=True
    )
    X_val_unscaled = pd.read_csv(
        "data/X_val_unscaled.csv",
        index_col=0,
        parse_dates=True
    )

    y_train = pd.read_csv("data/y_train.csv", index_col=0).squeeze()
    y_val = pd.read_csv("data/y_val.csv", index_col=0).squeeze()

    print(f"[INFO] Training observations : {len(X_train_scaled)}")
    print(f"[INFO] Validation observations: {len(X_val_scaled)}")

    return (
        X_train_scaled,
        X_val_scaled,
        X_train_unscaled,
        X_val_unscaled,
        y_train,
        y_val
    )


def plot_instrument_weights(weights_df: pd.DataFrame):
    print("[INFO] Generating portfolio allocation chart...")

    fig, ax = plt.subplots(figsize=(14, 6))

    weights_df.plot(
        kind='bar',
        ax=ax,
        width=0.8,
        color=['steelblue', 'darkorange', 'seagreen']
    )

    ax.set_title(
        "Frozen Portfolio Allocation Vectors (Plot 5)",
        fontsize=14,
        fontweight='bold'
    )
    ax.set_ylabel("Portfolio Weight")
    ax.set_xlabel("Assets")
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title='Replication Models')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(
        "plots/05_instrument_weights.png",
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()


def run_models_pipeline():
    print("\n" + "=" * 60)
    print("MILESTONE 3 | PORTFOLIO CONSTRUCTION")
    print("=" * 60)

    ensure_directories()

    (
        X_train_scaled,
        X_val_scaled,
        X_train_unscaled,
        X_val_unscaled,
        y_train,
        y_val
    ) = load_preprocessed_data()

    # Model B: Ridge Tracker
    (
        ridge_weights,
        alphas,
        tracking_errors,
        asset_counts,
        best_alpha
    ) = train_ridge_tracker(
        X_train_unscaled,
        y_train,
        X_val_unscaled,
        y_val
    )

    plot_ridge_sweep(
        alphas,
        tracking_errors,
        asset_counts,
        best_alpha
    )

    active_assets_count = (ridge_weights > 0).sum()

    print(
        f"\n[INFO] Optimal sparse portfolio size: "
        f"{active_assets_count} assets."
    )
    print(
        "[INFO] Applying identical asset-count constraints "
        "across all models."
    )

    # Model A: Equal Weight
    eq_weights = train_equal_weight(
        X_train_unscaled,
        y_train,
        X_val_unscaled,
        y_val,
        top_k=active_assets_count
    )

    # Model C: Sparse Autoencoder
    sae_weights = train_sae(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        top_k=active_assets_count
    )

    print("[INFO] Freezing portfolio weights for out-of-sample evaluation...")

    frozen_weights = pd.DataFrame({
        'Equal_Weight': eq_weights,
        'Ridge_Tracker': ridge_weights,
        'SAE': sae_weights
    }).fillna(0.0)

    frozen_weights = frozen_weights.loc[
        (frozen_weights != 0).any(axis=1)
    ]

    frozen_weights.to_csv(
        "models/frozen_weights_2022.csv"
    )

    plot_instrument_weights(frozen_weights)

    print("[SUCCESS] Portfolio construction completed.\n")