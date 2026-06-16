import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')


def train_ridge_tracker(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    min_assets: int = 3,
    max_assets: int = 15
):
    print("Training Ridge-based index replication model...")

    # Scale returns to improve numerical stability of the optimizer
    X_tr = X_train.values * 100.0
    y_tr = y_train.values * 100.0
    X_v = X_val.values * 100.0
    y_v = y_val.values * 100.0

    n_assets = X_tr.shape[1]

    # Regularization grid
    alphas = np.logspace(-4, -1, 50)

    tracking_errors = []
    asset_counts = []
    candidate_weights = []

    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    w0 = np.ones(n_assets) / n_assets

    for alpha in alphas:

        def objective(w):
            residuals = X_tr @ w - y_tr
            tracking_loss = np.mean(residuals ** 2)
            ridge_penalty = alpha * np.sum(w ** 2)
            return tracking_loss + ridge_penalty

        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )

        weights = result.x

        weights[weights < 1e-4] = 0.0

        if weights.sum() > 0:
            weights /= weights.sum()

        candidate_weights.append(weights)

        val_pred = (X_v / 100.0) @ weights
        val_te = np.std(val_pred - (y_v / 100.0)) * np.sqrt(252)

        tracking_errors.append(val_te)
        asset_counts.append(np.sum(weights > 0))

    valid_candidates = [
        (te, idx)
        for idx, (te, count) in enumerate(zip(tracking_errors, asset_counts))
        if min_assets <= count <= max_assets
    ]

    if valid_candidates:
        best_idx = min(valid_candidates, key=lambda x: x[0])[1]
        print(
            f"Selected solution with {asset_counts[best_idx]} active assets "
            f"within the target sparsity range."
        )
    else:
        print(
            "No solution satisfied the asset-count constraint. "
            "Selecting the minimum tracking-error portfolio."
        )
        best_idx = np.argmin(tracking_errors)

    best_weights = candidate_weights[best_idx]
    best_alpha = alphas[best_idx]

    print(f"Optimal alpha: {best_alpha:.8f}")
    print(f"Validation tracking error: {tracking_errors[best_idx] * 100:.2f}%")
    print(f"Active assets: {asset_counts[best_idx]}")

    weights_series = pd.Series(best_weights, index=X_train.columns)

    return (
        weights_series,
        alphas,
        tracking_errors,
        asset_counts,
        best_alpha
    )


def plot_ridge_sweep(alphas, tracking_errors, asset_counts, best_alpha):
    print("Generating Plot 4: Ridge regularization sweep...")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("L2 Regularization Strength (Alpha)", fontweight='bold')
    ax1.set_ylabel("Validation Tracking Error", color='steelblue')
    ax1.set_xscale('log')

    ax1.plot(
        alphas,
        tracking_errors,
        color='steelblue',
        linewidth=2
    )

    ax1.axvline(
        best_alpha,
        color='black',
        linestyle=':',
        label=f'Optimal Alpha ({best_alpha:.8f})'
    )

    ax1.tick_params(axis='y', labelcolor='steelblue')

    ax2 = ax1.twinx()

    ax2.set_ylabel("Active Assets", color='darkorange')

    ax2.plot(
        alphas,
        asset_counts,
        color='darkorange',
        linestyle='--'
    )

    ax2.tick_params(axis='y', labelcolor='darkorange')

    plt.title(
        "Ridge Regularization Sweep (Plot 4)",
        fontsize=12,
        fontweight='bold'
    )

    fig.tight_layout()
    plt.savefig("plots/04_ridge_sweep.png", dpi=150, bbox_inches='tight')
    plt.close()