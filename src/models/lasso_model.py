import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

def train_lasso(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    min_assets: int = 3,
    max_assets: int = 15
):
    print("Training Model B: Strict Simplex Optimizer (Scaled for Precision)...")

    # THE FIX: Multiply by 100 to escape the 1e-6 floating-point trap
    X_tr = X_train.values * 100.0
    y_tr = y_train.values * 100.0
    X_v = X_val.values * 100.0
    y_v = y_val.values * 100.0
    
    n_assets = X_tr.shape[1]

    # Adjusted L2 Grid for scaled variance
    alphas = np.logspace(-4, -1, 50)
    
    tracking_errors = []
    asset_counts = []
    models_weights = []

    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    w0 = np.ones(n_assets) / n_assets

    for alpha in alphas:
        def objective(w):
            residuals = X_tr @ w - y_tr
            te_var = np.var(residuals)
            l2_penalty = alpha * np.sum(w**2)
            return te_var + l2_penalty

        # THE FIX: Explicitly tighten the tolerance (ftol) so the solver cannot exit early
        res = minimize(
            objective, 
            w0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        w_opt = res.x

        w_opt[w_opt < 1e-4] = 0.0
        if np.sum(w_opt) > 0:
            w_opt = w_opt / np.sum(w_opt)

        models_weights.append(w_opt)

        # Re-scale back to true decimals for accurate Tracking Error reporting
        val_pred = (X_v / 100.0) @ w_opt
        val_te = np.std(val_pred - (y_v / 100.0)) * np.sqrt(252)

        tracking_errors.append(val_te)
        asset_counts.append(np.sum(w_opt > 0))

    valid_candidates = []
    for i, (te, count) in enumerate(zip(tracking_errors, asset_counts)):
        if min_assets <= count <= max_assets:
            valid_candidates.append((te, i))

    if valid_candidates:
        best_idx = min(valid_candidates, key=lambda x: x[0])[1]
        print(f"  Selected optimal alpha from sparse region ({asset_counts[best_idx]} assets)")
    else:
        print("  WARNING: No alpha satisfied sparsity constraint. Falling back to minimum TE.")
        best_idx = np.argmin(tracking_errors)

    best_weights = models_weights[best_idx]
    best_alpha = alphas[best_idx]

    print(f"  Optimal L2 Alpha: {best_alpha:.8f}")
    print(f"  Validation TE: {tracking_errors[best_idx]*100:.2f}% | Active Assets: {asset_counts[best_idx]}")

    weights_series = pd.Series(best_weights, index=X_train.columns)

    return weights_series, alphas, tracking_errors, asset_counts, best_alpha


def plot_lasso_sweep(alphas, tracking_errors, asset_counts, best_alpha):
    print("Generating Plot 4: Optimization Hyperparameter Sweep...")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'steelblue'
    ax1.set_xlabel('L2 Regularization Strength (Alpha)', fontweight='bold')
    ax1.set_ylabel('Validation Tracking Error', color=color)
    ax1.set_xscale('log')
    ax1.plot(alphas, tracking_errors, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax1.axvline(best_alpha, color='black', linestyle=':', label=f'Optimal Alpha ({best_alpha:.8f})')

    ax2 = ax1.twinx()
    color = 'darkorange'
    ax2.set_ylabel('Active Assets (Sparsity)', color=color)
    ax2.plot(alphas, asset_counts, color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Simplex Optimizer Parameter Matrix (Plot 4)", fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.savefig("plots/04_lasso_sweep.png", dpi=150, bbox_inches='tight')
    plt.close()