"""
src/autoencoder.py
------------------
Nonlinear Latent Macro Factor Discovery & Dynamic Factor-Proportional Clustering.
Compresses the high-dimensional cross-market correlation manifold into
8 latent factors, discovers macro clusters dynamically via silhouette optimization,
and apportions stock quotas proportional to S&P 500 empirical factor exposure.
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
from scipy.optimize import nnls
import matplotlib.pyplot as plt

class MacroFactorAutoencoder(nn.Module):
    """
    Bottleneck autoencoder learning compressed representations of global market co-movement.
    """
    def __init__(self, n_features: int, n_factors: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(32, n_factors),
            nn.Tanh()  # Bounded latent space
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_factors, 32),
            nn.GELU(),
            nn.Linear(32, n_features)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

def train_macro_autoencoder(
    X_train_scaled: pd.DataFrame,
    X_val_scaled: pd.DataFrame,
    n_factors: int = 8,
    epochs: int = 120,
    batch_size: int = 32,
    lr: float = 0.003,
    weight_decay: float = 1e-4,
    patience: int = 20,
    seed: int = 42
):
    """
    Trains the autoencoder on training scaled returns with early stopping on validation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_features = X_train_scaled.shape[1]
    model = MacroFactorAutoencoder(n_features=n_features, n_factors=n_factors)
    
    # Use np.copy() to guarantee writable arrays and suppress PyTorch warnings
    train_vals = np.copy(X_train_scaled.values)
    val_vals = np.copy(X_val_scaled.values)
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_vals, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True
    )
    val_tensor = torch.tensor(val_vals, dtype=torch.float32)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    
    best_loss = float('inf')
    best_weights = None
    patience_counter = 0
    
    print(f"\n[+] Training Macro Factor Autoencoder ({n_features} assets -> {n_factors} latent factors)...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for (batch_x,) in train_loader:
            optimizer.zero_grad()
            recon, _ = model(batch_x)
            loss = loss_fn(recon, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
            
        train_loss /= len(X_train_scaled)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_recon, _ = model(val_tensor)
            val_loss = loss_fn(val_recon, val_tensor).item()
            
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping triggered at epoch {epoch+1}. Best val reconstruction loss: {best_loss:.6f}")
                break
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    model.eval()
    print(f"[+] Autoencoder convergence achieved. Best Val Loss: {best_loss:.6f}")
    return model

def extract_factor_loadings(model, X_train_scaled: pd.DataFrame):
    """
    Projects training returns onto the latent space to extract the T x n_factors series,
    then computes the factor loadings B (n_assets x n_factors) via OLS:
    B = (Z^T Z)^{-1} Z^T X
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(np.copy(X_train_scaled.values), dtype=torch.float32)
        _, z = model(x_tensor)
        Z = z.numpy()  # T x n_factors
        
    X = X_train_scaled.values  # T x n_assets
    # B = (Z^T Z)^{-1} Z^T X
    Z_pinv = np.linalg.pinv(Z)
    loadings = (Z_pinv @ X).T  # n_assets x n_factors
    return loadings, Z

def select_ae_cluster_portfolio(
    loadings: np.ndarray,
    feature_names: pd.Index,
    X_train_raw: pd.DataFrame,
    y_train_raw: pd.Series,
    target_k: int,
    seed: int = 42
):
    """
    Dynamic Factor-Proportional Macro Clustering & Asset Selection.
    
    Instead of arbitrarily forcing 1 stock per cluster (which depresses exposure
    to dominant sectors like Tech), this purely data-driven algorithm:
    1. Dynamically discovers the natural number of macro regimes M on the latent
       manifold using silhouette score maximization.
    2. Forms M macro clusters via spherical K-Means.
    3. Solves NNLS regression of the S&P 500 benchmark on cluster composite returns
       to determine the index's empirical factor exposure to each cluster.
    4. Apportions the target_k stock budget across clusters proportionally using the
       Largest Remainder Method (Hamilton quota rule), allowing high-beta sectors
       (e.g., Tech/Semiconductors) to receive multiple assets.
    5. Selects top assets within each cluster and fits final NNLS portfolio weights.
    
    Zero hardcoded values: all cluster counts, factor shares, and quotas are purely
    derived from the market covariance structure and target index dynamics.
    """
    norms = np.linalg.norm(loadings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_loadings = loadings / norms
    
    n_assets = len(feature_names)
    
    # 1. Dynamically determine optimal cluster count M via silhouette score
    max_m = min(loadings.shape[1], max(2, target_k - 1), n_assets // 3)
    min_m = min(2, max_m)
    
    best_m = min_m
    if max_m > min_m:
        best_sil = -1.0
        for m in range(min_m, max_m + 1):
            km = KMeans(n_clusters=m, random_state=seed, n_init=10)
            labels = km.fit_predict(normalized_loadings)
            if len(set(labels)) > 1:
                sil = silhouette_score(normalized_loadings, labels)
                if sil > best_sil:
                    best_sil = sil
                    best_m = m
    
    # 2. Fit K-Means with dynamically discovered M
    kmeans = KMeans(n_clusters=best_m, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(normalized_loadings)
    centroids = kmeans.cluster_centers_
    
    # 3. Compute cluster composite returns across training period
    cluster_returns = np.zeros((len(X_train_raw), best_m))
    cluster_members = {}
    for c in range(best_m):
        members = np.where(cluster_labels == c)[0]
        cluster_members[c] = members
        if len(members) > 0:
            c_tickers = feature_names[members]
            cluster_returns[:, c] = X_train_raw[c_tickers].mean(axis=1).values
            
    # 4. Regress S&P 500 on cluster returns to find empirical factor exposures
    beta_weights, _ = nnls(cluster_returns, y_train_raw.values)
    if beta_weights.sum() > 0:
        proportions = beta_weights / beta_weights.sum()
    else:
        proportions = np.ones(best_m) / best_m
        
    # 5. Apportion target_k stock quotas using the Largest Remainder Method
    raw_quotas = target_k * proportions
    quotas = np.floor(raw_quotas).astype(int)
    
    # Active clusters with non-zero weight get at least 1 seat if budget permits
    if target_k >= best_m:
        for c in range(best_m):
            if proportions[c] > 0 and quotas[c] == 0 and len(cluster_members[c]) > 0:
                quotas[c] = 1
                
    # Bound quotas by cluster size
    for c in range(best_m):
        quotas[c] = min(quotas[c], len(cluster_members[c]))
        
    # Distribute remaining quota to clusters with highest fractional remainders
    remaining = target_k - quotas.sum()
    if remaining > 0:
        remainders = raw_quotas - np.floor(raw_quotas)
        sorted_clusters = np.argsort(-remainders)
        for c in sorted_clusters:
            if remaining == 0:
                break
            if quotas[c] < len(cluster_members[c]):
                quotas[c] += 1
                remaining -= 1
    elif remaining < 0:
        # If over-allocated due to minimums, reduce from lowest non-zero quota clusters
        remainders = raw_quotas - np.floor(raw_quotas)
        sorted_clusters = np.argsort(remainders)
        for c in sorted_clusters:
            if remaining == 0:
                break
            if quotas[c] > 1:
                quotas[c] -= 1
                remaining += 1
                
    # Fallback safety if still remaining (e.g. tiny universe)
    while quotas.sum() < target_k:
        available = [c for c in range(best_m) if quotas[c] < len(cluster_members[c])]
        if not available:
            break
        quotas[available[0]] += 1
        
    # 6. Select top assets in each cluster ranked by S&P 500 correlation
    y_raw_vals = y_train_raw.values
    selected_indices = []
    for c in range(best_m):
        k_c = quotas[c]
        if k_c <= 0:
            continue
        members = cluster_members[c]
        # Rank by empirical correlation to S&P 500
        corrs = np.array([
            np.corrcoef(X_train_raw[feature_names[m]].values, y_raw_vals)[0, 1]
            for m in members
        ])
        sorted_idx = np.argsort(-corrs)
        top_members = members[sorted_idx[:k_c]].tolist()
        selected_indices.extend(top_members)
        
    selected_tickers = feature_names[selected_indices].tolist()
    
    # 7. Final NNLS portfolio weight fitting
    X_sub = X_train_raw[selected_tickers].values
    weights, _ = nnls(X_sub, y_raw_vals)
    
    if weights.sum() > 0:
        norm_weights = weights / weights.sum()
    else:
        norm_weights = np.ones(len(selected_tickers)) / len(selected_tickers)
        
    portfolio = pd.DataFrame({
        'Ticker': selected_tickers,
        'Weight': norm_weights
    }).sort_values(by='Weight', ascending=False).reset_index(drop=True)
    
    return portfolio

def plot_latent_factors(Z, dates, save_path="results/autoencoder_factors.png"):
    """
    Visualizes the 8 learned macro factor time-series.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_factors = Z.shape[1]
    fig, axes = plt.subplots(n_factors, 1, figsize=(12, 1.8 * n_factors), sharex=True)
    
    for i in range(n_factors):
        axes[i].plot(dates, Z[:, i], color='#2ca02c', linewidth=0.8)
        axes[i].axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        axes[i].set_ylabel(f'Factor {i+1}', fontsize=9, fontweight='bold')
        axes[i].grid(True, alpha=0.25)
        
    axes[-1].set_xlabel('Date', fontsize=11)
    fig.suptitle('Autoencoder Latent Macro Factor Trajectories (Cross-Market Manifold)', fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[+] Saved latent factor trajectories to {save_path}")
