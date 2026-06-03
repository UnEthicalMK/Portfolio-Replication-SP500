import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

# =========================================================
# Gaussian Noise
# =========================================================

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.02):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training and self.sigma > 0:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x

# =========================================================
# Robust Linear SAE
# =========================================================

class RobustLinearSAE(nn.Module):
    def __init__(self, num_assets, latent_dim=3, dropout_rate=0.3, noise_sigma=0.02):
        super().__init__()
        
        self.noise = GaussianNoise(sigma=noise_sigma)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.encoder = nn.Linear(num_assets, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, num_assets, bias=False)
        self.tracking_head = nn.Linear(latent_dim, 1, bias=False)

    def raw_portfolio_signal(self):
        signal = self.tracking_head.weight @ self.encoder.weight
        return signal.squeeze()

    def get_portfolio_weights(self):
        signal = self.raw_portfolio_signal()
        return torch.softmax(signal, dim=0)

    def forward(self, x):
        x = self.noise(x)
        x = self.dropout(x)
        
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        prediction = self.tracking_head(latent)
        
        return prediction, reconstruction

# =========================================================
# Training
# =========================================================

def train_sae(X_train, y_train, X_val, y_val, top_k=None):
    print("Training Model C: Sparse Autoencoder...")

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    
    X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = RobustLinearSAE(
        num_assets=X_train.shape[1],
        latent_dim=3,
        dropout_rate=0.3,
        noise_sigma=0.02
    )

    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    mse_loss = nn.MSELoss()

    alpha_recon = 0.25
    lambda_l1 = 5e-4

    best_val_loss = float("inf")
    best_weights = None

    patience = 25
    no_improve = 0

    # ----------------------------------
    # Telemetry Tracking
    # ----------------------------------
    train_loss_history = []
    val_loss_history = []

    for epoch in range(500):
        model.train()
        epoch_train_loss = 0.0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()

            pred_y, recon_X = model(batch_X)

            tracking_loss = mse_loss(pred_y, batch_y)
            reconstruction_loss = mse_loss(recon_X, batch_X)

            l1_penalty = (
                torch.norm(model.encoder.weight, p=1) + 
                torch.norm(model.tracking_head.weight, p=1)
            )

            total_loss = (
                tracking_loss
                + (alpha_recon * reconstruction_loss)
                + (lambda_l1 * l1_penalty)
            )

            total_loss.backward()
            optimizer.step()
            
            epoch_train_loss += total_loss.item()
            
        avg_train_loss = epoch_train_loss / len(loader)

        # ----------------------------------
        # Validation
        # ----------------------------------
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(X_val_t)
            val_loss = mse_loss(val_pred, y_val_t).item()
            
            # Record metrics for the plot
            train_loss_history.append(avg_train_loss)
            val_loss_history.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.get_portfolio_weights().detach().cpu().numpy()
                no_improve = 0
            else:
                no_improve += 1

        if no_improve >= patience:
            print(f"  Early stopping triggered at epoch {epoch}")
            break

    # =====================================================
    # Diagnostic Plot Generation (Plot 9)
    # =====================================================
    print("  Generating Plot 9: SAE Learning Curves...")
    os.makedirs('plots', exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(train_loss_history, label='Training Loss (Includes L1 & Recon)', color='steelblue', linewidth=1.5)
    ax.plot(val_loss_history, label='Validation Loss (Pure Tracking MSE)', color='darkorange', linewidth=2)
    
    # Mark the epoch where the best weights were found
    best_epoch = len(train_loss_history) - patience - 1
    if best_epoch > 0:
        ax.axvline(best_epoch, color='black', linestyle=':', label=f'Optimal Weights Selected (Epoch {best_epoch})')

    ax.set_title("SAE Learning Curves — Generalization vs. Memorization (Plot 9)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plots/09_sae_loss_curves.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Safety fallback
    if best_weights is None:
        best_weights = model.get_portfolio_weights().detach().cpu().numpy()

    weights = pd.Series(best_weights, index=X_train.columns)

    # =====================================================
    # Top-K Constraint (Apples-to-Apples Handshake)
    # =====================================================
    if top_k is not None and top_k < len(weights):
        print(f"  Applying Top-{top_k} truncation...")
        
        keep_assets = weights.nlargest(top_k).index
        weights.loc[~weights.index.isin(keep_assets)] = 0.0
        
        # Re-normalize to ensure 100% capital allocation
        if weights.sum() > 0:
            weights /= weights.sum()

    return weights