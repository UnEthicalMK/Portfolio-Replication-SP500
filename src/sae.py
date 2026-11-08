"""Sparse autoencoder portfolio model with validation-aligned weights."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from torch import nn


class SparseAutoencoder(nn.Module):
    """Learns latent asset structure and a non-negative portfolio signal."""

    def __init__(
        self,
        n_assets: int,
        latent_dim: int,
        dropout: float,
        noise_sigma: float,
    ) -> None:
        super().__init__()
        self.noise_sigma = noise_sigma
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Linear(n_assets, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, n_assets, bias=False)
        self.tracking_head = nn.Linear(latent_dim, 1, bias=False)
        self.portfolio_logits = nn.Parameter(torch.zeros(n_assets))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noisy = x
        if self.training and self.noise_sigma:
            noisy = noisy + torch.randn_like(noisy) * self.noise_sigma
        latent = self.encoder(self.dropout(noisy))
        reconstruction = self.decoder(latent)
        prediction = self.tracking_head(latent)
        return prediction, reconstruction

    def dense_weights(self) -> torch.Tensor:
        return torch.softmax(self.portfolio_logits, dim=0)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _top_k_weights(
    dense_weights: torch.Tensor,
    columns: pd.Index,
    k: int,
    w_max: float,
) -> pd.Series:
    selected = torch.topk(dense_weights, k=min(k, len(columns))).indices
    weights = torch.zeros_like(dense_weights)
    weights[selected] = dense_weights[selected]
    weights = weights / weights.sum()
    for _ in range(len(columns)):
        over_cap = weights > w_max
        if not bool(over_cap.any()):
            break
        excess = (weights[over_cap] - w_max).sum()
        weights[over_cap] = w_max
        under_cap = (weights > 0) & ~over_cap
        if not bool(under_cap.any()):
            raise ValueError("SAE support cannot satisfy the configured weight cap")
        available = weights[under_cap]
        weights[under_cap] += excess * available / available.sum()
    if float(weights.max()) > w_max + 1e-8:
        raise ValueError("SAE weight-cap projection did not converge")
    return pd.Series(weights.detach().cpu().numpy(), index=columns)


def train_sae_candidate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    k: int,
    w_max: float,
    latent_dim: int,
    dropout: float,
    noise_sigma: float,
    reconstruction_weight: float,
    l1_weight: float,
    learning_rate: float,
    epochs: int,
    patience: int,
    seed: int,
) -> tuple[pd.Series, float]:
    """Train one SAE candidate and score its exact sparse portfolio on validation."""
    _seed_everything(seed)
    x_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)
    model = SparseAutoencoder(
        X_train.shape[1], latent_dim, dropout, noise_sigma
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    mse = nn.MSELoss()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        prediction, reconstruction = model(x_train)
        tracking_loss = mse(prediction, y_train_tensor)
        reconstruction_loss = mse(reconstruction, x_train)
        sparsity_loss = model.portfolio_logits.abs().mean()
        loss = (
            tracking_loss
            + reconstruction_weight * reconstruction_loss
            + l1_weight * sparsity_loss
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_weights = _top_k_weights(model.dense_weights(), X_train.columns, k, w_max)
        val_prediction = X_val[val_weights.index].mul(val_weights, axis=1).sum(axis=1)
        val_loss = float(np.mean((val_prediction.to_numpy() - y_val.to_numpy()) ** 2))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("SAE did not produce a validation candidate")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        weights = _top_k_weights(model.dense_weights(), X_train.columns, k, w_max)
    return weights, best_loss


def select_sae(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    k_set: tuple[int, ...],
    w_max: float,
    settings: object,
) -> tuple[pd.Series, dict[str, float]]:
    candidates: list[tuple[float, pd.Series, int, int]] = []
    for k in k_set:
        if k > X_train.shape[1] or k * w_max < 1:
            continue
        for seed in settings.seeds:
            weights, score = train_sae_candidate(
                X_train, y_train, X_val, y_val, k, w_max,
                settings.latent_dim, settings.dropout, settings.noise_sigma,
                settings.reconstruction_weight, settings.l1_weight,
                settings.learning_rate, settings.epochs, settings.patience, seed,
            )
            candidates.append((score, weights, k, seed))
    if not candidates:
        raise ValueError("No feasible SAE candidates")
    score, weights, k, seed = min(candidates, key=lambda item: item[0])
    return weights, {
        "validation_mse": score,
        "k": float(k),
        "seed": float(seed),
    }
