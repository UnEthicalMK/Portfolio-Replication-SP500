import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader

plt.style.use('seaborn-v0_8-whitegrid')


class GaussianNoise(nn.Module):
    """Applies Gaussian noise during training for regularisation."""

    def __init__(self, sigma=0.02):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training and self.sigma > 0:
            return x + torch.randn_like(x) * self.sigma
        return x


class RobustLinearSAE(nn.Module):
    """Linear sparse autoencoder with an additional tracking head."""

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
        return torch.softmax(self.raw_portfolio_signal(), dim=0)

    def forward(self, x):
        x = self.noise(x)
        x = self.dropout(x)

        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        prediction = self.tracking_head(latent)

        return prediction, reconstruction


def train_sae(X_train, y_train, X_val, y_val, top_k=None):
    print("Training Sparse Autoencoder replication model...")

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
    best_epoch = 0

    patience = 25
    no_improve = 0

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
                torch.norm(model.encoder.weight, p=1)
                + torch.norm(model.tracking_head.weight, p=1)
            )

            total_loss = (
                tracking_loss
                + alpha_recon * reconstruction_loss
                + lambda_l1 * l1_penalty
            )

            total_loss.backward()
            optimizer.step()

            epoch_train_loss += total_loss.item()

        avg_train_loss = epoch_train_loss / len(loader)

        model.eval()
        with torch.no_grad():
            val_pred, _ = model(X_val_t)
            val_loss = mse_loss(val_pred, y_val_t).item()

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.get_portfolio_weights().detach().cpu().numpy()
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    print("Generating Plot 9: SAE Learning Curves...")

    os.makedirs("plots", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        train_loss_history,
        label="Training Loss",
        color="steelblue",
        linewidth=1.5
    )

    ax.plot(
        val_loss_history,
        label="Validation Loss",
        color="darkorange",
        linewidth=2
    )

    ax.axvline(
        best_epoch,
        color="black",
        linestyle=":",
        label=f"Best Validation Epoch ({best_epoch})"
    )

    ax.set_title(
        "Sparse Autoencoder Learning Curves (Plot 9)",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/09_sae_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    if best_weights is None:
        best_weights = model.get_portfolio_weights().detach().cpu().numpy()

    weights = pd.Series(best_weights, index=X_train.columns)

    # Match sparsity level used by the benchmark models
    if top_k is not None and top_k < len(weights):
        print(f"Applying Top-{top_k} asset constraint...")

        keep_assets = weights.nlargest(top_k).index
        weights.loc[~weights.index.isin(keep_assets)] = 0.0

        if weights.sum() > 0:
            weights /= weights.sum()

    print(f"Validation loss: {best_val_loss:.6f}")

    return weights