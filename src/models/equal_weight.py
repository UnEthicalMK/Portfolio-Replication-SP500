import pandas as pd


def train_equal_weight(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    top_k: int
) -> pd.Series:
    """
    Equal-weight benchmark portfolio.

    Assets are ranked using in-sample correlation with the target,
    and capital is allocated equally across the top_k names.
    """

    print(f"Training Equal-Weight benchmark ({top_k} assets)...")

    # Rank assets using training-period correlations only
    correlations = X_train.corrwith(y_train).sort_values(ascending=False)

    # Select the strongest proxy assets
    selected_assets = correlations.head(top_k).index

    # Create portfolio weight vector
    weights = pd.Series(0.0, index=X_train.columns)
    weights[selected_assets] = 1.0 / top_k

    return weights