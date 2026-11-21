"""
src/stepwise.py
---------------
Greedy Forward Stepwise Selection Replicator.
Serves as the classical combinatorial baseline: greedily adds the single
foreign ADR that minimizes residual tracking error when solved via NNLS.
"""

import numpy as np
import pandas as pd
from scipy.optimize import nnls

def run_forward_stepwise(
    X_train_raw: pd.DataFrame,
    y_train_raw: pd.Series,
    max_k: int = 35
):
    """
    Sequentially builds portfolios of cardinality k = 1 to max_k.
    At step k, evaluates all unselected candidate assets and selects the one
    that yields the lowest residual sum of squares under NNLS.
    
    Returns:
        dict: {k: portfolio_df}
    """
    all_candidates = list(X_train_raw.columns)
    selected = []
    portfolios_by_k = {}
    
    y = y_train_raw.values
    print(f"\n[+] Executing Greedy Forward Stepwise Selection (k=1 to {max_k})...")
    
    for k in range(1, max_k + 1):
        best_candidate = None
        best_rss = float('inf')
        best_weights = None
        
        for cand in all_candidates:
            trial_set = selected + [cand]
            X_trial = X_train_raw[trial_set].values
            weights, residual = nnls(X_trial, y)
            rss = np.sum((y - X_trial @ weights) ** 2)
            
            if rss < best_rss:
                best_rss = rss
                best_candidate = cand
                best_weights = weights
                
        selected.append(best_candidate)
        all_candidates.remove(best_candidate)
        
        # Normalize weights
        if best_weights.sum() > 0:
            norm_w = best_weights / best_weights.sum()
        else:
            norm_w = np.ones(len(selected)) / len(selected)
            
        port_df = pd.DataFrame({
            'Ticker': selected,
            'Weight': norm_w
        }).sort_values(by='Weight', ascending=False).reset_index(drop=True)
        
        portfolios_by_k[k] = port_df
        if k in [5, 10, 15, 20, 25, 30]:
            print(f"    Stepwise k={k:>2}: Added '{best_candidate:<5}' | Top holding: {port_df.iloc[0]['Ticker']} ({port_df.iloc[0]['Weight']*100:.1f}%)")
            
    print(f"[+] Stepwise Selection complete for k=1..{max_k}.")
    return portfolios_by_k
