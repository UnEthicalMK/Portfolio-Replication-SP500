"""
src/data_pipeline.py
--------------------
Institutional Data Ingestion & Preprocessing Pipeline for the
Cross-Market Synthetic Index Replication Engine.

Downloads, sanitizes, and caches daily Total Return price data for
a curated universe of liquid foreign ADRs and the S&P 500 Total Return Index (^SP500TR).
"""

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf

# Ground truth benchmark: S&P 500 Total Return Index (with SPY as validated fallback)
BENCHMARK_TICKER = "^SP500TR"
FALLBACK_BENCHMARK = "SPY"

def get_curated_adr_universe():
    """
    Returns curated pool of ~60-70 highly liquid foreign ADRs traded on
    NYSE / NASDAQ, spanning all 11 global macro sectors with complete 2014-2024 history.
    """
    return [
        # 1. Technology, Software & Semiconductors
        "TSM", "ASML", "SAP", "SONY", "STM", "UMC", "LOGI", "ERIC", "NOK", "CAJ",
        # 2. Healthcare & Global Pharmaceuticals
        "NVO", "NVS", "AZN", "GSK", "SNY", "TAK", "TEVA",
        # 3. Energy & Utilities
        "SHEL", "BP", "TTE", "ENI", "EQNR", "CNQ", "SU", "IMO", "PBR",
        # 4. Financials & Global Banking
        "RY", "TD", "BMO", "BNS", "CM", "HSBC", "UBS", "SAN", "BBVA",
        "ING", "BBD", "ITUB", "MUFG", "SMFG", "MFG", "NWG", "DB",
        # 5. Industrials, Aerospace, Transport & Materials
        "BHP", "RIO", "VALE", "ABB", "CNI", "CP", "CRH", "ERJ", "PKX", "SCCO", "GFI", "AU",
        # 6. Consumer Discretionary, Staples & Commerce Platforms
        "BABA", "MELI", "BIDU", "JD", "BUD", "DEO", "UL", "HMC", "TM", "FMX",
        # 7. Telecom & Communication Services
        "AMX", "VOD", "ORAN", "TEF", "BCE", "TU", "CHT", "SKM"
    ]

def fetch_and_clean_ticker_batch(tickers, start_date="2014-01-01", end_date="2024-12-31", max_retries=3):
    """
    Downloads adjusted closing prices in batches with exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False
            )
            if data.empty:
                raise ValueError("Empty dataframe returned by yfinance")
            
            # Extract Close (which is auto-adjusted for dividends and splits)
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.levels[0]:
                    closes = data['Close']
                else:
                    closes = data
            else:
                closes = data[['Close']] if 'Close' in data.columns else data
            return closes
        except Exception as e:
            wait_time = (2 ** attempt) + 1
            print(f"[!] Batch fetch failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            
    print(f"[!] Warning: Failed to fetch batch {tickers} after {max_retries} retries.")
    return pd.DataFrame()

def download_full_dataset(
    tickers=None,
    benchmark=BENCHMARK_TICKER,
    start_date="2014-01-01",
    end_date="2024-12-31",
    raw_save_path="data/raw/adr_prices.csv"
):
    """
    Downloads full historical price series for benchmark + ADR universe.
    Saves to data/raw/adr_prices.csv.
    """
    os.makedirs(os.path.dirname(raw_save_path), exist_ok=True)
    if tickers is None:
        tickers = get_curated_adr_universe()
        
    all_tickers = [benchmark] + [t for t in tickers if t != benchmark]
    print(f"\n[+] Ingesting Total Return series for {len(all_tickers)} assets ({start_date} to {end_date})...")
    
    # Batch download to respect rate limits
    batch_size = 12
    collected_dfs = []
    
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        print(f"    Fetching batch {i//batch_size + 1}/{(len(all_tickers) + batch_size - 1)//batch_size}: {batch[:3]}... ({len(batch)} assets)")
        batch_df = fetch_and_clean_ticker_batch(batch, start_date=start_date, end_date=end_date)
        if not batch_df.empty:
            collected_dfs.append(batch_df)
        time.sleep(1.0)
        
    if not collected_dfs:
        raise RuntimeError("Failed to download any data from data source.")
        
    combined_prices = pd.concat(collected_dfs, axis=1)
    # Remove duplicate columns if any
    combined_prices = combined_prices.loc[:, ~combined_prices.columns.duplicated()]
    
    # Benchmark verification & fallback check
    if benchmark not in combined_prices.columns or combined_prices[benchmark].dropna().empty:
        print(f"[!] Primary benchmark '{benchmark}' not available. Fetching fallback '{FALLBACK_BENCHMARK}'...")
        fallback_df = fetch_and_clean_ticker_batch([FALLBACK_BENCHMARK], start_date=start_date, end_date=end_date)
        if not fallback_df.empty:
            combined_prices[benchmark] = fallback_df[FALLBACK_BENCHMARK]
            print(f"[+] Successfully loaded fallback benchmark '{FALLBACK_BENCHMARK}' into '{benchmark}'.")
        else:
            raise RuntimeError(f"Could not fetch benchmark '{benchmark}' nor fallback '{FALLBACK_BENCHMARK}'.")

    # Save to disk
    combined_prices.to_csv(raw_save_path)
    print(f"[+] Raw price data saved to {raw_save_path}: {combined_prices.shape[0]} days x {combined_prices.shape[1]} assets.")
    return combined_prices

def clean_and_align_prices(raw_prices, benchmark=BENCHMARK_TICKER, max_ffill_days=3, max_missing_pct=0.08):
    """
    Cleans raw price series:
    1. Benchmark must be pristine. Forward-fill up to max_ffill_days, drop remaining NaNs.
    2. Drops assets with missing ratio > max_missing_pct.
    3. Strictly causal forward-fill (max 3 days). Backward fill is strictly prohibited.
    4. Drops any leading/trailing unfillable rows.
    """
    df = raw_prices.copy()
    
    if benchmark not in df.columns:
        raise KeyError(f"Benchmark '{benchmark}' is missing from raw prices.")
        
    # Forward-fill benchmark minor holiday gaps
    df[benchmark] = df[benchmark].ffill(limit=max_ffill_days)
    df = df.dropna(subset=[benchmark])
    
    # Missingness filter on candidate features
    missing_pcts = df.isnull().mean()
    excessive_missing = missing_pcts[missing_pcts > max_missing_pct].index.tolist()
    if excessive_missing:
        print(f"[-] Dropping {len(excessive_missing)} assets with > {max_missing_pct*100:.1f}% missing history: {excessive_missing}")
        df = df.drop(columns=excessive_missing)
        
    # Causal forward fill only
    df = df.ffill(limit=max_ffill_days)
    
    # Final dropna to guarantee zero NaN values
    clean_df = df.dropna()
    print(f"[+] Cleaned price series: {clean_df.shape[0]} days x {clean_df.shape[1]} assets (Benchmark: '{benchmark}')")
    return clean_df

def compute_log_returns(clean_prices, clip_threshold=0.10, save_path="data/processed/adr_log_returns.csv"):
    """
    Computes stationary log returns: R_t = ln(P_t) - ln(P_{t-1}).
    Applies winsorization at +/- clip_threshold (10%) to protect against bad ticks.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    log_returns = np.log(clean_prices / clean_prices.shift(1)).dropna()
    
    # Winsorization
    log_returns = log_returns.clip(lower=-clip_threshold, upper=clip_threshold)
    
    log_returns.to_csv(save_path)
    print(f"[+] Stationary log returns computed & saved to {save_path}: {log_returns.shape[0]} days x {log_returns.shape[1]} assets.")
    return log_returns

def load_or_build_pipeline(
    raw_path="data/raw/adr_prices.csv",
    processed_path="data/processed/adr_log_returns.csv",
    benchmark=BENCHMARK_TICKER,
    force_download=False
):
    """
    High-level entry point:
    If processed data exists and force_download is False, loads instantly (<0.2s).
    Otherwise, executes the full download, cleaning, and returns computation.
    """
    if not force_download and os.path.exists(processed_path):
        df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
        if benchmark in df.columns and len(df) > 1000:
            print(f"[+] Loaded cached clean returns from {processed_path}: {df.shape[0]} days x {df.shape[1]} assets.")
            return df

    if not force_download and os.path.exists(raw_path):
        print(f"[+] Loading existing raw prices from {raw_path}...")
        raw_df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    else:
        raw_df = download_full_dataset(benchmark=benchmark, raw_save_path=raw_path)
        
    clean_df = clean_and_align_prices(raw_df, benchmark=benchmark)
    log_returns = compute_log_returns(clean_df, save_path=processed_path)
    return log_returns
