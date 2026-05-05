import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import yfinance as yf
import vectorbt as vbt


# ==========================================
# 1. DATA LOADING & CLEANING
# ==========================================
def load_and_clean_data(file_path):
    print(f"Loading and cleaning offline optimization data from '{file_path}'...")
    header_df = pd.read_csv(file_path, header=None, nrows=2)
    stock_names = header_df.iloc[0].ffill().values[1:]
    attributes = header_df.iloc[1].values[1:]

    df = pd.read_csv(file_path, skiprows=3, header=None, index_col=0)
    df.index.name = 'minute_index'
    df.columns = pd.MultiIndex.from_arrays([stock_names, attributes])

    df.replace(0, np.nan, inplace=True)
    df = df.ffill().bfill()
    return df


# ==========================================
# 2. CORE STRATEGY LOGIC
# ==========================================
def get_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def vectorized_backtest(df_stock, L, H, W_quantile, cost=0.001, slippage=0.001):
    df = df_stock.copy()
    if len(df) < 500:
        return -5.0, 0, 0.0, 0.0

    df['RSI'] = get_rsi(df['Close'])
    df['Vol_Thresh'] = df['Volume'].rolling(200).quantile(W_quantile)
    df['Ret'] = df['Close'].pct_change()

    df['Signal'] = 0
    df.loc[(df['RSI'] < L) & (df['Volume'] > df['Vol_Thresh']), 'Signal'] = 1
    df.loc[(df['RSI'] > H) & (df['Volume'] > df['Vol_Thresh']), 'Signal'] = -1

    df['Strat_Ret'] = df['Signal'].shift(1) * df['Ret']
    trades = df['Signal'].diff().fillna(0).abs()
    df['Strat_Ret'] -= (trades * (cost + slippage))

    num_trades = int(trades[trades != 0].count())

    # Regularization Penalty: Prevent selecting overfitted low-trade parameters
    estimated_years = (len(df) / 390) / 252 if len(df) > 390 else 1.0
    trades_per_year = num_trades / max(0.1, estimated_years)

    penalty_multiplier = 1.0
    if trades_per_year < 12:
        penalty_multiplier = 0.1
    elif trades_per_year < 24:
        penalty_multiplier = 0.6

    # Calculate metrics
    df['day_group'] = np.arange(len(df)) // 390
    daily_rets = df.groupby('day_group')['Strat_Ret'].sum()

    if len(daily_rets) < 2 or num_trades < 5:
        return -5.0, num_trades, df['Strat_Ret'].sum(), -1.0

    std = daily_rets.std()
    if std < 1e-8:
        return 0.0, num_trades, df['Strat_Ret'].sum(), 0.0

    raw_sharpe = (daily_rets.mean() / std) * np.sqrt(252)
    adjusted_sharpe = raw_sharpe * penalty_multiplier

    # Max Drawdown
    cum_rets = (1 + df['Strat_Ret']).cumprod()
    running_max = cum_rets.cummax()
    drawdowns = (cum_rets - running_max) / running_max
    max_dd = drawdowns.min()

    return adjusted_sharpe, num_trades, df['Strat_Ret'].sum(), max_dd


# ==========================================
# 3. PIPELINE: TWO-STAGE OPTIMIZER WITH VISUALS
# ==========================================
def run_optimization_and_backtest(opt_csv_path, real_tickers, start_date="2010-01-01", end_date="2022-12-31"):
    # Load 5-Stock Lab Data
    opt_df = load_and_clean_data(opt_csv_path)
    opt_tickers = list(opt_df.columns.levels[0].unique())

    # --------------------------------------------------------------
    # STAGE 1: COARSE PARAMETER SEARCH (Steps of 5)
    # --------------------------------------------------------------
    print("\n[Stage 1] Running COARSE parameter sweep on laboratory stocks...")
    L_coarse = [20, 25, 30, 35, 40]
    H_coarse = [65, 70, 75, 80, 85]
    W_coarse = [0.60, 0.70, 0.80]

    coarse_results = []
    for l, h, w in product(L_coarse, H_coarse, W_coarse):
        sharpes = []
        for ticker in opt_tickers:
            s, _, _, _ = vectorized_backtest(opt_df[ticker], l, h, w)
            sharpes.append(s)
        coarse_results.append({'L': l, 'H': h, 'W': w, 'Score': np.percentile(sharpes, 25)})

    coarse_res_df = pd.DataFrame(coarse_results)
    best_coarse_idx = coarse_res_df['Score'].idxmax()
    best_coarse = coarse_res_df.loc[best_coarse_idx]

    c_L, c_H, c_W = int(best_coarse['L']), int(best_coarse['H']), best_coarse['W']
    print(f" -> Coarse Search Winner: L={c_L}, H={c_H}, W={c_W:.2f} (Score: {best_coarse['Score']:.4f})")

    # Save Coarse Heatmap
    coarse_subset = coarse_res_df[coarse_res_df['W'] == c_W]
    coarse_matrix = coarse_subset.pivot(index='L', columns='H', values='Score')

    plt.figure(figsize=(8, 6))
    sns.heatmap(coarse_matrix, annot=True, cmap='RdYlGn', center=0, fmt=".4f")
    plt.title(f'Stage 1: Coarse Parameter Grid Search (W = {c_W:.2f})\nBest: L={c_L}, H={c_H}')
    plt.xlabel('Sell Threshold (H)')
    plt.ylabel('Buy Threshold (L)')
    plt.tight_layout()
    plt.savefig('coarse_grid_heatmap.png', dpi=300)
    plt.close()
    print(" -> Coarse Heatmap saved to 'coarse_grid_heatmap.png'")

    # --------------------------------------------------------------
    # STAGE 2: FINE PARAMETER SEARCH (Expanded & Centered Grid)
    # --------------------------------------------------------------
    print("\n[Stage 2] Running FINE parameter sweep around coarse center...")
    
    # Custom boundaries: L explicitly scans down to 25. H explicitly scans up to 79 (capturing 78).
    # Using step increments of 2 to build a clean 2D neighborhood.
    L_fine = list(range(25, 37, 2))  # Scans [25, 27, 29, 31, 33, 35]
    H_fine = list(range(67, 81, 2))  # Scans [67, 69, 71, 73, 75, 77, 79]
    W_fine = [max(0.40, c_W - 0.06), c_W, min(0.95, c_W + 0.06)]

    fine_results = []
    for l, h, w in product(L_fine, H_fine, W_fine):
        sharpes = []
        for ticker in opt_tickers:
            s, _, _, _ = vectorized_backtest(opt_df[ticker], l, h, w)
            sharpes.append(s)
        fine_results.append({'L': l, 'H': h, 'W': w, 'Score': np.percentile(sharpes, 25)})

    fine_res_df = pd.DataFrame(fine_results)

    # 2D Neighborhood Smoothing for Fine Plateau Detection
    best_plateau_score = -np.inf
    final_params = None  # (L, H, W)
    best_smoothed_matrix = None

    for w in W_fine:
        w_subset = fine_res_df[fine_res_df['W'] == w]
        pivot_grid = w_subset.pivot(index='L', columns='H', values='Score')

        # Uniform 2D rolling average filter
        smoothed = pivot_grid.rolling(window=3, center=True, min_periods=1).mean()
        smoothed = smoothed.T.rolling(window=3, center=True, min_periods=1).mean().T

        peak_smoothed_val = smoothed.max().max()
        if peak_smoothed_val > best_plateau_score:
            best_plateau_score = peak_smoothed_val
            best_l, best_h = smoothed.stack().idxmax()
            final_params = (best_l, best_h, w)
            best_smoothed_matrix = smoothed

    # Save Fine Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_smoothed_matrix, annot=True, cmap='RdYlGn', center=0, fmt=".4f")
    plt.title(
        f'Stage 2: Fine Smoothed Parameter Plateau (W = {final_params[2]:.2f})\nBest Selected: L={final_params[0]}, H={final_params[1]}')
    plt.xlabel('Sell Threshold (H)')
    plt.ylabel('Buy Threshold (L)')
    plt.tight_layout()
    plt.savefig('fine_grid_heatmap.png', dpi=300)
    plt.close()
    print(" -> Fine Heatmap saved to 'fine_grid_heatmap.png'")

    print(f"\n" + "=" * 65)
    print(f"TWO-STAGE OPTIMIZATION COMPLETE")
    print(f"Extracted Parameters: L={final_params[0]}, H={final_params[1]}, W={final_params[2]:.2f}")
    print(f"Plateau Neighborhood Score: {best_plateau_score:.4f}")
    print("=" * 65)

    # --------------------------------------------------------------
    # STAGE 3: VECTORBT PORTFOLIO EVALUATION & PLOT GENERATION
    # --------------------------------------------------------------
    print(f"\n[Stage 3] Downloading validation data for {len(real_tickers)} live assets...")

    clean_tickers = [t.replace('$', '').strip() for t in real_tickers]
    raw_download = yf.download(clean_tickers, start=start_date, end=end_date, progress=True)

    if raw_download.empty:
        raise ValueError("Yahoo Finance returned an empty dataset.")

    price = raw_download['Close'].copy()
    volume = raw_download['Volume'].copy()

    if isinstance(price.columns, pd.MultiIndex):
        price.columns = price.columns.get_level_values(0)
    if isinstance(volume.columns, pd.MultiIndex):
        volume.columns = volume.columns.get_level_values(0)

    price = price.dropna(how='all', axis=1).ffill().bfill()
    volume = volume.dropna(how='all', axis=1).ffill().bfill()
    active_tickers = list(price.columns)

    # Execute Strategy Core in VectorBT
    rsi_obj = vbt.RSI.run(price, window=14)
    rsi = rsi_obj.rsi
    if isinstance(rsi.columns, pd.MultiIndex):
        rsi.columns = rsi.columns.get_level_values(-1)

    vol_thresh = volume.rolling(200).quantile(final_params[2])
    if isinstance(vol_thresh.columns, pd.MultiIndex):
        vol_thresh.columns = vol_thresh.columns.get_level_values(-1)

    rsi = rsi[price.columns]
    vol_thresh = vol_thresh[price.columns]

    entries = ((rsi < final_params[0]) & (volume > vol_thresh)).fillna(False).astype(bool)
    exits = (rsi > final_params[1]).fillna(False).astype(bool)
    entries, exits = entries.vbt.signals.clean(exits)

    print("\nProcessing trades across the entire portfolio...")
    pf = vbt.Portfolio.from_signals(
        close=price,
        entries=entries,
        exits=exits,
        init_cash=1000,
        fees=0.001,
        slippage=0.001,
        freq='1D'
    )

    # Extract performance metrics
    sharpes = pf.sharpe_ratio()
    returns = pf.total_return() * 100.0
    max_dds = pf.max_drawdown() * 100.0
    trades_count = pf.trades.count()

    for s in [sharpes, returns, max_dds, trades_count]:
        if isinstance(s.index, pd.MultiIndex):
            s.index = s.index.get_level_values(-1)

    print(f"\n==========================================================================================")
    print(
        f"               REAL-WORLD TRADING PERFORMANCE REPORT (L={final_params[0]}, H={final_params[1]}, W={final_params[2]})")
    print(f"==========================================================================================")
    print(f"{'Ticker':<12} | {'Sharpe':<12} | {'Return':<12} | {'Max DD':<12} | {'Trades':<8}")
    print("-" * 65)

    for ticker in active_tickers:
        sh_val = sharpes.get(ticker, 0.0)
        ret_val = returns.get(ticker, 0.0)
        dd_val = max_dds.get(ticker, 0.0)
        tr_val = trades_count.get(ticker, 0)

        sh_val = 0.0 if np.isnan(sh_val) else sh_val
        ret_val = 0.0 if np.isnan(ret_val) else ret_val
        dd_val = 0.0 if np.isnan(dd_val) else dd_val

        print(f"{ticker:<12} | {sh_val:<12.4f} | {ret_val:>10.2f}% | {dd_val:>10.2f}% | {int(tr_val):<8}")

    print("-" * 65)
    print(f"{'AVERAGE':<12} | {np.nanmean(sharpes):<12.4f} | {np.nanmean(returns):>10.2f}%")
    print(f"==========================================================================================\n")

    # --------------------------------------------------------------
    # STAGE 4: SAVE INTERACTIVE PLOTS FOR ALL 15 TICKERS
    # --------------------------------------------------------------
    output_dir = "vbt_real_world_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Writing interactive HTML visual backtests to './{output_dir}/'...")

    for ticker in active_tickers:
        try:
            fig = pf[ticker].plot()
            fig.update_layout(
                title=f"VBT Backtest: {ticker} (L={final_params[0]}, H={final_params[1]}, W={final_params[2]})")

            clean_name = ticker.replace('^', '')
            file_path = f"{output_dir}/{clean_name}_backtest.html"
            fig.write_html(file_path)
            print(f" -> Saved {file_path}")
        except Exception as e:
            print(f" -> Failed to export plot for {ticker}: {e}")

    print("\nAll systems run complete. Heatmaps and interactive plots have been fully generated.")


if __name__ == "__main__":
    REAL_WORLD_ASSETS = [
        'AAPL', 'AMD', 'AMZN', 'DIS', 'GOOGL', 'INTC', 'JPM',
        'META', 'MS', 'MSFT', 'NFLX', 'NVDA', 'TSLA', 'V', '^NSEI'
    ]
    OPTIMIZATION_CSV = 'data.csv'

    if os.path.exists(OPTIMIZATION_CSV):
        run_optimization_and_backtest(OPTIMIZATION_CSV, REAL_WORLD_ASSETS)
    else:
        print(f"Error: Parameter optimization source '{OPTIMIZATION_CSV}' not found.")
