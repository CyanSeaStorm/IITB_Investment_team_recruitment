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
        return -5.0, 0, 0.0, -1.0, -5.0  # Sharpe, trades, return, max_dd, calmar

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
    tot_return = df['Strat_Ret'].sum()

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
        return -5.0, num_trades, tot_return, -1.0, -5.0

    std = daily_rets.std()
    if std < 1e-8:
        return 0.0, num_trades, tot_return, 0.0, 0.0

    raw_sharpe = (daily_rets.mean() / std) * np.sqrt(252)
    adjusted_sharpe = raw_sharpe * penalty_multiplier

    # Max Drawdown & Calmar Ratio
    cum_rets = (1 + df['Strat_Ret']).cumprod()
    running_max = cum_rets.cummax()
    drawdowns = (cum_rets - running_max) / running_max
    max_dd = drawdowns.min()  # Will be negative (e.g. -0.15)

    # Avoid division by zero, scale calmar safely
    abs_dd = abs(max_dd)
    if abs_dd < 1e-5:
        calmar = adjusted_sharpe  # Fallback if zero drawdown
    else:
        # Calmar = Annualized Return / Max Drawdown
        annualized_return = daily_rets.mean() * 252
        calmar = (annualized_return / abs_dd) * penalty_multiplier

    return adjusted_sharpe, num_trades, tot_return, max_dd, calmar


# ==========================================
# 3. PIPELINE: TWO-STAGE MULTI-OBJECTIVE OPTIMIZER
# ==========================================
def run_optimization_and_backtest(opt_csv_path, real_tickers, start_date="2010-01-01", end_date="2022-12-31"):
    # Load 5-Stock Lab Data
    opt_df = load_and_clean_data(opt_csv_path)
    opt_tickers = list(opt_df.columns.levels[0].unique())

    # Multi-Objective Weights: 40% Sharpe, 30% Returns, 30% Calmar
    w1, w2, w3 = 0.40, 0.30, 0.30

    # --------------------------------------------------------------
    # STAGE 1: COARSE PARAMETER SEARCH (Steps of 5)
    # --------------------------------------------------------------
    print("\n[Stage 1] Running COARSE multi-objective parameter sweep...")
    L_coarse = [20, 25, 30, 35, 40]
    H_coarse = [65, 70, 75, 80, 85]
    W_coarse = [0.60, 0.70, 0.80]

    coarse_results = []
    for l, h, w in product(L_coarse, H_coarse, W_coarse):
        sharpes, returns, calmars = [], [], []
        for ticker in opt_tickers:
            s, _, r, _, c = vectorized_backtest(opt_df[ticker], l, h, w)
            sharpes.append(s)
            returns.append(r)
            calmars.append(c)
        
        # Pull 25th percentile across portfolio for robustness
        coarse_results.append({
            'L': l, 'H': h, 'W': w,
            'Raw_Sharpe': np.percentile(sharpes, 25),
            'Raw_Return': np.percentile(returns, 25),
            'Raw_Calmar': np.percentile(calmars, 25)
        })

    coarse_res_df = pd.DataFrame(coarse_results)

    # Min-Max Normalization to [0, 1] range
    for metric in ['Raw_Sharpe', 'Raw_Return', 'Raw_Calmar']:
        min_v = coarse_res_df[metric].min()
        max_v = coarse_res_df[metric].max()
        denom = (max_v - min_v) if (max_v - min_v) > 1e-8 else 1.0
        coarse_res_df[metric + '_norm'] = (coarse_res_df[metric] - min_v) / denom

    # Calculate Balanced Utility Score
    coarse_res_df['Score'] = (
        w1 * coarse_res_df['Raw_Sharpe_norm'] +
        w2 * coarse_res_df['Raw_Return_norm'] +
        w3 * coarse_res_df['Raw_Calmar_norm']
    )

    best_coarse_idx = coarse_res_df['Score'].idxmax()
    best_coarse = coarse_res_df.loc[best_coarse_idx]

    c_L, c_H, c_W = int(best_coarse['L']), int(best_coarse['H']), best_coarse['W']
    print(f" -> Coarse Winner: L={c_L}, H={c_H}, W={c_W:.2f} (Composite Score: {best_coarse['Score']:.4f})")

    # Save Coarse Heatmap
    coarse_subset = coarse_res_df[coarse_res_df['W'] == c_W]
    coarse_matrix = coarse_subset.pivot(index='L', columns='H', values='Score')

    plt.figure(figsize=(8, 6))
    sns.heatmap(coarse_matrix, annot=True, cmap='RdYlGn', center=0.5, fmt=".4f")
    plt.title(f'Stage 1: Coarse Multi-Objective Grid Search (W = {c_W:.2f})\nBest: L={c_L}, H={c_H}')
    plt.xlabel('Sell Threshold (H)')
    plt.ylabel('Buy Threshold (L)')
    plt.tight_layout()
    plt.savefig('coarse_grid_heatmap.png', dpi=300)
    plt.close()

    # --------------------------------------------------------------
    # STAGE 2: FINE PARAMETER SEARCH (Fully Centered Grid)
    # --------------------------------------------------------------
    print("\n[Stage 2] Running FINE multi-objective parameter sweep around coarse center...")
    
    # Fully expanded search space
    L_fine = list(range(23, 39, 2))  
    H_fine = list(range(67, 85, 2))  
    W_fine = [max(0.40, c_W - 0.06), c_W, min(0.95, c_W + 0.06)]

    fine_results = []
    for l, h, w in product(L_fine, H_fine, W_fine):
        sharpes, returns, calmars = [], [], []
        for ticker in opt_tickers:
            s, _, r, _, c = vectorized_backtest(opt_df[ticker], l, h, w)
            sharpes.append(s)
            returns.append(r)
            calmars.append(c)
        fine_results.append({
            'L': l, 'H': h, 'W': w,
            'Raw_Sharpe': np.percentile(sharpes, 25),
            'Raw_Return': np.percentile(returns, 25),
            'Raw_Calmar': np.percentile(calmars, 25)
        })

    fine_res_df = pd.DataFrame(fine_results)

    # Normalize Fine results globally
    for metric in ['Raw_Sharpe', 'Raw_Return', 'Raw_Calmar']:
        min_v = fine_res_df[metric].min()
        max_v = fine_res_df[metric].max()
        denom = (max_v - min_v) if (max_v - min_v) > 1e-8 else 1.0
        fine_res_df[metric + '_norm'] = (fine_res_df[metric] - min_v) / denom

    fine_res_df['Score'] = (
        w1 * fine_res_df['Raw_Sharpe_norm'] +
        w2 * fine_res_df['Raw_Return_norm'] +
        w3 * fine_res_df['Raw_Calmar_norm']
    )

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

    # Save Fine Multi-Objective Heatmap
    plt.figure(figsize=(9, 7))
    sns.heatmap(best_smoothed_matrix, annot=True, cmap='RdYlGn', center=0.5, fmt=".4f")
    plt.title(
        f'Stage 2: Fine Multi-Objective Smoothed Plateau (W = {final_params[2]:.2f})\n'
        f'Optimal: L={final_params[0]}, H={final_params[1]} | weights=[{w1:.2f}, {w2:.2f}, {w3:.2f}]'
    )
    plt.xlabel('Sell Threshold (H)')
    plt.ylabel('Buy Threshold (L)')
    plt.tight_layout()
    plt.savefig('fine_grid_heatmap.png', dpi=300)
    plt.close()
    print(" -> Balanced Multi-Objective Heatmap saved to 'fine_grid_heatmap.png'")

    print(f"\n" + "=" * 65)
    print(f"TWO-STAGE MULTI-OBJECTIVE OPTIMIZATION COMPLETE")
    print(f"Extracted Parameters: L={final_params[0]}, H={final_params[1]}, W={final_params[2]:.2f}")
    print(f"Composite Plateau Score: {best_plateau_score:.4f}")
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

    # Calculate Calmar for portfolio assets
    # Calmar Ratio = Annualized Return / Max Drawdown
    calmars = (pf.annualized_return() / pf.max_drawdown().abs())

    for s in [sharpes, returns, max_dds, trades_count, calmars]:
        if isinstance(s.index, pd.MultiIndex):
            s.index = s.index.get_level_values(-1)

    print(f"\n==========================================================================================")
    print(
        f"               REAL-WORLD TRADING PERFORMANCE REPORT (L={final_params[0]}, H={final_params[1]}, W={final_params[2]})")
    print(f"==========================================================================================")
    print(f"{'Ticker':<12} | {'Sharpe':<10} | {'Return':<11} | {'Max DD':<11} | {'Calmar':<10} | {'Trades':<8}")
    print("-" * 78)

    for ticker in active_tickers:
        sh_val = sharpes.get(ticker, 0.0)
        ret_val = returns.get(ticker, 0.0)
        dd_val = max_dds.get(ticker, 0.0)
        cal_val = calmars.get(ticker, 0.0)
        tr_val = trades_count.get(ticker, 0)

        sh_val = 0.0 if np.isnan(sh_val) else sh_val
        ret_val = 0.0 if np.isnan(ret_val) else ret_val
        dd_val = 0.0 if np.isnan(dd_val) else dd_val
        cal_val = 0.0 if np.isnan(cal_val) or np.isinf(cal_val) else cal_val

        print(f"{ticker:<12} | {sh_val:<10.4f} | {ret_val:>9.2f}% | {dd_val:>9.2f}% | {cal_val:<10.4f} | {int(tr_val):<8}")

    print("-" * 78)
    print(f"{'AVERAGE':<12} | {np.nanmean(sharpes):<10.4f} | {np.nanmean(returns):>9.2f}% | {np.nanmean(max_dds):>9.2f}% | {np.nanmean(calmars[~np.isinf(calmars)]):<10.4f}")
    print(f"==========================================================================================\n")

    # Save validation plots
    output_dir = "vbt_real_world_plots"
    os.makedirs(output_dir, exist_ok=True)
    for ticker in active_tickers:
        try:
            fig = pf[ticker].plot()
            fig.update_layout(
                title=f"VBT Backtest: {ticker} (L={final_params[0]}, H={final_params[1]}, W={final_params[2]})")
            clean_name = ticker.replace('^', '')
            fig.write_html(f"{output_dir}/{clean_name}_backtest.html")
        except Exception as e:
            print(f" -> Failed to export plot for {ticker}: {e}")


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
