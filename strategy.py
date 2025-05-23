import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

def test_adf_stationarity(series, threshold=0.05):
    """Run ADF test and return whether the series is stationary."""
    p_value = adfuller(series.dropna())[1]
    return p_value < threshold, p_value

def get_stationary_pairs(data, pairs, pval_threshold=0.05):
    """Return pairs with stationary spread according to ADF test."""
    stationary = []
    for t1, t2, corr in pairs:
        spread = data[t1] - data[t2]
        stationary_flag, pval = test_adf_stationarity(spread, pval_threshold)
        if stationary_flag:
            stationary.append((t1, t2, corr, pval))
    return pd.DataFrame(stationary, columns=['Ticker1', 'Ticker2', 'Correlation', 'ADF p-value'])

def calculate_hedge_ratio(y, x):
    """Estimate hedge ratio via linear regression (y ~ x)."""
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    return model.params[1]  # coefficient of x

def compute_zscore(spread):
    """Compute z-score of a spread series."""
    return (spread - spread.mean()) / spread.std()

def generate_signals(z, entry_threshold=1.0, exit_threshold=0.3):
    """Generate trading signals based on z-score."""
    signals = pd.DataFrame(index=z.index)
    signals['long_entry'] = z < -entry_threshold
    signals['short_entry'] = z > entry_threshold
    signals['exit'] = z.abs() < exit_threshold

    signals['signal'] = 0
    signals.loc[signals['long_entry'], 'signal'] = 1
    signals.loc[signals['short_entry'], 'signal'] = -1
    signals.loc[signals['exit'], 'signal'] = 0

    # Forward-fill signal to maintain position
    signals['position'] = signals['signal'].replace(0, method='ffill').shift(1).fillna(0)

    return signals

def compute_pnl(spread, signals):
    """Compute daily P&L based on spread and positions."""
    spread_ret = spread.diff().fillna(0)
    signals['pnl'] = signals['position'] * spread_ret
    return signals