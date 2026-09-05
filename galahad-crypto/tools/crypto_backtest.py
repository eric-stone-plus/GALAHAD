#!/usr/bin/env python3
"""
Crypto backtesting framework.
Multi-exchange ready (Binance + OKX conventions via ccxt when fetching
live data; ships with a deterministic sample-data generator for offline
runs).
Strategies: SMA cross / RSI / Bollinger.

Backtest-only tool: it never places orders.
"""

import datetime
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Missing dependencies: pip install pandas numpy")
    sys.exit(1)


@dataclass
class TradeConfig:
    """Trading configuration."""
    symbol: str = "BTC/USDT"
    exchange: str = "binance"  # binance / okx
    timeframe: str = "1h"
    start_date: str = "2024-01-01"
    end_date: str = "2026-06-15"
    initial_capital: float = 10000  # USDT
    position_size: float = 0.1     # 10% per trade
    fee_rate: float = 0.001        # 0.1% fee


@dataclass
class Trade:
    """Trade record."""
    entry_time: str
    exit_time: str
    symbol: str
    side: str  # "long" / "short"
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    fee: float
    strategy: str


@dataclass
class BacktestResult:
    """Backtest result."""
    strategy: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float = 0
    final_capital: float = 0
    total_return_pct: float = 0
    annual_return_pct: float = 0
    max_drawdown_pct: float = 0
    sharpe_ratio: float = 0
    win_rate: float = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0
    avg_loss: float = 0
    profit_factor: float = 0
    trades: list = field(default_factory=list)


def generate_sample_data(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    """
    Generate synthetic price data (swap in a ccxt fetch for real runs).
    """
    np.random.seed(42)
    dates = pd.date_range(start=start, end=end, freq="1h" if timeframe == "1h" else "1D")
    n = len(dates)

    # Random-walk prices
    returns = np.random.normal(0.0001, 0.02, n)
    prices = 30000 * np.exp(np.cumsum(returns))  # start at $30,000

    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices * (1 + np.random.uniform(-0.005, 0.005, n)),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
        "close": prices,
        "volume": np.random.uniform(100, 10000, n),
    })
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators."""
    # SMA
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["bb_mid"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std

    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    return df


def strategy_sma_cross(df: pd.DataFrame, config: TradeConfig) -> list:
    """SMA crossover strategy."""
    trades = []
    position = None

    for i in range(50, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        # Golden cross -> buy
        if (prev["sma_20"] <= prev["sma_50"] and
            row["sma_20"] > row["sma_50"] and
            position is None):
            position = {
                "entry_time": str(row["timestamp"]),
                "entry_price": row["close"],
                "side": "long",
                "quantity": (config.initial_capital * config.position_size) / row["close"],
            }

        # Death cross -> sell
        elif (prev["sma_20"] >= prev["sma_50"] and
              row["sma_20"] < row["sma_50"] and
              position is not None):
            exit_price = row["close"]
            qty = position["quantity"]
            pnl = (exit_price - position["entry_price"]) * qty
            fee = (position["entry_price"] * qty + exit_price * qty) * config.fee_rate

            trades.append(Trade(
                entry_time=position["entry_time"],
                exit_time=str(row["timestamp"]),
                symbol=config.symbol,
                side="long",
                entry_price=position["entry_price"],
                exit_price=exit_price,
                quantity=qty,
                pnl=pnl - fee,
                pnl_pct=(exit_price / position["entry_price"] - 1) * 100,
                fee=fee,
                strategy="SMA_Cross",
            ))
            position = None

    return trades


def strategy_rsi(df: pd.DataFrame, config: TradeConfig) -> list:
    """RSI overbought/oversold strategy."""
    trades = []
    position = None

    for i in range(20, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        # RSI < 30 -> buy
        if row["rsi"] < 30 and prev["rsi"] >= 30 and position is None:
            position = {
                "entry_time": str(row["timestamp"]),
                "entry_price": row["close"],
                "side": "long",
                "quantity": (config.initial_capital * config.position_size) / row["close"],
            }

        # RSI > 70 -> sell
        elif row["rsi"] > 70 and prev["rsi"] <= 70 and position is not None:
            exit_price = row["close"]
            qty = position["quantity"]
            pnl = (exit_price - position["entry_price"]) * qty
            fee = (position["entry_price"] * qty + exit_price * qty) * config.fee_rate

            trades.append(Trade(
                entry_time=position["entry_time"],
                exit_time=str(row["timestamp"]),
                symbol=config.symbol,
                side="long",
                entry_price=position["entry_price"],
                exit_price=exit_price,
                quantity=qty,
                pnl=pnl - fee,
                pnl_pct=(exit_price / position["entry_price"] - 1) * 100,
                fee=fee,
                strategy="RSI",
            ))
            position = None

    return trades


def strategy_bollinger(df: pd.DataFrame, config: TradeConfig) -> list:
    """Bollinger band mean-reversion strategy."""
    trades = []
    position = None

    for i in range(25, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        # Touch lower band -> buy
        if row["close"] < row["bb_lower"] and position is None:
            position = {
                "entry_time": str(row["timestamp"]),
                "entry_price": row["close"],
                "side": "long",
                "quantity": (config.initial_capital * config.position_size) / row["close"],
            }

        # Touch upper band -> sell
        elif row["close"] > row["bb_upper"] and position is not None:
            exit_price = row["close"]
            qty = position["quantity"]
            pnl = (exit_price - position["entry_price"]) * qty
            fee = (position["entry_price"] * qty + exit_price * qty) * config.fee_rate

            trades.append(Trade(
                entry_time=position["entry_time"],
                exit_time=str(row["timestamp"]),
                symbol=config.symbol,
                side="long",
                entry_price=position["entry_price"],
                exit_price=exit_price,
                quantity=qty,
                pnl=pnl - fee,
                pnl_pct=(exit_price / position["entry_price"] - 1) * 100,
                fee=fee,
                strategy="Bollinger",
            ))
            position = None

    return trades


def _bar_sharpe(trades: list, config: TradeConfig, df: pd.DataFrame) -> float:
    """Annualized Sharpe ratio from per-bar mark-to-market equity returns.

    Equity sits in cash between trades and is marked to the bar close while
    a position is open; each trade's fee is charged half at entry, half at
    exit. Annualization = sqrt(bars per year) for the configured timeframe
    (risk-free rate 0). Zero when fewer than two returns or zero variance.
    """
    if not trades or df.empty:
        return 0.0
    periods_per_year = {"1h": 24 * 365, "1d": 365}.get(config.timeframe, 365)
    entries = {pd.Timestamp(t.entry_time): t for t in trades}
    exits = {pd.Timestamp(t.exit_time): t for t in trades}
    equity = float(config.initial_capital)
    cash, qty = equity, 0.0
    open_trade = None
    prev = equity
    rets = []
    for i, row in enumerate(df.itertuples()):
        bar_ts = pd.Timestamp(row.timestamp)
        if open_trade is None and bar_ts in entries:
            open_trade = entries[bar_ts]
            qty = float(open_trade.quantity)
            cash = equity - qty * float(open_trade.entry_price) - float(open_trade.fee) / 2.0
        if open_trade is not None and bar_ts in exits:
            equity = cash + qty * float(open_trade.exit_price) - float(open_trade.fee) / 2.0
            cash, qty, open_trade = equity, 0.0, None
        elif open_trade is not None:
            equity = cash + qty * float(row.close)
        if i > 0:
            rets.append(equity / prev - 1.0)
        prev = equity
    r = np.asarray(rets, dtype=float)
    std = r.std(ddof=1) if len(r) >= 2 else 0.0
    if std <= 0:
        return 0.0
    return float(r.mean() / std * np.sqrt(periods_per_year))


def calculate_metrics(trades: list, config: TradeConfig, df: pd.DataFrame) -> BacktestResult:
    """Compute backtest metrics."""
    result = BacktestResult(
        strategy=",".join(set(t.strategy for t in trades)),
        symbol=config.symbol,
        timeframe=config.timeframe,
        start_date=config.start_date,
        end_date=config.end_date,
        initial_capital=config.initial_capital,
    )

    if not trades:
        return result

    result.total_trades = len(trades)
    result.trades = [asdict(t) for t in trades]

    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl <= 0]

    result.winning_trades = len(winning)
    result.losing_trades = len(losing)
    result.win_rate = len(winning) / len(trades) * 100 if trades else 0

    total_pnl = sum(t.pnl for t in trades)
    result.final_capital = config.initial_capital + total_pnl
    result.total_return_pct = (result.final_capital / config.initial_capital - 1) * 100

    # Annualized
    days = (pd.Timestamp(config.end_date) - pd.Timestamp(config.start_date)).days
    if days > 0:
        result.annual_return_pct = result.total_return_pct * 365 / days

    result.avg_win = sum(t.pnl for t in winning) / len(winning) if winning else 0
    result.avg_loss = sum(t.pnl for t in losing) / len(losing) if losing else 0

    gross_profit = sum(t.pnl for t in winning)
    gross_loss = abs(sum(t.pnl for t in losing))
    result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max Drawdown
    equity = [config.initial_capital]
    for t in trades:
        equity.append(equity[-1] + t.pnl)
    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > max_dd:
            max_dd = dd
    result.max_drawdown_pct = max_dd * 100

    result.sharpe_ratio = _bar_sharpe(trades, config, df)

    return result


def format_backtest_report(result: BacktestResult) -> str:
    """Format the backtest report."""
    lines = [
        "=" * 60,
        "Crypto Backtest Report",
        "=" * 60,
        f"Strategy      : {result.strategy}",
        f"Symbol        : {result.symbol}",
        f"Timeframe     : {result.timeframe}",
        f"Period        : {result.start_date} ~ {result.end_date}",
        "",
        f"Initial capital : ${result.initial_capital:,.2f}",
        f"Final capital   : ${result.final_capital:,.2f}",
        f"Total return    : {result.total_return_pct:+.2f}%",
        f"Annual return   : {result.annual_return_pct:+.2f}%",
        f"Max drawdown    : {result.max_drawdown_pct:.2f}%",
        f"Sharpe ratio    : {result.sharpe_ratio:.2f}",
        "",
        f"Total trades  : {result.total_trades}",
        f"Winning trades: {result.winning_trades}",
        f"Losing trades : {result.losing_trades}",
        f"Win rate      : {result.win_rate:.1f}%",
        f"Average win   : ${result.avg_win:,.2f}",
        f"Average loss  : ${result.avg_loss:,.2f}",
        f"Profit factor : {result.profit_factor:.2f}",
        "=" * 60,
    ]
    return "\n".join(lines)


def run_backtest():
    """Run all strategy backtests."""
    config = TradeConfig()

    print("Generating sample data...")
    df = generate_sample_data(config.symbol, config.timeframe, config.start_date, config.end_date)
    print(f"Bars: {len(df)}\n")

    print("Computing indicators...")
    df = calculate_indicators(df)

    strategies = [
        ("SMA Cross", strategy_sma_cross),
        ("RSI", strategy_rsi),
        ("Bollinger", strategy_bollinger),
    ]

    all_results = []
    for name, strategy_fn in strategies:
        print(f"Running strategy: {name}")
        trades = strategy_fn(df, config)
        result = calculate_metrics(trades, config, df)
        all_results.append(result)
        print(format_backtest_report(result))
        print()

    # Save results next to the module tree (repo-relative, no host paths)
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for r in all_results:
        summary.append({
            "strategy": r.strategy,
            "total_return_pct": r.total_return_pct,
            "max_drawdown_pct": r.max_drawdown_pct,
            "sharpe_ratio": r.sharpe_ratio,
            "win_rate": r.win_rate,
            "total_trades": r.total_trades,
            "profit_factor": r.profit_factor,
        })

    with open(out_dir / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved: {out_dir / 'backtest_summary.json'}")

    return all_results


if __name__ == "__main__":
    run_backtest()
