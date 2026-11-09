"""
Chạy backtest + walk-forward + stress test
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from backtest.engine import BacktestEngine, walk_forward, stress_test
from core.exchange import ExchangeWrapper
import ccxt


def fetch_historical(symbols, timeframe="15m", days=180):
    """Lấy data lịch sử từ Binance public"""
    exchange = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
    data = {}
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=days)).isoformat())

    for sym in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(sym, timeframe, since=since, limit=1500)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            data[sym] = df
            logger.info(f"Fetched {sym}: {len(df)} bars")
        except Exception as e:
            logger.warning(f"Failed {sym}: {e}")
    return data


def main():
    logger.info("=== Starting rigorous backtest ===")

    # Top coins (có thể thay đổi)
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]

    data = fetch_historical(symbols, timeframe="15m", days=120)
    if not data:
        logger.error("No data fetched")
        return

    # 1. Base backtest
    engine = BacktestEngine(initial_capital=10000, commission=0.0004, slippage=0.0002)
    result = engine.run(data)
    print("\n========== BASE BACKTEST ==========")
    for k, v in result.metrics.items():
        print(f"{k}: {v}")

    # 2. Stress test
    print("\n========== STRESS TEST ==========")
    stress_results = stress_test(data)
    for name, res in stress_results.items():
        m = res.metrics
        print(f"[{name}] Return: {m.get('total_return_pct', 0):.2f}% | MaxDD: {m.get('max_drawdown_pct', 0):.2f}% | Trades: {m.get('total_trades', 0)}")

    # 3. Walk-forward (nếu đủ data)
    print("\n========== WALK-FORWARD ==========")
    wf_results = walk_forward(data, train_days=45, test_days=15, step_days=15)
    for i, res in enumerate(wf_results):
        m = res.metrics
        print(f"Window {i+1}: Return {m.get('total_return_pct', 0):.2f}% | WR {m.get('win_rate', 0):.1f}% | Trades {m.get('total_trades', 0)}")

    logger.info("Backtest completed")


if __name__ == "__main__":
    main()
