"""
Market Scanner Agent - quét Top N coin + chạy tất cả strategy
Hỗ trợ OI + Funding cho strategy OI_Funding
"""
from typing import List
import pandas as pd
from loguru import logger
from core.exchange import ExchangeWrapper
from agents.strategies import get_all_strategies
from core.models import Signal


class ScannerAgent:
    def __init__(self, exchange: ExchangeWrapper, top_n: int = 20, timeframes: List[str] = None):
        self.exchange = exchange
        self.top_n = top_n
        self.timeframes = timeframes or ["15m", "1h"]
        self.strategies = get_all_strategies()
        logger.info(f"ScannerAgent ready | Top {top_n} | TFs={self.timeframes} | Strategies={len(self.strategies)}")

    def scan(self) -> List[Signal]:
        symbols = self.exchange.get_top_symbols(self.top_n)
        logger.info(f"Scanning {len(symbols)} symbols: {symbols[:5]}...")

        all_signals: List[Signal] = []

        for symbol in symbols:
            funding_rate = None
            oi_value = None
            try:
                fr = self.exchange.fetch_funding_rate(symbol)
                if fr and "fundingRate" in fr:
                    funding_rate = float(fr["fundingRate"])
                oi = self.exchange.fetch_open_interest(symbol)
                if oi and "openInterestAmount" in oi:
                    oi_value = float(oi["openInterestAmount"])
                elif oi and "openInterest" in oi:
                    oi_value = float(oi["openInterest"])
            except Exception:
                pass

            for tf in self.timeframes:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, tf, limit=200)
                    if not ohlcv or len(ohlcv) < 50:
                        continue
                    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

                    for strat in self.strategies:
                        if strat.name == "OI_Funding":
                            oi_series = None
                            if oi_value is not None:
                                oi_series = pd.Series([oi_value * 0.97] * (len(df) - 1) + [oi_value])
                            signals = strat.generate(df, symbol, tf, oi_series=oi_series, funding_rate=funding_rate)
                        else:
                            signals = strat.generate(df, symbol, tf)
                        all_signals.extend(signals)
                except Exception as e:
                    logger.warning(f"Scan error {symbol} {tf}: {e}")
                    continue

        all_signals.sort(key=lambda s: s.confidence, reverse=True)
        logger.info(f"Found {len(all_signals)} raw signals")
        return all_signals
