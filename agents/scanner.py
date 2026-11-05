"""
Market Scanner Agent - quét Top N coin + chạy TẤT CẢ strategy được bật
Hỗ trợ OI + Funding cho strategy OI_Funding
"""
from typing import List, Optional, Dict, Any
from collections import Counter
import pandas as pd
from loguru import logger
from core.exchange import ExchangeWrapper
from agents.strategies import get_all_strategies, list_strategy_names
from core.models import Signal


class ScannerAgent:
    def __init__(
        self,
        exchange: ExchangeWrapper,
        top_n: int = 20,
        timeframes: List[str] = None,
        enabled_strategies: Optional[List[str]] = None,
        strategy_params: Optional[Dict[str, dict]] = None,
        excluded_symbols: Optional[List[str]] = None,
    ):
        self.exchange = exchange
        self.top_n = top_n
        self.timeframes = timeframes or ["15m", "1h"]
        self.enabled_strategies = enabled_strategies
        self.strategy_params = strategy_params or {}
        self.strategies = get_all_strategies(
            enabled=enabled_strategies,
            strategy_params=strategy_params,
        )
        self.last_dfs: Dict[str, pd.DataFrame] = {}
        self.excluded_symbols: set = set()
        self.set_excluded_symbols(excluded_symbols or [])
        names = [s.name for s in self.strategies]
        logger.info(
            f"ScannerAgent ready | Top {top_n} | TFs={self.timeframes} | "
            f"Excluded={len(self.excluded_symbols)} | "
            f"Strategies ({len(names)}): {names}"
        )

    @staticmethod
    def normalize_symbol(sym: str, quote: str = "USDT") -> str:
        if not sym:
            return ""
        s = str(sym).strip().upper().replace(" ", "")
        if not s:
            return ""
        if f":{quote}" in s:
            return s
        if s.endswith(f"/{quote}"):
            return f"{s}:{quote}"
        if s.endswith(quote) and "/" not in s:
            base = s[: -len(quote)]
            if base:
                return f"{base}/{quote}:{quote}"
        if "/" in s and ":" not in s:
            return f"{s}:{quote}"
        if "/" not in s and ":" not in s:
            return f"{s}/{quote}:{quote}"
        return s

    def set_excluded_symbols(self, symbols) -> int:
        out = set()
        for x in symbols or []:
            n = self.normalize_symbol(str(x))
            if n:
                out.add(n)
                base = n.split("/")[0]
                if base:
                    out.add(base)
        self.excluded_symbols = out
        return len([s for s in out if "/" in s])

    def is_excluded(self, symbol: str) -> bool:
        if not self.excluded_symbols:
            return False
        n = self.normalize_symbol(symbol)
        if n in self.excluded_symbols:
            return True
        base = (n.split("/")[0] if n else "").upper()
        if base and base in self.excluded_symbols:
            return True
        if str(symbol).strip().upper() in self.excluded_symbols:
            return True
        return False

    def excluded_list_display(self) -> list:
        return sorted(s for s in self.excluded_symbols if "/" in s)

    def reload_strategies(
        self,
        enabled: Optional[List[str]] = None,
        strategy_params: Optional[Dict[str, dict]] = None,
    ):
        if enabled is not None:
            self.enabled_strategies = enabled
        if strategy_params is not None:
            self.strategy_params = strategy_params
        self.strategies = get_all_strategies(
            enabled=self.enabled_strategies,
            strategy_params=self.strategy_params,
        )
        logger.info(f"Strategies reloaded: {[s.name for s in self.strategies]}")

    def scan(self) -> List[Signal]:
        symbols = self.exchange.get_top_symbols(self.top_n)
        if self.excluded_symbols:
            before = len(symbols)
            symbols = [s for s in symbols if not self.is_excluded(s)]
            skipped = before - len(symbols)
            if skipped:
                logger.info(f"Excluded {skipped} symbol(s) from scan blacklist")
        logger.info(
            f"Scanning {len(symbols)} symbols | strategies={[s.name for s in self.strategies]} "
            f"| blacklist={len(self.excluded_list_display())}"
        )

        all_signals: List[Signal] = []
        per_strat: Counter = Counter()
        self.last_dfs = {}

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
                    df = pd.DataFrame(
                        ohlcv,
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    self.last_dfs[symbol] = df
                    self.last_dfs[f"{symbol}_{tf}"] = df

                    for strat in self.strategies:
                        try:
                            if strat.name == "OI_Funding":
                                oi_series = None
                                if oi_value is not None:
                                    oi_series = pd.Series(
                                        [oi_value * 0.97] * (len(df) - 1) + [oi_value]
                                    )
                                signals = strat.generate(
                                    df, symbol, tf,
                                    oi_series=oi_series,
                                    funding_rate=funding_rate,
                                )
                            else:
                                signals = strat.generate(df, symbol, tf)

                            if signals:
                                all_signals.extend(signals)
                                per_strat[strat.name] += len(signals)
                        except Exception as e:
                            logger.debug(f"Strategy {strat.name} error {symbol} {tf}: {e}")
                except Exception as e:
                    logger.warning(f"Scan error {symbol} {tf}: {e}")
                    continue

        all_signals.sort(key=lambda s: s.confidence, reverse=True)

        if per_strat:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(per_strat.items(), key=lambda x: -x[1]))
            logger.info(f"Found {len(all_signals)} raw signals | by strategy: {detail}")
        else:
            logger.info("Found 0 raw signals")

        return all_signals

    @staticmethod
    def available_strategies() -> List[str]:
        return list_strategy_names()
