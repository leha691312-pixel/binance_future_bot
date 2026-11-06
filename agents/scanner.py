"""
Market Scanner - Top N symbols × strategies.
Hỗ trợ quét song song (ThreadPoolExecutor) để rút ngắn thời gian cycle.
"""
from typing import List, Optional, Dict, Any, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        parallel: bool = True,
        max_workers: int = 8,
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
        self.parallel = bool(parallel)
        self.max_workers = max(1, int(max_workers or 1))
        names = [s.name for s in self.strategies]
        logger.info(
            f"ScannerAgent ready | Top {top_n} | TFs={self.timeframes} | "
            f"parallel={self.parallel} workers={self.max_workers} | "
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
        return len(self.excluded_symbols)

    def is_excluded(self, symbol: str) -> bool:
        if not self.excluded_symbols:
            return False
        n = self.normalize_symbol(symbol)
        if n in self.excluded_symbols:
            return True
        base = n.split("/")[0] if n else ""
        return bool(base and base in self.excluded_symbols)

    def excluded_list_display(self) -> list:
        return sorted(self.excluded_symbols)

    def reload_strategies(
        self,
        enabled: Optional[List[str]] = None,
        strategy_params: Optional[Dict[str, dict]] = None,
    ):
        if strategy_params is not None:
            self.strategy_params = strategy_params
        if enabled is not None:
            self.enabled_strategies = enabled
        self.strategies = get_all_strategies(
            enabled=self.enabled_strategies,
            strategy_params=self.strategy_params,
        )
        logger.info(f"Strategies reloaded: {[s.name for s in self.strategies]}")

    def _scan_one_symbol(self, symbol: str) -> Tuple[List[Signal], Dict[str, pd.DataFrame], Counter]:
        """Quét 1 symbol × mọi TF/strategy. Thread-safe: không ghi self.last_dfs."""
        local_signals: List[Signal] = []
        local_dfs: Dict[str, pd.DataFrame] = {}
        per_strat: Counter = Counter()

        funding_rate = None
        oi_value = None
        try:
            fr = self.exchange.fetch_funding_rate(symbol)
            if fr and "fundingRate" in fr:
                funding_rate = float(fr["fundingRate"])
        except Exception:
            pass
        try:
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
                local_dfs[symbol] = df
                local_dfs[f"{symbol}_{tf}"] = df

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
                            local_signals.extend(signals)
                            per_strat[strat.name] += len(signals)
                    except Exception as e:
                        logger.debug(f"Strategy {strat.name} error {symbol} {tf}: {e}")
            except Exception as e:
                logger.warning(f"Scan error {symbol} {tf}: {e}")
                continue

        return local_signals, local_dfs, per_strat

    def scan(self) -> List[Signal]:
        symbols = self.exchange.get_top_symbols(self.top_n)
        before = len(symbols)
        symbols = [s for s in symbols if not self.is_excluded(s)]
        if before != len(symbols):
            logger.info(f"Excluded {before - len(symbols)} symbol(s) from scan blacklist")

        logger.info(
            f"Scanning {len(symbols)} symbols | TFs={self.timeframes} | "
            f"strategies={len(self.strategies)} | parallel={self.parallel} "
            f"workers={self.max_workers if self.parallel else 1} | "
            f"blacklist={len(self.excluded_symbols)}"
        )

        all_signals: List[Signal] = []
        per_strat: Counter = Counter()
        self.last_dfs = {}

        if not symbols:
            logger.info("Found 0 raw signals")
            return []

        workers = self.max_workers if self.parallel else 1
        workers = max(1, min(workers, len(symbols)))

        if workers == 1:
            for symbol in symbols:
                sigs, dfs, cnt = self._scan_one_symbol(symbol)
                all_signals.extend(sigs)
                self.last_dfs.update(dfs)
                per_strat.update(cnt)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(self._scan_one_symbol, sym): sym for sym in symbols}
                for fut in as_completed(futs):
                    sym = futs[fut]
                    try:
                        sigs, dfs, cnt = fut.result()
                        all_signals.extend(sigs)
                        self.last_dfs.update(dfs)
                        per_strat.update(cnt)
                    except Exception as e:
                        logger.warning(f"Parallel scan failed {sym}: {e}")

        all_signals.sort(key=lambda s: s.confidence, reverse=True)

        if per_strat:
            detail = ", ".join(
                f"{k}={v}" for k, v in sorted(per_strat.items(), key=lambda x: -x[1])
            )
            logger.info(f"Found {len(all_signals)} raw signals | by strategy: {detail}")
        else:
            logger.info("Found 0 raw signals")

        return all_signals

    @staticmethod
    def available_strategies() -> List[str]:
        return list_strategy_names()
