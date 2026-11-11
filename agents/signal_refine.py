"""
Lam sach signal sau khi scan:
1) Consensus – can >= N strategy cung symbol + side
2) HTF bias – khong vao nguoc xu huong khung lon (EMA)
3) (Optional) bias strategy phai cung huong voi trigger
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from loguru import logger
from core.models import Signal, Side


DEFAULT_STYLE = {
    "ORB": "scalp",
    "LiquiditySweep": "scalp",
    "MARibbon": "scalp",
    "StochBB": "scalp",
    "PivotScalp": "scalp",
    "EMARSIScalp": "scalp",
    "OrderBlock": "day",
    "PremiumDiscount": "day",
    "SupplyDemand": "day",
    "FVG": "day",
    "BOS_CHOCH": "day",
    "TrendMA": "day",
    "MomentumRSI": "day",
    "PullbackEMA": "day",
    "RangeBounce": "day",
    "OI_Funding": "swing",
    "SwingMACross": "swing",
    "SwingRSI": "swing",
    "FibPullback": "swing",
    "BBSqueeze": "swing",
}

DEFAULT_BIAS_STRATEGIES = {
    "TrendMA",
    "BOS_CHOCH",
    "SwingMACross",
    "OI_Funding",
}
DEFAULT_TRIGGER_STRATEGIES = {
    "FVG",
    "PremiumDiscount",
    "OrderBlock",
    "SupplyDemand",
    "PullbackEMA",
    "LiquiditySweep",
    "StochBB",
    "MARibbon",
    "PivotScalp",
    "EMARSIScalp",
    "ORB",
    "MomentumRSI",
    "RangeBounce",
    "FibPullback",
    "BBSqueeze",
    "SwingRSI",
}


def _side_key(sig: Signal) -> str:
    return sig.side.value if hasattr(sig.side, "value") else str(sig.side)


def apply_consensus(
    signals: List[Signal],
    min_agree: int = 2,
    same_style: bool = False,
    style_map: Optional[Dict[str, str]] = None,
) -> List[Signal]:
    if not signals or min_agree <= 1:
        return list(signals)

    style_map = style_map or DEFAULT_STYLE
    groups: Dict[Tuple[str, str], List[Signal]] = defaultdict(list)

    for s in signals:
        groups[(s.symbol, _side_key(s))].append(s)

    out: List[Signal] = []
    dropped = 0

    for (symbol, side), group in groups.items():
        by_strat: Dict[str, Signal] = {}
        for s in group:
            name = s.strategy or "unknown"
            prev = by_strat.get(name)
            if prev is None or s.confidence > prev.confidence:
                by_strat[name] = s

        if same_style:
            style_buckets: Dict[str, List[Signal]] = defaultdict(list)
            for s in by_strat.values():
                st = style_map.get(s.strategy, "day")
                style_buckets[st].append(s)
            best_style, best_list = max(
                style_buckets.items(), key=lambda kv: len(kv[1])
            )
            candidates = best_list
            style_tag = best_style
        else:
            candidates = list(by_strat.values())
            style_tag = "any"

        n = len(candidates)
        if n < min_agree:
            dropped += len(group)
            continue

        best = max(
            candidates,
            key=lambda s: (s.confidence, s.risk_reward()),
        )
        names = sorted({s.strategy for s in candidates})
        boost = min(0.15, 0.04 * (n - 1))
        new_conf = min(0.98, float(best.confidence) + boost)

        reason = best.reason or ""
        agree_note = f"CONSENSUS x{n} [{', '.join(names)}]"
        if reason:
            reason = f"{agree_note} | {reason}"
        else:
            reason = agree_note

        extra = dict(best.extra or {})
        extra["agree_count"] = n
        extra["agree_strategies"] = names
        extra["consensus_style"] = style_tag

        out.append(
            Signal(
                symbol=best.symbol,
                side=best.side,
                entry=best.entry,
                sl=best.sl,
                tp=best.tp,
                confidence=new_conf,
                strategy=best.strategy,
                reason=reason,
                timeframe=best.timeframe,
                timestamp=best.timestamp,
                extra=extra,
            )
        )

    logger.info(
        f"Consensus: in={len(signals)} out={len(out)} dropped~{dropped} "
        f"| min_agree={min_agree} same_style={same_style}"
    )
    return out


def apply_bias_trigger(
    signals: List[Signal],
    require_bias: bool = True,
    bias_strategies: Optional[set] = None,
    trigger_strategies: Optional[set] = None,
) -> List[Signal]:
    if not require_bias or not signals:
        return list(signals)

    bias_strategies = bias_strategies or DEFAULT_BIAS_STRATEGIES
    trigger_strategies = trigger_strategies or DEFAULT_TRIGGER_STRATEGIES

    groups: Dict[Tuple[str, str], List[Signal]] = defaultdict(list)
    for s in signals:
        groups[(s.symbol, _side_key(s))].append(s)

    out: List[Signal] = []
    for key, group in groups.items():
        names = {s.strategy for s in group}
        has_bias = bool(names & bias_strategies)
        has_trig = bool(names & trigger_strategies)
        unknown = names - bias_strategies - trigger_strategies
        if unknown:
            has_trig = True
        if has_bias and has_trig:
            out.extend(group)
        elif not has_bias and has_trig:
            if len(names) >= 2:
                out.extend(group)

    logger.info(f"Bias/Trigger filter: in={len(signals)} out={len(out)}")
    return out


def _htf_trend(df: pd.DataFrame, ema_fast: int, ema_slow: int) -> Optional[str]:
    if df is None or (hasattr(df, "empty") and df.empty):
        return None
    if len(df) < max(ema_slow, ema_fast) + 2:
        return None
    close = df["close"].astype(float)
    ef = close.ewm(span=ema_fast, adjust=False).mean()
    es = close.ewm(span=ema_slow, adjust=False).mean()
    c = float(close.iloc[-1])
    f = float(ef.iloc[-1])
    s = float(es.iloc[-1])
    if f > s and c >= s * 0.998:
        return "LONG"
    if f < s and c <= s * 1.002:
        return "SHORT"
    return None


def apply_htf_bias(
    exchange,
    signals: List[Signal],
    timeframe: str = "1h",
    ema_fast: int = 21,
    ema_slow: int = 50,
    df_cache: Optional[Dict[str, pd.DataFrame]] = None,
) -> List[Signal]:
    if not signals:
        return []

    df_cache = df_cache if df_cache is not None else {}
    trend_cache: Dict[str, Optional[str]] = {}
    out: List[Signal] = []
    blocked = 0

    for s in signals:
        sym = s.symbol
        if sym not in trend_cache:
            df = df_cache.get(f"{sym}_{timeframe}")
            if df is None:
                df = df_cache.get(sym)
            if df is None or (hasattr(df, "empty") and df.empty) or len(df) < ema_slow + 2:
                try:
                    ohlcv = exchange.fetch_ohlcv(sym, timeframe, limit=max(120, ema_slow + 30))
                    if ohlcv and len(ohlcv) >= ema_slow:
                        df = pd.DataFrame(
                            ohlcv,
                            columns=["timestamp", "open", "high", "low", "close", "volume"],
                        )
                        df_cache[f"{sym}_{timeframe}"] = df
                except Exception as e:
                    logger.debug(f"HTF fetch {sym} {timeframe}: {e}")
                    df = None
            trend_cache[sym] = _htf_trend(df, ema_fast, ema_slow) if df is not None else None

        bias = trend_cache[sym]
        side = _side_key(s)
        if bias is None:
            out.append(s)
            continue
        if side == bias:
            extra = dict(s.extra or {})
            extra["htf_bias"] = bias
            extra["htf_tf"] = timeframe
            out.append(
                Signal(
                    symbol=s.symbol,
                    side=s.side,
                    entry=s.entry,
                    sl=s.sl,
                    tp=s.tp,
                    confidence=s.confidence,
                    strategy=s.strategy,
                    reason=s.reason,
                    timeframe=s.timeframe,
                    timestamp=s.timestamp,
                    extra=extra,
                )
            )
        else:
            blocked += 1

    logger.info(
        f"HTF bias ({timeframe} EMA{ema_fast}/{ema_slow}): "
        f"in={len(signals)} out={len(out)} blocked={blocked}"
    )
    return out


def refine_signals(
    signals: List[Signal],
    exchange=None,
    cfg: Optional[dict] = None,
    df_cache: Optional[Dict[str, pd.DataFrame]] = None,
    style_map: Optional[Dict[str, str]] = None,
) -> List[Signal]:
    cfg = cfg or {}
    if not signals:
        return []

    out = list(signals)

    if cfg.get("bias_trigger_enabled", False):
        out = apply_bias_trigger(
            out,
            require_bias=True,
            bias_strategies=set(cfg.get("bias_strategies") or DEFAULT_BIAS_STRATEGIES),
            trigger_strategies=set(cfg.get("trigger_strategies") or DEFAULT_TRIGGER_STRATEGIES),
        )

    if cfg.get("consensus_enabled", True):
        out = apply_consensus(
            out,
            min_agree=int(cfg.get("min_strategy_agree", 2)),
            same_style=bool(cfg.get("consensus_same_style", False)),
            style_map=style_map or DEFAULT_STYLE,
        )

    if cfg.get("htf_bias_enabled", True) and exchange is not None:
        out = apply_htf_bias(
            exchange,
            out,
            timeframe=str(cfg.get("htf_timeframe", "1h")),
            ema_fast=int(cfg.get("htf_ema_fast", 21)),
            ema_slow=int(cfg.get("htf_ema_slow", 50)),
            df_cache=df_cache,
        )

    out.sort(key=lambda s: s.confidence, reverse=True)
    logger.info(f"Signal refine done: final={len(out)}")
    return out
