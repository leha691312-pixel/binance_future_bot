from typing import List, Optional, Dict, Any
from .orb import ORBStrategy
from .liquidity_sweep import LiquiditySweepStrategy
from .order_block import OrderBlockStrategy
from .supply_demand import SupplyDemandStrategy
from .fvg import FVGStrategy
from .bos_choch import BOSCHOCHStrategy
from .oi_funding import OIFundingStrategy
from .trend_ma import TrendMAStrategy
from .momentum_rsi import MomentumRSIStrategy
from .pullback_ema import PullbackEMAStrategy
from .range_bounce import RangeBounceStrategy
from .ma_ribbon import MARibbonStrategy
from .stoch_bb import StochBBStrategy
from .pivot_scalp import PivotScalpStrategy
from .ema_rsi_scalp import EMARSIScalpStrategy
from .swing_ma_cross import SwingMACrossStrategy
from .swing_rsi import SwingRSIStrategy
from .fib_pullback import FibPullbackStrategy
from .bb_squeeze import BBSqueezeStrategy
from .premium_discount import PremiumDiscountStrategy
from .base import BaseStrategy

STRATEGY_MAP = {
    "ORB": ORBStrategy,
    "LiquiditySweep": LiquiditySweepStrategy,
    "OrderBlock": OrderBlockStrategy,
    "SupplyDemand": SupplyDemandStrategy,
    "FVG": FVGStrategy,
    "BOS_CHOCH": BOSCHOCHStrategy,
    "OI_Funding": OIFundingStrategy,
    "TrendMA": TrendMAStrategy,
    "MomentumRSI": MomentumRSIStrategy,
    "PullbackEMA": PullbackEMAStrategy,
    "RangeBounce": RangeBounceStrategy,
    "MARibbon": MARibbonStrategy,
    "StochBB": StochBBStrategy,
    "PivotScalp": PivotScalpStrategy,
    "EMARSIScalp": EMARSIScalpStrategy,
    # Strike.money swing
    "SwingMACross": SwingMACrossStrategy,
    "SwingRSI": SwingRSIStrategy,
    "FibPullback": FibPullbackStrategy,
    "BBSqueeze": BBSqueezeStrategy,
    "PremiumDiscount": PremiumDiscountStrategy,
}

STRATEGY_ALIASES = {
    "orb": "ORB",
    "liquidity_sweep": "LiquiditySweep",
    "liquiditysweep": "LiquiditySweep",
    "sweep": "LiquiditySweep",
    "order_block": "OrderBlock",
    "orderblock": "OrderBlock",
    "ob": "OrderBlock",
    "supply_demand": "SupplyDemand",
    "supplydemand": "SupplyDemand",
    "sd": "SupplyDemand",
    "fvg": "FVG",
    "bos_choch": "BOS_CHOCH",
    "bos": "BOS_CHOCH",
    "choch": "BOS_CHOCH",
    "oi_funding": "OI_Funding",
    "oi": "OI_Funding",
    "funding": "OI_Funding",
    "trend": "TrendMA",
    "trend_ma": "TrendMA",
    "trendma": "TrendMA",
    "momentum": "MomentumRSI",
    "momentum_rsi": "MomentumRSI",
    "momentumrsi": "MomentumRSI",
    "rsi": "MomentumRSI",
    "pullback": "PullbackEMA",
    "pullback_ema": "PullbackEMA",
    "pullbackema": "PullbackEMA",
    "range": "RangeBounce",
    "range_bounce": "RangeBounce",
    "rangebounce": "RangeBounce",
    "ma_ribbon": "MARibbon",
    "maribbon": "MARibbon",
    "ribbon": "MARibbon",
    "stoch_bb": "StochBB",
    "stochbb": "StochBB",
    "stochastic": "StochBB",
    "pivot": "PivotScalp",
    "pivot_scalp": "PivotScalp",
    "pivotscalp": "PivotScalp",
    "ema_rsi_scalp": "EMARSIScalp",
    "emarsiscalp": "EMARSIScalp",
    "ema_scalp": "EMARSIScalp",
    "swing_ma": "SwingMACross",
    "swing_ma_cross": "SwingMACross",
    "swingmacross": "SwingMACross",
    "swing_rsi": "SwingRSI",
    "swingrsi": "SwingRSI",
    "fib": "FibPullback",
    "fib_pullback": "FibPullback",
    "fibpullback": "FibPullback",
    "bb_squeeze": "BBSqueeze",
    "bbsqueeze": "BBSqueeze",
    "premium_discount": "PremiumDiscount",
    "premiumdiscount": "PremiumDiscount",
    "premium": "PremiumDiscount",
    "discount": "PremiumDiscount",
    "squeeze": "BBSqueeze",
}


def normalize_strategy_name(name: str) -> Optional[str]:
    if not name:
        return None
    n = str(name).strip()
    if n in STRATEGY_MAP:
        return n
    return STRATEGY_ALIASES.get(n.lower())


def get_all_strategies(
    params: dict = None,
    enabled: Optional[List[str]] = None,
    strategy_params: Optional[Dict[str, dict]] = None,
) -> List[BaseStrategy]:
    params = params or {}
    strategy_params = strategy_params or {}

    if enabled is None:
        names = list(STRATEGY_MAP.keys())
    else:
        names = []
        for raw in enabled:
            canon = normalize_strategy_name(raw)
            if canon and canon not in names:
                names.append(canon)

    strategies = []
    for name in names:
        cls = STRATEGY_MAP.get(name)
        if not cls:
            continue
        p = dict(params)
        p.update(strategy_params.get(name, {}))
        strategies.append(cls(p))
    return strategies


def list_strategy_names() -> List[str]:
    return list(STRATEGY_MAP.keys())
