from typing import List, Optional, Dict, Any
from .orb import ORBStrategy
from .liquidity_sweep import LiquiditySweepStrategy
from .order_block import OrderBlockStrategy
from .supply_demand import SupplyDemandStrategy
from .fvg import FVGStrategy
from .bos_choch import BOSCHOCHStrategy
from .oi_funding import OIFundingStrategy
from .base import BaseStrategy

STRATEGY_MAP = {
    "ORB": ORBStrategy,
    "LiquiditySweep": LiquiditySweepStrategy,
    "OrderBlock": OrderBlockStrategy,
    "SupplyDemand": SupplyDemandStrategy,
    "FVG": FVGStrategy,
    "BOS_CHOCH": BOSCHOCHStrategy,
    "OI_Funding": OIFundingStrategy,
}

# Alias phổ biến (viết thường / viết tắt)
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
    """
    Trả về list strategy đã khởi tạo.
    - enabled: list tên strategy được bật (None = tất cả)
    - strategy_params: {"FVG": {"rr": 2.5}, ...} override params từng strategy
    """
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
