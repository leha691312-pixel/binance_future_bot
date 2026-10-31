from .orb import ORBStrategy
from .liquidity_sweep import LiquiditySweepStrategy
from .order_block import OrderBlockStrategy
from .supply_demand import SupplyDemandStrategy
from .fvg import FVGStrategy
from .bos_choch import BOSCHOCHStrategy
from .oi_funding import OIFundingStrategy

STRATEGY_MAP = {
    "ORB": ORBStrategy,
    "LiquiditySweep": LiquiditySweepStrategy,
    "OrderBlock": OrderBlockStrategy,
    "SupplyDemand": SupplyDemandStrategy,
    "FVG": FVGStrategy,
    "BOS_CHOCH": BOSCHOCHStrategy,
    "OI_Funding": OIFundingStrategy,
}

def get_all_strategies(params: dict = None):
    return [cls(params) for cls in STRATEGY_MAP.values()]
