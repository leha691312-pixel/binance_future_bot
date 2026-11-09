"""
Order guard: hủy SL/TP khi không còn vị thế trên symbol.
Gọi mỗi cycle + lúc startup.
"""
from typing import Dict, List, Any, Optional
from loguru import logger


def _symbol_key(symbol: str) -> str:
    if not symbol:
        return ""
    s = str(symbol).upper().replace(" ", "")
    s = s.replace(":USDT", "").replace("/USDT", "USDT")
    if "/" in s:
        s = s.replace("/", "")
    return s


def has_open_position(exchange, symbol: str) -> bool:
    """True nếu còn vị thế size > 0 trên symbol."""
    if not symbol:
        return False
    key = _symbol_key(symbol)
    try:
        for p in exchange.get_positions() or []:
            if abs(float(getattr(p, "size", 0) or 0)) <= 0:
                continue
            if _symbol_key(getattr(p, "symbol", "")) == key:
                return True
    except Exception as e:
        logger.debug(f"has_open_position {symbol}: {e}")
    return False


def _list_symbols_with_open_orders(exchange) -> List[str]:
    """Mọi symbol đang có lệnh thường hoặc conditional/algo."""
    symbols: List[str] = []
    seen = set()
    if getattr(exchange, "mode", None) is not None:
        mode_val = getattr(exchange.mode, "value", str(exchange.mode))
        if str(mode_val).lower() == "paper":
            return symbols
    try:
        for o in (exchange.exchange.fetch_open_orders() if exchange.exchange else []) or []:
            s = o.get("symbol")
            if s and s not in seen:
                seen.add(s)
                symbols.append(s)
    except Exception as e:
        logger.debug(f"list open orders: {e}")
    try:
        if hasattr(exchange, "_fapi"):
            algos = exchange._fapi("openAlgoOrders", "GET", {}) or []
            if isinstance(algos, dict):
                algos = algos.get("orders") or algos.get("data") or []
            for a in algos or []:
                s = a.get("symbol")
                if not s:
                    continue
                if hasattr(exchange, "_algo_id_to_ccxt_symbol"):
                    s = exchange._algo_id_to_ccxt_symbol(s)
                if s and s not in seen:
                    seen.add(s)
                    symbols.append(s)
    except Exception as e:
        logger.debug(f"list algo orders: {e}")
    return symbols


def cancel_sl_tp_if_no_position(exchange, symbol: Optional[str] = None) -> Dict[str, Any]:
    """Kiểm tra lại và hủy SL/TP khi KHÔNG sở hữu vị thế coin đó.

    - symbol: chỉ check 1 mã
    - symbol=None: quét mọi mã đang có lệnh mở / conditional

    Returns: {checked, canceled, kept, n_canceled}
    """
    result: Dict[str, Any] = {
        "checked": [],
        "canceled": [],
        "kept": [],
        "n_canceled": 0,
    }
    mode = getattr(exchange, "mode", None)
    mode_val = getattr(mode, "value", str(mode or "")).lower()
    if mode_val == "paper":
        return result

    if symbol:
        symbols = [symbol]
    else:
        symbols = _list_symbols_with_open_orders(exchange)

    for sym in symbols:
        if not sym:
            continue
        result["checked"].append(sym)
        if has_open_position(exchange, sym):
            result["kept"].append(sym)
            logger.debug(f"Keep SL/TP {sym}: still has position")
            continue
        logger.warning(
            f"No position on {sym} but open SL/TP remain → cancel protective orders"
        )
        try:
            n = exchange.cancel_all_orders(sym) or 0
            result["n_canceled"] += int(n)
            result["canceled"].append(sym)
            logger.info(f"Canceled SL/TP for flat symbol {sym} (n~={n})")
        except Exception as e:
            logger.error(f"cancel_sl_tp_if_no_position failed {sym}: {e}")

    if result["canceled"]:
        logger.info(
            f"SL/TP cleanup: canceled={len(result['canceled'])} "
            f"kept={len(result['kept'])} checked={len(result['checked'])}"
        )
    return result
