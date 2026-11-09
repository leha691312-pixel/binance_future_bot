"""
Bot ghi status mỗi cycle → data/status.json
Dashboard đọc file này để hiển thị real-time.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional


STATUS_PATH = Path("data/status.json")


def write_status(
    mode: str = "",
    kill_switch: bool = False,
    kill_reason: str = "",
    equity: float = 0,
    daily_pnl: float = 0,
    open_positions: int = 0,
    consecutive_losses: int = 0,
    raw_signals: int = 0,
    filtered_signals: int = 0,
    executed: int = 0,
    skip_symbols: int = 0,
    recent_signals: List[Dict] = None,
    positions: List[Dict] = None,
    last_error: str = "",
    strategies: Optional[List[str]] = None,
    ml: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """Ghi status.json. Nhận thêm strategies, ml và mọi kwargs khác để không lỗi khi main mở rộng."""
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "kill_switch": kill_switch,
        "kill_reason": kill_reason,
        "equity": equity,
        "daily_pnl": daily_pnl,
        "open_positions": open_positions,
        "consecutive_losses": consecutive_losses,
        "raw_signals": raw_signals,
        "filtered_signals": filtered_signals,
        "executed": executed,
        "skip_symbols": skip_symbols,
        "recent_signals": recent_signals or [],
        "positions": positions or [],
        "last_error": last_error,
        "strategies": strategies or [],
        "ml": ml or {},
    }
    # Ghi thêm các field tùy chọn từ kwargs (paused, cycle_info, ...)
    for k, v in kwargs.items():
        if k not in data and v is not None:
            data[k] = v

    STATUS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_status() -> Dict[str, Any]:
    if not STATUS_PATH.exists():
        return {}
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
