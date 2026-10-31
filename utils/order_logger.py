"""
Lưu mọi lệnh (entry / SL / TP) ra file Excel
File: data/orders.xlsx
"""
from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger

try:
    import openpyxl
    from openpyxl import Workbook, load_workbook
    HAS_XLSX = True
except ImportError:
    HAS_XLSX = False
    import csv


class OrderLogger:
    def __init__(self, path: str = "data/orders.xlsx"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.path.with_suffix(".csv")
        self._ensure_file()

    def _ensure_file(self):
        headers = [
            "time", "mode", "action", "symbol", "side", "size",
            "entry", "sl", "tp", "order_id", "strategy", "status", "note",
        ]
        if HAS_XLSX:
            if not self.path.exists():
                wb = Workbook()
                ws = wb.active
                ws.title = "Orders"
                ws.append(headers)
                wb.save(self.path)
        else:
            if not self.csv_path.exists():
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    import csv
                    csv.writer(f).writerow(headers)

    def log(
        self,
        action: str,
        symbol: str,
        side: str,
        size: float,
        entry: float = 0,
        sl: float = 0,
        tp: float = 0,
        order_id: str = "",
        strategy: str = "",
        status: str = "",
        mode: str = "",
        note: str = "",
    ):
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            mode,
            action,
            symbol,
            side,
            size,
            entry,
            sl,
            tp,
            order_id,
            strategy,
            status,
            note,
        ]
        try:
            if HAS_XLSX:
                wb = load_workbook(self.path)
                ws = wb.active
                ws.append(row)
                wb.save(self.path)
            else:
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    import csv
                    csv.writer(f).writerow(row)
            logger.debug(f"Order logged: {action} {symbol} {side}")
        except Exception as e:
            logger.warning(f"Failed to log order to file: {e}")
