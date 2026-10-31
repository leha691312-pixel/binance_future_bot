"""
Exchange wrapper hỗ trợ 3 mode: paper / testnet / live
Sử dụng ccxt cho tính nhất quán.
"""
import os
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import ccxt
from loguru import logger
from core.models import Mode, Side, Position, OrderStatus


class ExchangeWrapper:
    def __init__(self, mode: str = "paper", config: dict = None):
        self.mode = Mode(mode)
        self.config = config or {}
        self.exchange = None
        self.paper_balance = float(self.config.get("initial_capital", 10000))
        self.paper_positions: Dict[str, Position] = {}
        self.paper_orders: Dict[str, dict] = {}
        self.paper_trades: List[dict] = []
        self._init_exchange()

    def _init_exchange(self):
        if self.mode == Mode.PAPER:
            logger.info("Exchange mode: PAPER (local simulation)")
            return

        options = {
            "defaultType": "future",
            "adjustForTimeDifference": True,
        }

        if self.mode == Mode.TESTNET:
            self.exchange = ccxt.binanceusdm({
                "apiKey": os.getenv("BINANCE_TESTNET_API_KEY", ""),
                "secret": os.getenv("BINANCE_TESTNET_API_SECRET", ""),
                "options": options,
                "enableRateLimit": True,
            })
            self.exchange.set_sandbox_mode(True)
            logger.info("Exchange mode: TESTNET (demo.binance.com)")
        else:  # LIVE
            self.exchange = ccxt.binanceusdm({
                "apiKey": os.getenv("BINANCE_API_KEY", ""),
                "secret": os.getenv("BINANCE_API_SECRET", ""),
                "options": options,
                "enableRateLimit": True,
            })
            logger.warning("Exchange mode: LIVE – REAL MONEY")

        # Load markets
        try:
            self.exchange.load_markets()
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 500) -> List[list]:
        if self.mode == Mode.PAPER:
            # In pure paper we still need real data for signals → use public endpoint
            public = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
            return public.fetch_ohlcv(symbol, timeframe, limit=limit)
        return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    def fetch_ticker(self, symbol: str) -> dict:
        if self.mode == Mode.PAPER:
            public = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
            return public.fetch_ticker(symbol)
        return self.exchange.fetch_ticker(symbol)

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        if self.mode == Mode.PAPER:
            public = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
            return public.fetch_order_book(symbol, limit)
        return self.exchange.fetch_order_book(symbol, limit)

    def fetch_funding_rate(self, symbol: str) -> dict:
        try:
            if self.mode == Mode.PAPER:
                public = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
                return public.fetch_funding_rate(symbol)
            return self.exchange.fetch_funding_rate(symbol)
        except Exception:
            return {}

    def fetch_open_interest(self, symbol: str) -> dict:
        try:
            if self.mode == Mode.PAPER:
                public = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
                return public.fetch_open_interest(symbol)
            return self.exchange.fetch_open_interest(symbol)
        except Exception:
            return {}

    def get_top_symbols(self, n: int = 20, quote: str = "USDT") -> List[str]:
        """Lấy top N theo volume 24h"""
        if self.mode == Mode.PAPER:
            public = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
            tickers = public.fetch_tickers()
        else:
            tickers = self.exchange.fetch_tickers()

        usdt_pairs = []
        for sym, t in tickers.items():
            if sym.endswith(f"/{quote}") and t.get("quoteVolume"):
                usdt_pairs.append((sym, t["quoteVolume"]))

        usdt_pairs.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in usdt_pairs[:n]]

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------
    def get_balance(self) -> float:
        if self.mode == Mode.PAPER:
            return self.paper_balance
        bal = self.exchange.fetch_balance()
        return float(bal.get("USDT", {}).get("free", 0))

    def get_positions(self) -> List[Position]:
        if self.mode == Mode.PAPER:
            return list(self.paper_positions.values())
        # Live / Testnet
        positions = []
        raw = self.exchange.fetch_positions()
        for p in raw:
            if float(p.get("contracts", 0)) > 0:
                side = Side.LONG if p["side"] == "long" else Side.SHORT
                positions.append(Position(
                    symbol=p["symbol"],
                    side=side,
                    entry_price=float(p["entryPrice"]),
                    size=float(p["contracts"]),
                    sl=0.0,  # cần track riêng
                    tp=0.0,
                    order_id=str(p.get("id", "")),
                    strategy="unknown",
                    open_time=datetime.utcnow(),
                    leverage=int(p.get("leverage", 1)),
                    unrealized_pnl=float(p.get("unrealizedPnl", 0)),
                ))
        return positions

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------
    def create_order(
        self,
        symbol: str,
        side: str,          # "buy" / "sell"
        amount: float,
        price: Optional[float] = None,
        order_type: str = "market",
        params: dict = None,
    ) -> dict:
        params = params or {}
        if self.mode == Mode.PAPER:
            return self._paper_create_order(symbol, side, amount, price, order_type, params)

        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params,
            )
            logger.info(f"[{self.mode.value}] Order created: {symbol} {side} {amount}")
            return order
        except Exception as e:
            logger.error(f"Order failed: {e}")
            raise

    def _paper_create_order(self, symbol, side, amount, price, order_type, params):
        order_id = str(uuid.uuid4())[:12]
        ticker = self.fetch_ticker(symbol)
        fill_price = price or ticker["last"]

        # Simple slippage simulation
        slip = 0.0002
        if side == "buy":
            fill_price *= (1 + slip)
        else:
            fill_price *= (1 - slip)

        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": fill_price,
            "status": "closed",
            "timestamp": int(time.time() * 1000),
            "info": {"paper": True},
        }
        self.paper_orders[order_id] = order
        logger.info(f"[PAPER] Order filled: {symbol} {side} {amount} @ {fill_price:.6f}")
        return order

    def set_leverage(self, symbol: str, leverage: int):
        if self.mode == Mode.PAPER:
            return True
        try:
            self.exchange.set_leverage(leverage, symbol)
            return True
        except Exception as e:
            logger.warning(f"Set leverage failed: {e}")
            return False

    def cancel_all_orders(self, symbol: str = None):
        if self.mode == Mode.PAPER:
            self.paper_orders.clear()
            return
        try:
            if symbol:
                self.exchange.cancel_all_orders(symbol)
            else:
                # cancel all open
                for pos in self.get_positions():
                    self.exchange.cancel_all_orders(pos.symbol)
        except Exception as e:
            logger.error(f"Cancel all failed: {e}")

    def close_position(self, symbol: str, side: Side):
        """Market close position"""
        positions = self.get_positions()
        for p in positions:
            if p.symbol == symbol and p.side == side:
                close_side = "sell" if side == Side.LONG else "buy"
                return self.create_order(symbol, close_side, p.size, order_type="market")
        return None

    def flatten_all(self):
        """Kill switch: đóng hết position + hủy order"""
        logger.critical("FLATTEN ALL POSITIONS")
        self.cancel_all_orders()
        for p in self.get_positions():
            self.close_position(p.symbol, p.side)
        if self.mode == Mode.PAPER:
            self.paper_positions.clear()
