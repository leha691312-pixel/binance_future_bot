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
            # Binance đã bỏ sandbox/testnet cũ cho Futures.
            # Dùng Demo Trading mới: enable_demo_trading(True)
            # API key phải tạo tại https://demo.binance.com (Demo Trading API)
            self.exchange = ccxt.binanceusdm({
                "apiKey": os.getenv("BINANCE_TESTNET_API_KEY", "") or os.getenv("BINANCE_DEMO_API_KEY", ""),
                "secret": os.getenv("BINANCE_TESTNET_API_SECRET", "") or os.getenv("BINANCE_DEMO_API_SECRET", ""),
                "options": options,
                "enableRateLimit": True,
            })
            if hasattr(self.exchange, "enable_demo_trading"):
                self.exchange.enable_demo_trading(True)
            else:
                # Fallback cho bản ccxt cũ hơn
                try:
                    self.exchange.set_sandbox_mode(True)
                except Exception as e:
                    logger.error(f"Demo/Sandbox not supported: {e}")
                    raise
            logger.info("Exchange mode: DEMO TRADING (demo.binance.com)")
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
        """Lấy top N theo volume 24h.
        Symbol Futures CCXT dạng BTC/USDT:USDT (không phải BTC/USDT).
        """
        tickers = {}
        try:
            # Luôn ưu tiên public endpoint để lấy volume (ổn định hơn demo)
            public = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
            tickers = public.fetch_tickers()
        except Exception as e:
            logger.warning(f"Public fetch_tickers failed: {e}")
            try:
                if self.exchange:
                    tickers = self.exchange.fetch_tickers()
            except Exception as e2:
                logger.warning(f"Exchange fetch_tickers failed: {e2}")

        usdt_pairs = []
        for sym, t in tickers.items():
            # Hỗ trợ cả BTC/USDT và BTC/USDT:USDT
            is_usdt = (
                sym.endswith(f"/{quote}")
                or sym.endswith(f"/{quote}:{quote}")
                or f"/{quote}:" in sym
            )
            # Bỏ qua quarterlies / delivery nếu muốn chỉ perpetual
            if not is_usdt:
                continue
            vol = t.get("quoteVolume") or t.get("baseVolume") or 0
            if vol and float(vol) > 0:
                usdt_pairs.append((sym, float(vol)))

        usdt_pairs.sort(key=lambda x: x[1], reverse=True)
        result = [s[0] for s in usdt_pairs[:n]]

        # Fallback cứng nếu vẫn rỗng (mạng lỗi / demo hạn chế)
        if not result:
            logger.warning("Top symbols empty → using default major pairs")
            result = [
                "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT",
                "SOL/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT",
                "ADA/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT",
                "DOT/USDT:USDT",
            ][:n]
        return result

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
                lev = p.get("leverage") or 1
                try:
                    lev = int(float(lev))
                except (TypeError, ValueError):
                    lev = 1
                ep = p.get("entryPrice") or 0
                upnl = p.get("unrealizedPnl") or 0
                positions.append(Position(
                    symbol=p["symbol"],
                    side=side,
                    entry_price=float(ep) if ep else 0.0,
                    size=float(p.get("contracts") or 0),
                    sl=0.0,
                    tp=0.0,
                    order_id=str(p.get("id") or ""),
                    strategy="unknown",
                    open_time=datetime.utcnow(),
                    leverage=lev,
                    unrealized_pnl=float(upnl) if upnl else 0.0,
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
        ot = (order_type or "market").upper()
        params = params or {}

        # Protective orders (SL/TP) → giữ status open, không fill ngay
        if ot in ("STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP", "TAKE_PROFIT"):
            stop_price = params.get("stopPrice") or price
            order = {
                "id": order_id,
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": stop_price,
                "stopPrice": stop_price,
                "type": ot,
                "status": "open",
                "timestamp": int(time.time() * 1000),
                "info": {"paper": True, "reduceOnly": params.get("reduceOnly", True)},
            }
            self.paper_orders[order_id] = order
            logger.info(f"[PAPER] {ot} placed: {symbol} {side} {amount} @ stop={stop_price}")
            return order

        # Market / limit entry → fill ngay
        ticker = self.fetch_ticker(symbol)
        fill_price = price or ticker["last"]
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
            "type": ot,
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
