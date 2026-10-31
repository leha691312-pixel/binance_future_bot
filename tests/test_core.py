"""
Unit tests cơ bản
Chạy: pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from core.models import Signal, Side, TradeResult
from agents.risk_gate import RiskGate
from agents.strategies.orb import ORBStrategy
from agents.strategies.fvg import FVGStrategy
from agents.strategies.bos_choch import BOSCHOCHStrategy
from agents.strategies.liquidity_sweep import LiquiditySweepStrategy
from agents.ml_agent import MLAgent


def make_ohlcv(n=100, trend=0.001):
    np.random.seed(42)
    close = 100 * np.cumprod(1 + np.random.randn(n) * 0.01 + trend)
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.random.uniform(1000, 5000, n)
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
    })
    return df


class TestModels:
    def test_signal_rr(self):
        s = Signal("BTC/USDT", Side.LONG, 100, 95, 110, 0.7, "ORB", "test", "15m")
        assert abs(s.risk_reward() - 2.0) < 1e-6

    def test_signal_to_dict(self):
        s = Signal("ETH/USDT", Side.SHORT, 2000, 2050, 1900, 0.65, "FVG", "reason", "1h")
        d = s.to_dict()
        assert d["side"] == "SHORT"
        assert "rr" in d


class TestRiskGate:
    def test_approve_good_signal(self):
        rg = RiskGate(equity=10000)
        s = Signal("BTC/USDT", Side.LONG, 100, 95, 110, 0.7, "ORB", "test", "15m")
        ok, reason = rg.check(s)
        assert ok is True
        assert reason == "APPROVED"

    def test_reject_low_rr(self):
        rg = RiskGate(equity=10000)
        s = Signal("BTC/USDT", Side.LONG, 100, 99, 101, 0.7, "ORB", "test", "15m")
        ok, reason = rg.check(s)
        assert ok is False
        assert "RR_TOO_LOW" in reason

    def test_kill_switch(self):
        rg = RiskGate(equity=10000)
        rg.activate_kill_switch("TEST")
        s = Signal("BTC/USDT", Side.LONG, 100, 95, 110, 0.7, "ORB", "test", "15m")
        ok, reason = rg.check(s)
        assert ok is False
        assert "KILL_SWITCH" in reason

    def test_duplicate_window(self):
        rg = RiskGate(equity=10000)
        s = Signal("BTC/USDT", Side.LONG, 100, 95, 110, 0.7, "ORB", "test", "15m")
        rg.register_order(s, "oid1", 1.0)
        ok, reason = rg.check(s)
        assert ok is False
        assert "DUPLICATE" in reason


class TestStrategies:
    def test_orb_runs(self):
        df = make_ohlcv(80)
        strat = ORBStrategy()
        signals = strat.generate(df, "BTC/USDT", "15m")
        assert isinstance(signals, list)

    def test_fvg_runs(self):
        df = make_ohlcv(80)
        strat = FVGStrategy()
        signals = strat.generate(df, "BTC/USDT", "15m")
        assert isinstance(signals, list)

    def test_bos_choch_runs(self):
        df = make_ohlcv(100)
        strat = BOSCHOCHStrategy()
        signals = strat.generate(df, "BTC/USDT", "15m")
        assert isinstance(signals, list)

    def test_liquidity_runs(self):
        df = make_ohlcv(60)
        strat = LiquiditySweepStrategy()
        signals = strat.generate(df, "ETH/USDT", "1h")
        assert isinstance(signals, list)


class TestMLAgent:
    def test_extract_features(self):
        ml = MLAgent()
        s = Signal("BTC/USDT", Side.LONG, 100, 95, 110, 0.7, "FVG", "test", "15m")
        df = make_ohlcv(50)
        feats = ml.extract_features(s, df)
        assert len(feats) == 11
        assert feats[0] == 0.7

    def test_train_demo(self):
        if not hasattr(MLAgent, "train_demo"):
            pytest.skip("ML not available")
        ml = MLAgent(model_path="data/models/test_model.joblib")
        report = ml.train_demo()
        if "error" in report:
            pytest.skip(report["error"])
        assert report["test_acc"] > 0.4
        assert ml.model is not None

    def test_filter_passthrough(self):
        ml = MLAgent()
        ml.model = None  # force pass-through
        s = Signal("BTC/USDT", Side.LONG, 100, 95, 110, 0.7, "ORB", "test", "15m")
        out = ml.filter([s])
        assert len(out) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
