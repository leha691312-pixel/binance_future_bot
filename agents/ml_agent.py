"""
ML Agent nâng cao - Feature engineering + train thật + filter signals
"""
from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from core.models import Signal, Side

try:
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    HAS_ML = True
except ImportError:
    HAS_ML = False


class MLAgent:
    def __init__(self, model_path: str = "data/models/signal_filter.joblib"):
        self.model_path = Path(model_path)
        self.scaler_path = self.model_path.with_suffix(".scaler.joblib")
        self.model = None
        self.scaler = None
        self.enabled = HAS_ML
        if self.enabled:
            self._load()

    def _load(self):
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                if self.scaler_path.exists():
                    self.scaler = joblib.load(self.scaler_path)
                logger.info(f"ML model loaded from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load ML model: {e}")
                self.model = None
        else:
            logger.info("ML Agent: no model yet (pass-through until trained)")

    def extract_features(self, signal: Signal, df: Optional[pd.DataFrame] = None) -> np.ndarray:
        strategy_map = {
            "ORB": 0.1, "LiquiditySweep": 0.2, "OrderBlock": 0.3,
            "SupplyDemand": 0.4, "FVG": 0.5, "BOS_CHOCH": 0.6, "OI_Funding": 0.7,
        }
        feats = [
            signal.confidence,
            signal.risk_reward(),
            1.0 if signal.side == Side.LONG else 0.0,
            strategy_map.get(signal.strategy, 0.5),
        ]

        if df is not None and len(df) >= 20:
            last = df.iloc[-1]
            window = df.iloc[-20:]
            body = abs(last["close"] - last["open"])
            full = last["high"] - last["low"]
            body_ratio = body / full if full > 0 else 0.5

            atr = self._atr(df, 14).iloc[-1]
            atr_pct = float(atr / last["close"]) if last["close"] > 0 and not pd.isna(atr) else 0.01

            vol_ma = window["volume"].mean()
            volume_ratio = float(last["volume"] / vol_ma) if vol_ma > 0 else 1.0

            rsi = self._rsi(df["close"], 14).iloc[-1]
            if pd.isna(rsi):
                rsi = 50.0

            high_20 = window["high"].max()
            low_20 = window["low"].min()
            dist_from_high = (high_20 - last["close"]) / last["close"]
            dist_from_low = (last["close"] - low_20) / last["close"]

            sma_fast = df["close"].iloc[-8:].mean()
            sma_slow = df["close"].iloc[-20:].mean()
            trend_strength = (sma_fast - sma_slow) / last["close"]

            feats.extend([
                body_ratio, atr_pct, volume_ratio, float(rsi) / 100.0,
                float(dist_from_high), float(dist_from_low), float(trend_strength),
            ])
        else:
            feats.extend([0.5, 0.01, 1.0, 0.5, 0.02, 0.02, 0.0])

        return np.array(feats, dtype=np.float64)

    def filter(self, signals: List[Signal], dfs: dict = None, min_confidence: float = 0.55) -> List[Signal]:
        if not signals:
            return []

        filtered = [s for s in signals if s.confidence >= min_confidence and s.risk_reward() >= 1.5]
        if not filtered:
            return []

        if self.model is None:
            return filtered

        try:
            X = []
            for s in filtered:
                df = dfs.get(s.symbol) if dfs else None
                X.append(self.extract_features(s, df))
            X = np.array(X)

            if self.scaler is not None:
                X = self.scaler.transform(X)

            probs = self.model.predict_proba(X)[:, 1]
            ranked = sorted(zip(filtered, probs), key=lambda x: x[1], reverse=True)
            return [s for s, p in ranked if p >= 0.52]
        except Exception as e:
            logger.warning(f"ML filter error: {e}")
            return filtered

    def train_from_trades(self, trade_records: List[dict], min_samples: int = 50) -> dict:
        if not HAS_ML:
            return {"error": "sklearn not installed"}

        if len(trade_records) < min_samples:
            return {"error": f"Need at least {min_samples} samples, got {len(trade_records)}"}

        X, y = [], []
        for rec in trade_records:
            if "features" in rec:
                X.append(rec["features"])
                y.append(int(rec["label"]))

        X = np.array(X)
        y = np.array(y)

        if len(np.unique(y)) < 2:
            return {"error": "Need both win and loss samples"}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        model = HistGradientBoostingClassifier(
            max_iter=150, max_depth=4, learning_rate=0.08, random_state=42
        )
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = 0.0

        self.model = model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

        report = {
            "train_acc": round(train_acc, 4),
            "test_acc": round(test_acc, 4),
            "auc": round(auc, 4),
            "n_samples": len(y),
            "n_wins": int(y.sum()),
            "n_losses": int(len(y) - y.sum()),
            "saved_to": str(self.model_path),
        }
        logger.info(f"ML model trained: {report}")
        return report

    def generate_synthetic_training_data(self, n: int = 400) -> List[dict]:
        records = []
        rng = np.random.default_rng(42)
        for _ in range(n):
            conf = rng.uniform(0.55, 0.9)
            rr = rng.uniform(1.5, 3.5)
            is_long = float(rng.integers(0, 2))
            strat_id = float(rng.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
            body_ratio = rng.uniform(0.2, 0.9)
            atr_pct = rng.uniform(0.005, 0.04)
            vol_ratio = rng.uniform(0.6, 2.5)
            rsi = rng.uniform(0.25, 0.75)
            dist_h = rng.uniform(0, 0.05)
            dist_l = rng.uniform(0, 0.05)
            trend = rng.uniform(-0.03, 0.03)

            feats = [conf, rr, is_long, strat_id, body_ratio, atr_pct, vol_ratio, rsi, dist_h, dist_l, trend]
            score = conf * 0.4 + min(rr / 3, 1) * 0.3 + (1 - abs(rsi - 0.5)) * 0.2 + rng.normal(0, 0.15)
            label = 1 if score > 0.55 else 0
            records.append({"features": feats, "label": label})
        return records

    def train_demo(self) -> dict:
        records = self.generate_synthetic_training_data(400)
        return self.train_from_trades(records, min_samples=100)

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
