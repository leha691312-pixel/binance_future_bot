"""
ML Agent - Feature engineering + train từ lệnh thật + filter signals

Pipeline:
1. Mở lệnh → lưu features theo key symbol_side
2. Đóng lệnh → label = 1 nếu PnL > 0, append vào dataset
3. Đủ ≥ min_samples → train_from_trades → ghi model
"""
from typing import List, Optional, Dict, Any
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
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
    def __init__(
        self,
        model_path: str = "data/models/signal_filter.joblib",
        samples_path: str = "data/models/train_samples.jsonl",
        min_samples: int = 50,
        retrain_every_new: int = 10,
        min_retrain_hours: float = 6.0,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = self.model_path.with_suffix(".scaler.joblib")
        self.samples_path = Path(samples_path)
        self.min_samples = int(min_samples)
        self.retrain_every_new = int(retrain_every_new)
        self.min_retrain_hours = float(min_retrain_hours)

        self.model = None
        self.scaler = None
        self.enabled = HAS_ML

        # Features lúc mở lệnh: key = "BTC/USDT:USDT_LONG"
        self.pending_features: Dict[str, List[float]] = {}
        self._samples_since_train = 0
        self._last_train_time: Optional[datetime] = None

        if self.enabled:
            self._load()
            self._load_pending_meta()

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
            logger.info("ML Agent: no model yet (pass-through until trained on real trades)")

    def _load_pending_meta(self):
        meta = self.samples_path.with_suffix(".meta.json")
        if meta.exists():
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                self._samples_since_train = int(data.get("samples_since_train", 0))
                ts = data.get("last_train_time")
                if ts:
                    self._last_train_time = datetime.fromisoformat(ts)
            except Exception:
                pass

    def _save_meta(self):
        meta = self.samples_path.with_suffix(".meta.json")
        try:
            meta.parent.mkdir(parents=True, exist_ok=True)
            meta.write_text(
                json.dumps({
                    "samples_since_train": self._samples_since_train,
                    "last_train_time": self._last_train_time.isoformat() if self._last_train_time else None,
                    "n_samples": self.count_samples(),
                }, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug(f"Save ML meta failed: {e}")

    # ------------------------------------------------------------------
    # FEATURES
    # ------------------------------------------------------------------
    def extract_features(self, signal: Signal, df: Optional[pd.DataFrame] = None) -> np.ndarray:
        strategy_map = {
            "ORB": 0.1, "LiquiditySweep": 0.2, "OrderBlock": 0.3,
            "SupplyDemand": 0.4, "FVG": 0.5, "BOS_CHOCH": 0.6, "OI_Funding": 0.7,
            "TrendMA": 0.75, "MomentumRSI": 0.8, "PullbackEMA": 0.85, "RangeBounce": 0.9,
            "MARibbon": 0.15, "StochBB": 0.18, "PivotScalp": 0.22, "EMARSIScalp": 0.25,
            "SwingMACross": 0.92, "SwingRSI": 0.94, "FibPullback": 0.96, "BBSqueeze": 0.98,
        }
        feats = [
            float(signal.confidence),
            float(signal.risk_reward()),
            1.0 if signal.side == Side.LONG else 0.0,
            float(strategy_map.get(signal.strategy, 0.5)),
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
                float(body_ratio), float(atr_pct), float(volume_ratio), float(rsi) / 100.0,
                float(dist_from_high), float(dist_from_low), float(trend_strength),
            ])
        else:
            feats.extend([0.5, 0.01, 1.0, 0.5, 0.02, 0.02, 0.0])

        return np.array(feats, dtype=np.float64)

    # ------------------------------------------------------------------
    # OPEN / CLOSE PIPELINE
    # ------------------------------------------------------------------
    def on_open(self, signal: Signal, df: Optional[pd.DataFrame] = None, key: str = ""):
        """Lưu feature lúc mở lệnh."""
        if not key:
            key = f"{signal.symbol}_{signal.side.value}"
        try:
            feats = self.extract_features(signal, df).tolist()
            self.pending_features[key] = feats
            # Persist pending để bot restart không mất
            self._save_pending()
            logger.debug(f"ML on_open stored features for {key}")
        except Exception as e:
            logger.warning(f"ML on_open failed {key}: {e}")

    def on_close(self, key: str, pnl: float, meta: Optional[dict] = None) -> Optional[dict]:
        """
        Khi đóng lệnh: label = 1 nếu pnl > 0, append sample, có thể retrain.
        Trả về report train nếu vừa train, else None.
        """
        feats = self.pending_features.pop(key, None)
        self._save_pending()

        if feats is None:
            logger.debug(f"ML on_close: no pending features for {key} (skip sample)")
            return None

        label = 1 if float(pnl) > 0 else 0
        rec = {
            "features": feats,
            "label": label,
            "pnl": float(pnl),
            "key": key,
            "ts": datetime.utcnow().isoformat(),
        }
        if meta:
            rec["meta"] = meta

        self._append_sample(rec)
        self._samples_since_train += 1
        self._save_meta()

        logger.info(
            f"ML sample saved | {key} label={'WIN' if label else 'LOSS'} "
            f"pnl={pnl:.4f} | total_samples={self.count_samples()}"
        )

        return self.maybe_retrain()

    def _pending_path(self) -> Path:
        return self.samples_path.with_suffix(".pending.json")

    def _save_pending(self):
        try:
            path = self._pending_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.pending_features), encoding="utf-8")
        except Exception:
            pass

    def _load_pending_into_memory(self):
        path = self._pending_path()
        if path.exists():
            try:
                self.pending_features = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self.pending_features = {}

    def _append_sample(self, rec: dict):
        self.samples_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.samples_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def count_samples(self) -> int:
        if not self.samples_path.exists():
            return 0
        n = 0
        try:
            with open(self.samples_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        n += 1
        except Exception:
            pass
        return n

    def load_all_samples(self) -> List[dict]:
        records = []
        if not self.samples_path.exists():
            return records
        try:
            with open(self.samples_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if "features" in rec and "label" in rec:
                        records.append(rec)
        except Exception as e:
            logger.warning(f"Load samples failed: {e}")
        return records

    def maybe_retrain(self, force: bool = False) -> Optional[dict]:
        """Train khi đủ mẫu và (đủ mẫu mới / đủ thời gian / force)."""
        if not HAS_ML:
            return None

        n = self.count_samples()
        if n < self.min_samples:
            logger.debug(f"ML retrain skip: {n}/{self.min_samples} samples")
            return None

        if not force:
            if self._samples_since_train < self.retrain_every_new and self.model is not None:
                return None
            if self._last_train_time and self.model is not None:
                hours = (datetime.utcnow() - self._last_train_time).total_seconds() / 3600
                if hours < self.min_retrain_hours and self._samples_since_train < self.retrain_every_new * 2:
                    return None

        records = self.load_all_samples()
        report = self.train_from_trades(records, min_samples=self.min_samples)
        if "error" not in report:
            self._samples_since_train = 0
            self._last_train_time = datetime.utcnow()
            self._save_meta()
            logger.info(f"ML auto-retrain done: {report}")
            return report
        logger.warning(f"ML retrain failed: {report}")
        return report

    # ------------------------------------------------------------------
    # FILTER / TRAIN
    # ------------------------------------------------------------------
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
                df = None
                if dfs:
                    df = dfs.get(s.symbol) or dfs.get(f"{s.symbol}_{s.timeframe}")
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
            if "features" in rec and "label" in rec:
                X.append(rec["features"])
                y.append(int(rec["label"]))

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.int32)

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
            "n_samples": int(len(y)),
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
            strat_id = float(rng.uniform(0.1, 0.98))
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

    def status(self) -> dict:
        return {
            "has_model": self.model is not None,
            "n_samples": self.count_samples(),
            "pending_open": len(self.pending_features),
            "samples_since_train": self._samples_since_train,
            "last_train": self._last_train_time.isoformat() if self._last_train_time else None,
            "min_samples": self.min_samples,
        }

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
