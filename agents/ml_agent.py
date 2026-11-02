"""
ML Agent - train hiệu quả hơn từ lệnh thật

Cải tiến:
- Feature giàu (ATR percentile, session, SL/TP dist, HTF proxy, funding)
- Label fee-aware + R-multiple (không chỉ pnl > 0)
- Time-based train/test split
- collect_only: chỉ gom sample, không filter bằng model
- Deploy gate: chỉ ghi model nếu AUC >= min và đủ win/loss
- Retrain khi +N mẫu hoặc sau X giờ
"""
from typing import List, Optional, Dict, Any, Tuple
import json
import math
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger
from core.models import Signal, Side

try:
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    from sklearn.preprocessing import StandardScaler
    HAS_ML = True
except ImportError:
    HAS_ML = False


FEATURE_DIM = 18  # cố định để tương thích train/filter


class MLAgent:
    def __init__(
        self,
        model_path: str = "data/models/signal_filter.joblib",
        samples_path: str = "data/models/train_samples.jsonl",
        min_samples: int = 80,
        retrain_every_new: int = 25,
        min_retrain_hours: float = 12.0,
        collect_only: bool = False,
        filter_enabled: bool = True,
        min_auc_deploy: float = 0.55,
        min_class_count: int = 20,
        fee_rate: float = 0.0004,
        min_r_multiple: float = 0.25,
        prob_threshold: float = 0.55,
        test_ratio: float = 0.25,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = self.model_path.with_suffix(".scaler.joblib")
        self.metrics_path = self.model_path.with_suffix(".metrics.json")
        self.samples_path = Path(samples_path)

        self.min_samples = int(min_samples)
        self.retrain_every_new = int(retrain_every_new)
        self.min_retrain_hours = float(min_retrain_hours)
        self.collect_only = bool(collect_only)
        self.filter_enabled = bool(filter_enabled) and not self.collect_only
        self.min_auc_deploy = float(min_auc_deploy)
        self.min_class_count = int(min_class_count)
        self.fee_rate = float(fee_rate)
        self.min_r_multiple = float(min_r_multiple)
        self.prob_threshold = float(prob_threshold)
        self.test_ratio = float(test_ratio)

        self.model = None
        self.scaler = None
        self.enabled = HAS_ML
        self.last_metrics: Dict[str, Any] = {}

        self.pending_features: Dict[str, List[float]] = {}
        self.pending_meta: Dict[str, dict] = {}  # risk_amount, notional, ...
        self._samples_since_train = 0
        self._last_train_time: Optional[datetime] = None

        if self.enabled:
            self._load()
            self._load_pending_meta()
            self._load_pending_into_memory()

    # ------------------------------------------------------------------
    # LOAD / SAVE
    # ------------------------------------------------------------------
    def _load(self):
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                if self.scaler_path.exists():
                    self.scaler = joblib.load(self.scaler_path)
                if self.metrics_path.exists():
                    self.last_metrics = json.loads(self.metrics_path.read_text(encoding="utf-8"))
                logger.info(f"ML model loaded | metrics={self.last_metrics}")
            except Exception as e:
                logger.warning(f"Failed to load ML model: {e}")
                self.model = None
        else:
            logger.info("ML Agent: no model yet (collecting / pass-through)")

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
                    "last_metrics": self.last_metrics,
                }, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug(f"Save ML meta failed: {e}")

    def _pending_path(self) -> Path:
        return self.samples_path.with_suffix(".pending.json")

    def _pending_meta_path(self) -> Path:
        return self.samples_path.with_suffix(".pending_meta.json")

    def _save_pending(self):
        try:
            path = self._pending_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.pending_features), encoding="utf-8")
            self._pending_meta_path().write_text(json.dumps(self.pending_meta), encoding="utf-8")
        except Exception:
            pass

    def _load_pending_into_memory(self):
        path = self._pending_path()
        if path.exists():
            try:
                self.pending_features = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self.pending_features = {}
        mp = self._pending_meta_path()
        if mp.exists():
            try:
                self.pending_meta = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                self.pending_meta = {}

    # ------------------------------------------------------------------
    # FEATURES (18 dims)
    # ------------------------------------------------------------------
    def extract_features(
        self,
        signal: Signal,
        df: Optional[pd.DataFrame] = None,
        context: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Vector cố định 18 chiều — chỉ dùng info tại thời điểm entry (no leakage).
        context: funding_rate, htf_trend (optional)
        """
        context = context or {}
        strategy_map = {
            "ORB": 0.1, "LiquiditySweep": 0.2, "OrderBlock": 0.3,
            "SupplyDemand": 0.4, "FVG": 0.5, "BOS_CHOCH": 0.6, "OI_Funding": 0.7,
            "TrendMA": 0.75, "MomentumRSI": 0.8, "PullbackEMA": 0.85, "RangeBounce": 0.9,
            "MARibbon": 0.15, "StochBB": 0.18, "PivotScalp": 0.22, "EMARSIScalp": 0.25,
            "SwingMACross": 0.92, "SwingRSI": 0.94, "FibPullback": 0.96, "BBSqueeze": 0.98,
        }

        conf = float(signal.confidence)
        rr = float(signal.risk_reward())
        is_long = 1.0 if signal.side == Side.LONG else 0.0
        strat_id = float(strategy_map.get(signal.strategy, 0.5))

        entry = float(signal.entry) or 1.0
        sl_dist = abs(entry - float(signal.sl)) / entry if entry else 0.01
        tp_dist = abs(float(signal.tp) - entry) / entry if entry else 0.02

        # session hour UTC sin/cos
        hour = datetime.utcnow().hour + datetime.utcnow().minute / 60.0
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)

        body_ratio = 0.5
        atr_pct = 0.01
        atr_pctile = 0.5
        volume_ratio = 1.0
        rsi_n = 0.5
        dist_from_high = 0.02
        dist_from_low = 0.02
        trend_strength = 0.0
        funding = float(context.get("funding_rate") or 0.0)
        htf_trend = float(context.get("htf_trend") or 0.0)

        if df is not None and len(df) >= 30:
            last = df.iloc[-1]
            window = df.iloc[-20:]
            body = abs(last["close"] - last["open"])
            full = last["high"] - last["low"]
            body_ratio = float(body / full) if full > 0 else 0.5

            atr_s = self._atr(df, 14)
            atr = atr_s.iloc[-1]
            atr_pct = float(atr / last["close"]) if last["close"] > 0 and not pd.isna(atr) else 0.01
            atr_hist = (atr_s / df["close"]).dropna().iloc[-50:]
            if len(atr_hist) >= 10:
                atr_pctile = float((atr_hist < atr_pct).mean())

            vol_ma = window["volume"].mean()
            volume_ratio = float(last["volume"] / vol_ma) if vol_ma > 0 else 1.0

            rsi = self._rsi(df["close"], 14).iloc[-1]
            rsi_n = float(rsi) / 100.0 if not pd.isna(rsi) else 0.5

            high_20 = float(window["high"].max())
            low_20 = float(window["low"].min())
            dist_from_high = (high_20 - float(last["close"])) / float(last["close"])
            dist_from_low = (float(last["close"]) - low_20) / float(last["close"])

            sma_fast = float(df["close"].iloc[-8:].mean())
            sma_slow = float(df["close"].iloc[-20:].mean())
            trend_strength = (sma_fast - sma_slow) / float(last["close"])

            # HTF proxy: SMA50 vs price nếu đủ nến
            if len(df) >= 50 and htf_trend == 0.0:
                sma50 = float(df["close"].iloc[-50:].mean())
                htf_trend = (float(last["close"]) - sma50) / float(last["close"])

        feats = [
            conf, rr, is_long, strat_id,
            body_ratio, atr_pct, atr_pctile, volume_ratio, rsi_n,
            dist_from_high, dist_from_low, trend_strength,
            sl_dist, tp_dist, hour_sin, hour_cos,
            funding, htf_trend,
        ]
        arr = np.array(feats, dtype=np.float64)
        if arr.shape[0] != FEATURE_DIM:
            # pad / trim an toàn
            out = np.zeros(FEATURE_DIM, dtype=np.float64)
            n = min(FEATURE_DIM, arr.shape[0])
            out[:n] = arr[:n]
            return out
        return arr

    # ------------------------------------------------------------------
    # LABEL
    # ------------------------------------------------------------------
    def make_label(
        self,
        pnl: float,
        risk_amount: float = 0.0,
        notional: float = 0.0,
        entry: float = 0.0,
        size: float = 0.0,
    ) -> Tuple[int, dict]:
        """
        WIN nếu:
        - pnl > estimated_fee
        - và (nếu có risk_amount) pnl/risk >= min_r_multiple
        """
        pnl = float(pnl)
        if notional <= 0 and entry > 0 and size > 0:
            notional = abs(entry * size)
        fee_est = abs(notional) * self.fee_rate * 2  # open+close approx

        r_mult = None
        if risk_amount and risk_amount > 0:
            r_mult = pnl / risk_amount

        win = pnl > fee_est
        if r_mult is not None:
            win = win and (r_mult >= self.min_r_multiple)
        elif abs(pnl) < 1e-12:
            # paper / unknown → không coi là win
            win = False

        info = {
            "fee_est": round(fee_est, 6),
            "r_multiple": round(r_mult, 4) if r_mult is not None else None,
            "min_r": self.min_r_multiple,
        }
        return (1 if win else 0), info

    # ------------------------------------------------------------------
    # OPEN / CLOSE
    # ------------------------------------------------------------------
    def on_open(
        self,
        signal: Signal,
        df: Optional[pd.DataFrame] = None,
        key: str = "",
        context: Optional[dict] = None,
        trade_meta: Optional[dict] = None,
    ):
        if not key:
            key = f"{signal.symbol}_{signal.side.value}"
        try:
            feats = self.extract_features(signal, df, context=context).tolist()
            self.pending_features[key] = feats
            self.pending_meta[key] = {
                "entry": float(signal.entry),
                "sl": float(signal.sl),
                "tp": float(signal.tp),
                "strategy": signal.strategy,
                "timeframe": signal.timeframe,
                **(trade_meta or {}),
            }
            self._save_pending()
            logger.debug(f"ML on_open {key} feats_dim={len(feats)}")
        except Exception as e:
            logger.warning(f"ML on_open failed {key}: {e}")

    def on_close(self, key: str, pnl: float, meta: Optional[dict] = None) -> Optional[dict]:
        feats = self.pending_features.pop(key, None)
        pmeta = self.pending_meta.pop(key, {}) or {}
        self._save_pending()

        if feats is None:
            logger.debug(f"ML on_close: no pending features for {key}")
            return None

        # merge meta
        m = {**pmeta, **(meta or {})}
        risk_amount = float(m.get("risk_amount") or 0)
        notional = float(m.get("notional") or 0)
        entry = float(m.get("entry") or 0)
        size = float(m.get("size") or 0)

        # ước lượng risk_amount nếu thiếu: |entry-sl|*size
        if risk_amount <= 0 and entry and size and m.get("sl") is not None:
            risk_amount = abs(entry - float(m["sl"])) * abs(size)

        label, label_info = self.make_label(
            pnl, risk_amount=risk_amount, notional=notional, entry=entry, size=size
        )

        # Chuẩn hóa chiều feature (mẫu cũ 11 vs mới 18)
        feats = self._normalize_feat_vec(feats)

        rec = {
            "features": feats,
            "label": label,
            "pnl": float(pnl),
            "key": key,
            "ts": datetime.utcnow().isoformat(),
            "label_info": label_info,
            "meta": {
                "strategy": m.get("strategy", ""),
                "timeframe": m.get("timeframe", ""),
                "trade_style": m.get("trade_style", ""),
                "risk_amount": risk_amount,
                "notional": notional,
            },
        }

        self._append_sample(rec)
        self._samples_since_train += 1
        self._save_meta()

        logger.info(
            f"ML sample | {key} label={'WIN' if label else 'LOSS'} "
            f"pnl={pnl:.4f} R={label_info.get('r_multiple')} | n={self.count_samples()}"
        )

        if self.collect_only:
            logger.debug("ML collect_only: skip retrain deploy")
            return None

        return self.maybe_retrain()

    def _normalize_feat_vec(self, feats: List[float]) -> List[float]:
        arr = list(feats)
        if len(arr) < FEATURE_DIM:
            arr = arr + [0.0] * (FEATURE_DIM - len(arr))
        elif len(arr) > FEATURE_DIM:
            arr = arr[:FEATURE_DIM]
        return arr

    def _append_sample(self, rec: dict):
        self.samples_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.samples_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def count_samples(self) -> int:
        if not self.samples_path.exists():
            return 0
        n = 0
        with open(self.samples_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    def load_all_samples(self) -> List[dict]:
        records = []
        if not self.samples_path.exists():
            return records
        with open(self.samples_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "features" in rec and "label" in rec:
                        rec["features"] = self._normalize_feat_vec(rec["features"])
                        records.append(rec)
                except Exception:
                    continue
        return records

    # ------------------------------------------------------------------
    # TRAIN (time split + deploy gate)
    # ------------------------------------------------------------------
    def maybe_retrain(self, force: bool = False) -> Optional[dict]:
        if not HAS_ML:
            return None

        n = self.count_samples()
        if n < self.min_samples:
            logger.debug(f"ML retrain skip: {n}/{self.min_samples}")
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
        if report.get("deployed"):
            self._samples_since_train = 0
            self._last_train_time = datetime.utcnow()
            self._save_meta()
            logger.info(f"ML deploy OK: {report}")
        else:
            logger.warning(f"ML train not deployed: {report}")
        return report

    def _time_split(
        self, records: List[dict]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict], List[dict]]:
        """Split theo thời gian: cũ = train, mới = test."""
        def ts_key(r):
            try:
                return datetime.fromisoformat(r.get("ts", "1970-01-01"))
            except Exception:
                return datetime(1970, 1, 1)

        ordered = sorted(records, key=ts_key)
        n = len(ordered)
        n_test = max(1, int(n * self.test_ratio))
        n_train = n - n_test
        if n_train < 10:
            n_train = max(1, n - 1)
            n_test = n - n_train

        train_recs = ordered[:n_train]
        test_recs = ordered[n_train:]

        X_train = np.array([r["features"] for r in train_recs], dtype=np.float64)
        y_train = np.array([int(r["label"]) for r in train_recs], dtype=np.int32)
        X_test = np.array([r["features"] for r in test_recs], dtype=np.float64)
        y_test = np.array([int(r["label"]) for r in test_recs], dtype=np.int32)
        return X_train, X_test, y_train, y_test, train_recs, test_recs

    def train_from_trades(self, trade_records: List[dict], min_samples: int = 50) -> dict:
        if not HAS_ML:
            return {"error": "sklearn not installed", "deployed": False}

        if len(trade_records) < min_samples:
            return {
                "error": f"Need at least {min_samples} samples, got {len(trade_records)}",
                "deployed": False,
            }

        y_all = np.array([int(r["label"]) for r in trade_records])
        n_wins = int(y_all.sum())
        n_losses = int(len(y_all) - n_wins)
        if n_wins < self.min_class_count or n_losses < self.min_class_count:
            return {
                "error": f"Need ≥{self.min_class_count} wins and losses (wins={n_wins}, losses={n_losses})",
                "deployed": False,
                "n_wins": n_wins,
                "n_losses": n_losses,
            }

        X_train, X_test, y_train, y_test, train_recs, test_recs = self._time_split(trade_records)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            return {"error": "Train or test set missing a class (time split)", "deployed": False}

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=3,
            learning_rate=0.05,
            min_samples_leaf=15,
            l2_regularization=0.1,
            random_state=42,
        )
        model.fit(X_train_s, y_train)

        train_acc = float(model.score(X_train_s, y_train))
        test_acc = float(model.score(X_test_s, y_test))
        y_prob = model.predict_proba(X_test_s)[:, 1]
        y_pred = (y_prob >= self.prob_threshold).astype(int)

        try:
            auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            auc = 0.0
        try:
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
        except Exception:
            prec, rec = 0.0, 0.0

        # Simulated filter PnL trên test set
        test_pnls = [float(r.get("pnl") or 0) for r in test_recs]
        baseline_pnl = float(np.sum(test_pnls))
        filtered_pnl = float(np.sum([p for p, pr in zip(test_pnls, y_prob) if pr >= self.prob_threshold]))
        n_kept = int(np.sum(y_prob >= self.prob_threshold))

        report = {
            "train_acc": round(train_acc, 4),
            "test_acc": round(test_acc, 4),
            "auc": round(auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "n_samples": len(trade_records),
            "n_train": len(train_recs),
            "n_test": len(test_recs),
            "n_wins": n_wins,
            "n_losses": n_losses,
            "baseline_test_pnl": round(baseline_pnl, 4),
            "filtered_test_pnl": round(filtered_pnl, 4),
            "n_kept_test": n_kept,
            "prob_threshold": self.prob_threshold,
            "deployed": False,
        }

        # Deploy gate
        better_pnl = filtered_pnl >= baseline_pnl * 0.95  # không tệ hơn baseline quá nhiều
        if auc >= self.min_auc_deploy and better_pnl:
            self.model = model
            self.scaler = scaler
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, self.model_path)
            joblib.dump(scaler, self.scaler_path)
            report["deployed"] = True
            report["saved_to"] = str(self.model_path)
            self.last_metrics = report
            self.metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            logger.info(f"ML model DEPLOYED: AUC={auc:.3f} filtered_pnl={filtered_pnl:.2f}")
        else:
            report["deploy_reason"] = (
                f"AUC {auc:.3f} < {self.min_auc_deploy} or filtered_pnl {filtered_pnl:.2f} << baseline {baseline_pnl:.2f}"
            )
            logger.warning(f"ML model NOT deployed: {report['deploy_reason']}")

        return report

    # ------------------------------------------------------------------
    # FILTER
    # ------------------------------------------------------------------
    def filter(
        self,
        signals: List[Signal],
        dfs: dict = None,
        min_confidence: float = 0.55,
        contexts: Optional[Dict[str, dict]] = None,
    ) -> List[Signal]:
        if not signals:
            return []

        filtered = [s for s in signals if s.confidence >= min_confidence and s.risk_reward() >= 1.5]
        if not filtered:
            return []

        # collect_only hoặc chưa bật filter hoặc chưa có model
        if self.collect_only or not self.filter_enabled or self.model is None:
            return filtered

        try:
            X = []
            for s in filtered:
                df = None
                if dfs:
                    df = dfs.get(s.symbol) or dfs.get(f"{s.symbol}_{s.timeframe}")
                ctx = (contexts or {}).get(s.symbol, {})
                X.append(self.extract_features(s, df, context=ctx))
            X = np.array(X, dtype=np.float64)
            if self.scaler is not None:
                X = self.scaler.transform(X)
            probs = self.model.predict_proba(X)[:, 1]
            ranked = sorted(zip(filtered, probs), key=lambda x: x[1], reverse=True)
            out = [s for s, p in ranked if p >= self.prob_threshold]
            logger.info(
                f"ML filter: {len(filtered)} → {len(out)} (threshold={self.prob_threshold})"
            )
            return out
        except Exception as e:
            logger.warning(f"ML filter error: {e}")
            return filtered

    def status(self) -> dict:
        return {
            "has_model": self.model is not None,
            "collect_only": self.collect_only,
            "filter_enabled": self.filter_enabled and self.model is not None,
            "n_samples": self.count_samples(),
            "pending_open": len(self.pending_features),
            "samples_since_train": self._samples_since_train,
            "last_train": self._last_train_time.isoformat() if self._last_train_time else None,
            "min_samples": self.min_samples,
            "min_auc_deploy": self.min_auc_deploy,
            "prob_threshold": self.prob_threshold,
            "last_metrics": self.last_metrics,
        }

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
