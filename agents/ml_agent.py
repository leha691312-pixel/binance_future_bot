"""ML Agent - collect samples from closed trades, train filter, optional inference."""
from typing import List, Optional, Dict, Any
import json
import math
from pathlib import Path
from datetime import datetime
from loguru import logger
from core.models import Signal, Side

try:
    import joblib
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    HAS_ML = True
except ImportError:
    HAS_ML = False
    np = None

FEATURE_DIM = 18


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
        min_class_count: int = 10,
        fee_rate: float = 0.0004,
        min_r_multiple: float = 0.25,
        prob_threshold: float = 0.55,
        test_ratio: float = 0.25,
        **kwargs,
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
        self.pending_meta: Dict[str, dict] = {}
        self._samples_since_train = 0
        self._last_train_time: Optional[datetime] = None

        self._load_pending()
        self._load_meta()
        if self.enabled and self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                if self.scaler_path.exists():
                    self.scaler = joblib.load(self.scaler_path)
                if self.metrics_path.exists():
                    self.last_metrics = json.loads(self.metrics_path.read_text(encoding="utf-8"))
                logger.info(f"ML model loaded | metrics={self.last_metrics}")
            except Exception as e:
                logger.warning(f"ML model load failed: {e}")
        else:
            logger.info("ML Agent: no model yet (collecting / pass-through)")

        # Auto-train on startup if enough samples and no model
        if self.enabled and self.model is None and not self.collect_only:
            n = self.count_samples()
            if n >= self.min_samples:
                logger.info(f"ML startup train: {n} samples >= {self.min_samples}")
                self.maybe_retrain(force=True)

    def _pending_path(self) -> Path:
        return self.samples_path.with_suffix(".pending.json")

    def _pending_meta_path(self) -> Path:
        return self.samples_path.with_suffix(".pending_meta.json")

    def _meta_path(self) -> Path:
        return self.samples_path.with_suffix(".meta.json")

    def _load_pending(self):
        try:
            p = self._pending_path()
            if p.exists():
                self.pending_features = json.loads(p.read_text(encoding="utf-8"))
            mp = self._pending_meta_path()
            if mp.exists():
                self.pending_meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            pass

    def _save_pending(self):
        try:
            self.samples_path.parent.mkdir(parents=True, exist_ok=True)
            self._pending_path().write_text(json.dumps(self.pending_features), encoding="utf-8")
            self._pending_meta_path().write_text(json.dumps(self.pending_meta), encoding="utf-8")
        except Exception:
            pass

    def _load_meta(self):
        try:
            mp = self._meta_path()
            if mp.exists():
                data = json.loads(mp.read_text(encoding="utf-8"))
                self._samples_since_train = int(data.get("samples_since_train", 0))
                ts = data.get("last_train_time")
                if ts:
                    self._last_train_time = datetime.fromisoformat(ts)
        except Exception:
            pass

    def _save_meta(self):
        try:
            self.samples_path.parent.mkdir(parents=True, exist_ok=True)
            self._meta_path().write_text(
                json.dumps(
                    {
                        "samples_since_train": self._samples_since_train,
                        "last_train_time": self._last_train_time.isoformat()
                        if self._last_train_time
                        else None,
                        "n_samples": self.count_samples(),
                        "last_metrics": self.last_metrics,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug(f"Save ML meta failed: {e}")

    def count_samples(self) -> int:
        if not self.samples_path.exists():
            return 0
        n = 0
        with open(self.samples_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    def extract_features(self, signal: Signal, df=None, context=None) -> list:
        context = context or {}
        conf = float(getattr(signal, "confidence", 0.5) or 0.5)
        try:
            rr = float(signal.risk_reward())
        except Exception:
            rr = 1.5
        is_long = 1.0 if signal.side == Side.LONG else 0.0
        entry = float(signal.entry or 1)
        sl = float(signal.sl or entry)
        tp = float(signal.tp or entry)
        sl_dist = abs(entry - sl) / entry if entry else 0.01
        tp_dist = abs(tp - entry) / entry if entry else 0.02
        hour = datetime.utcnow().hour + datetime.utcnow().minute / 60.0
        body_ratio = 0.5
        atr_pct = 0.01
        atr_pctile = 0.5
        volume_ratio = 1.0
        rsi_n = 0.5
        dist_from_high = 0.02
        dist_from_low = 0.02
        trend_strength = 0.0
        if df is not None and hasattr(df, "__len__") and len(df) >= 30:
            try:
                last = df.iloc[-1]
                window = df.iloc[-20:]
                body = abs(float(last["close"]) - float(last["open"]))
                full = float(last["high"]) - float(last["low"])
                body_ratio = float(body / full) if full > 0 else 0.5
                closes = df["close"].astype(float)
                highs = df["high"].astype(float)
                lows = df["low"].astype(float)
                tr = (highs - lows).iloc[-14:]
                atr = float(tr.mean()) if len(tr) else 0.0
                atr_pct = atr / float(last["close"]) if float(last["close"]) > 0 else 0.01
                vol = df["volume"].astype(float).iloc[-20:]
                volume_ratio = float(vol.iloc[-1] / vol.mean()) if vol.mean() > 0 else 1.0
                delta = closes.diff()
                gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
                loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
                if loss and loss > 0:
                    rsi_n = float(100 - (100 / (1 + gain / loss))) / 100.0
                high_20 = float(window["high"].max())
                low_20 = float(window["low"].min())
                dist_from_high = (high_20 - float(last["close"])) / float(last["close"])
                dist_from_low = (float(last["close"]) - low_20) / float(last["close"])
                sma_f = float(closes.iloc[-8:].mean())
                sma_s = float(closes.iloc[-20:].mean())
                trend_strength = (sma_f - sma_s) / float(last["close"])
            except Exception:
                pass
        feats = [
            conf, rr, is_long, 0.5,
            body_ratio, atr_pct, atr_pctile, volume_ratio, rsi_n,
            dist_from_high, dist_from_low, trend_strength,
            sl_dist, tp_dist,
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            float(context.get("funding_rate") or 0),
            float(context.get("htf_trend") or 0),
        ]
        while len(feats) < FEATURE_DIM:
            feats.append(0.0)
        return feats[:FEATURE_DIM]

    def on_open(self, signal: Signal, df=None, key: str = "", context=None, trade_meta=None):
        if not key:
            key = f"{signal.symbol}_{signal.side.value}"
        try:
            feats = self.extract_features(signal, df, context=context)
            self.pending_features[key] = feats
            self.pending_meta[key] = {
                "entry": float(signal.entry),
                "sl": float(signal.sl),
                "tp": float(signal.tp),
                "strategy": getattr(signal, "strategy", ""),
                "timeframe": getattr(signal, "timeframe", ""),
                **(trade_meta or {}),
            }
            self._save_pending()
            logger.info(
                f"ML on_open {key} feats_dim={len(feats)} | pending={len(self.pending_features)}"
            )
        except Exception as e:
            logger.warning(f"ML on_open failed {key}: {e}")

    def on_close(self, key: str, pnl: float, meta: Optional[dict] = None) -> Optional[dict]:
        feats = self.pending_features.pop(key, None)
        pmeta = self.pending_meta.pop(key, {}) or {}
        if feats is None and key:
            for ak in (
                key.replace("_LONG", "_long"),
                key.replace("_SHORT", "_short"),
                key.replace("_long", "_LONG"),
                key.replace("_short", "_SHORT"),
            ):
                if ak in self.pending_features:
                    feats = self.pending_features.pop(ak)
                    pmeta = self.pending_meta.pop(ak, {}) or {}
                    key = ak
                    break
        self._save_pending()
        if feats is None:
            logger.warning(
                f"ML on_close: no pending features for {key} | "
                f"pending_keys={list(self.pending_features.keys())[:8]}"
            )
            return None

        m = {**pmeta, **(meta or {})}
        risk_amount = float(m.get("risk_amount") or 0)
        notional = float(m.get("notional") or 0)
        entry = float(m.get("entry") or 0)
        size = float(m.get("size") or 0)
        if risk_amount <= 0 and entry and size and m.get("sl") is not None:
            risk_amount = abs(entry - float(m["sl"])) * abs(size)

        fee_est = abs(notional) * self.fee_rate * 2
        r_mult = (pnl / risk_amount) if risk_amount > 0 else None
        win = pnl > fee_est
        if r_mult is not None:
            win = win and (r_mult >= self.min_r_multiple)
        elif abs(pnl) < 1e-12:
            win = False
        label = 1 if win else 0

        feats_list = list(feats)[:FEATURE_DIM] + [0.0] * max(0, FEATURE_DIM - len(feats))
        rec = {
            "features": feats_list,
            "label": label,
            "pnl": float(pnl),
            "key": key,
            "ts": datetime.utcnow().isoformat(),
            "label_info": {
                "fee_est": fee_est,
                "r_multiple": r_mult,
                "min_r": self.min_r_multiple,
            },
            "meta": {
                "strategy": m.get("strategy", ""),
                "timeframe": m.get("timeframe", ""),
                "trade_style": m.get("trade_style", ""),
                "risk_amount": risk_amount,
                "notional": notional,
            },
        }
        self.samples_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.samples_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._samples_since_train += 1
        self._save_meta()
        n = self.count_samples()
        logger.info(
            f"ML sample | {key} label={'WIN' if label else 'LOSS'} "
            f"pnl={pnl:.4f} R={r_mult} | n={n}"
        )

        if self.collect_only:
            logger.debug("ML collect_only: skip retrain")
            return None
        return self.maybe_retrain()

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
                        feats = list(rec["features"])
                        if len(feats) < FEATURE_DIM:
                            feats = feats + [0.0] * (FEATURE_DIM - len(feats))
                        rec["features"] = feats[:FEATURE_DIM]
                        records.append(rec)
                except Exception:
                    continue
        return records

    def maybe_retrain(self, force: bool = False) -> Optional[dict]:
        if not HAS_ML:
            logger.warning("ML retrain skip: sklearn not installed")
            return None

        n = self.count_samples()
        if n < self.min_samples:
            logger.debug(f"ML retrain skip: {n}/{self.min_samples}")
            return None

        if not force:
            if self.model is None:
                force = True
            else:
                if self._samples_since_train < self.retrain_every_new:
                    return None
                if self._last_train_time is not None:
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

    def train_from_trades(self, trade_records: List[dict], min_samples: int = 50) -> dict:
        if not HAS_ML:
            return {"error": "sklearn not installed", "deployed": False}

        if len(trade_records) < min_samples:
            return {
                "error": f"Need at least {min_samples} samples, got {len(trade_records)}",
                "deployed": False,
            }

        def ts_key(r):
            try:
                return datetime.fromisoformat(r.get("ts", "1970-01-01"))
            except Exception:
                return datetime(1970, 1, 1)

        ordered = sorted(trade_records, key=ts_key)
        n = len(ordered)
        n_test = max(1, int(n * self.test_ratio))
        n_train = max(1, n - n_test)
        train_recs = ordered[:n_train]
        test_recs = ordered[n_train:]

        X_train = np.array([r["features"] for r in train_recs], dtype=np.float64)
        y_train = np.array([int(r["label"]) for r in train_recs], dtype=np.int32)
        X_test = np.array([r["features"] for r in test_recs], dtype=np.float64)
        y_test = np.array([int(r["label"]) for r in test_recs], dtype=np.int32)

        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        if n_pos < self.min_class_count or n_neg < self.min_class_count:
            return {
                "error": f"Need >= {self.min_class_count} per class (win={n_pos} loss={n_neg})",
                "deployed": False,
                "n_pos": n_pos,
                "n_neg": n_neg,
            }

        try:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = HistGradientBoostingClassifier(
                max_depth=4,
                learning_rate=0.08,
                max_iter=120,
                min_samples_leaf=10,
                random_state=42,
            )
            model.fit(X_train_s, y_train)

            proba = model.predict_proba(X_test_s)[:, 1]
            try:
                auc = float(roc_auc_score(y_test, proba)) if len(set(y_test.tolist())) > 1 else 0.5
            except Exception:
                auc = 0.5

            report = {
                "n_samples": n,
                "n_train": n_train,
                "n_test": len(test_recs),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "auc": round(auc, 4),
                "deployed": False,
            }

            if auc < self.min_auc_deploy:
                report["deploy_reason"] = f"AUC {auc:.3f} < min {self.min_auc_deploy}"
                self.last_metrics = report
                return report

            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, self.model_path)
            joblib.dump(scaler, self.scaler_path)
            self.model = model
            self.scaler = scaler
            report["deployed"] = True
            report["deploy_reason"] = "ok"
            self.last_metrics = report
            self.metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            logger.info(f"ML model DEPLOYED: AUC={auc:.3f} n={n} pos={n_pos} neg={n_neg}")
            return report
        except Exception as e:
            logger.error(f"ML train failed: {e}")
            return {"error": str(e), "deployed": False}

    def filter(self, signals: List[Signal], dfs=None) -> List[Signal]:
        if self.collect_only or not self.filter_enabled or self.model is None:
            return list(signals)
        out = []
        for s in signals:
            try:
                feats = self.extract_features(s)
                X = [feats]
                if self.scaler is not None:
                    X = self.scaler.transform(X)
                proba = float(self.model.predict_proba(X)[0][1])
                if proba >= self.prob_threshold:
                    s.confidence = max(float(getattr(s, "confidence", 0) or 0), proba)
                    out.append(s)
            except Exception:
                out.append(s)
        out.sort(key=lambda x: getattr(x, "confidence", 0), reverse=True)
        return out

    def status(self) -> dict:
        return {
            "has_model": self.model is not None,
            "collect_only": self.collect_only,
            "filter_enabled": self.filter_enabled,
            "n_samples": self.count_samples(),
            "pending_open": len(self.pending_features),
            "samples_since_train": self._samples_since_train,
            "last_train": self._last_train_time.isoformat() if self._last_train_time else None,
            "min_samples": self.min_samples,
            "prob_threshold": self.prob_threshold,
            "last_metrics": self.last_metrics,
        }
