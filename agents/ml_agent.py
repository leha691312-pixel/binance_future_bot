"""ML Agent - sample collection + optional filter."""
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
        min_class_count: int = 20,
        fee_rate: float = 0.0004,
        min_r_multiple: float = 0.25,
        prob_threshold: float = 0.55,
        test_ratio: float = 0.25,
        **kwargs,
    ):
        self.model_path = Path(model_path)
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
        self._last_train_time = None
        self._load_pending()
        if self.enabled and self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                sp = self.model_path.with_suffix(".scaler.joblib")
                if sp.exists():
                    self.scaler = joblib.load(sp)
                logger.info("ML model loaded")
            except Exception as e:
                logger.warning(f"ML model load failed: {e}")

    def _pending_path(self):
        return self.samples_path.with_suffix(".pending.json")

    def _pending_meta_path(self):
        return self.samples_path.with_suffix(".pending_meta.json")

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
        feats = [
            conf, rr, is_long, 0.5,
            0.5, 0.01, 0.5, 1.0, 0.5,
            0.02, 0.02, 0.0,
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
            logger.info(f"ML on_open {key} feats_dim={len(feats)} | pending={len(self.pending_features)}")
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
        n = sum(1 for line in open(self.samples_path, encoding="utf-8") if line.strip())
        logger.info(
            f"ML sample | {key} label={'WIN' if label else 'LOSS'} "
            f"pnl={pnl:.4f} R={r_mult} | n={n}"
        )
        return None

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
                    s.confidence = max(float(s.confidence), proba)
                    out.append(s)
            except Exception:
                out.append(s)
        out.sort(key=lambda x: x.confidence, reverse=True)
        return out

    def status(self) -> dict:
        n = 0
        if self.samples_path.exists():
            n = sum(1 for line in open(self.samples_path, encoding="utf-8") if line.strip())
        return {
            "has_model": self.model is not None,
            "collect_only": self.collect_only,
            "filter_enabled": self.filter_enabled,
            "n_samples": n,
            "pending_open": len(self.pending_features),
            "samples_since_train": self._samples_since_train,
            "last_train": self._last_train_time.isoformat() if self._last_train_time else None,
            "min_samples": self.min_samples,
            "prob_threshold": self.prob_threshold,
            "last_metrics": self.last_metrics,
        }

    def maybe_retrain(self, force: bool = False):
        return None
