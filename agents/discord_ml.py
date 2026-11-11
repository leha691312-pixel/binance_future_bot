"""ML filter toggles for Discord control."""
from __future__ import annotations

from typing import Any, Callable

_get_state = None
_save_settings = None


def bind(get_state: Callable, save_settings_yaml: Callable) -> None:
    global _get_state, _save_settings
    _get_state = get_state
    _save_settings = save_settings_yaml


def parse_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on", "enable", "enabled"):
        return True
    if v in ("0", "false", "no", "off", "disable", "disabled"):
        return False
    raise ValueError(f"Bool invalid: {value}")


def apply_ml_runtime() -> str:
    st = _get_state()
    ml = getattr(st, "ml_agent", None)
    if ml is None:
        return "ML agent not ready (after restart)."
    collect = bool(st.settings.get("ml_collect_only", False))
    filt = bool(st.settings.get("ml_filter_enabled", True))
    thr = float(st.settings.get("ml_prob_threshold", getattr(ml, "prob_threshold", 0.55)))
    ml.collect_only = collect
    ml.filter_enabled = bool(filt) and (not collect)
    ml.prob_threshold = thr
    has = ml.model is not None
    active = (not collect) and filt and has
    return (
        f"collect_only={collect} filter_enabled={filt} thr={thr:.2f} "
        f"has_model={has} -> filter **{'ON' if active else 'OFF'}**"
    )


def set_ml_filter(enabled: bool) -> str:
    st = _get_state()
    st.settings["ml_filter_enabled"] = bool(enabled)
    if enabled:
        st.settings["ml_collect_only"] = False
    _save_settings(st.settings)
    return f"ml_filter_enabled={enabled}\n" + apply_ml_runtime()


def set_ml_collect(enabled: bool) -> str:
    st = _get_state()
    st.settings["ml_collect_only"] = bool(enabled)
    if enabled:
        st.settings["ml_filter_enabled"] = False
    _save_settings(st.settings)
    return f"ml_collect_only={enabled}\n" + apply_ml_runtime()


def set_ml_threshold(value: str) -> str:
    st = _get_state()
    thr = float(value)
    if thr < 0.3 or thr > 0.95:
        return "threshold must be 0.30-0.95"
    st.settings["ml_prob_threshold"] = thr
    _save_settings(st.settings)
    return f"ml_prob_threshold={thr}\n" + apply_ml_runtime()


def status_text() -> str:
    st = _get_state()
    lines = [
        f"ml_filter_enabled = `{st.settings.get('ml_filter_enabled')}`",
        f"ml_collect_only = `{st.settings.get('ml_collect_only')}`",
        f"ml_prob_threshold = `{st.settings.get('ml_prob_threshold', 0.55)}`",
        apply_ml_runtime(),
    ]
    ml = getattr(st, "ml_agent", None)
    if ml and hasattr(ml, "status"):
        lines.append(f"```{ml.status()}```")
    return "\n".join(lines)
