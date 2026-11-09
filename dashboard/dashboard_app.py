"""
Streamlit Dashboard - đọc dữ liệu thật từ bot
- data/status.json  (bot ghi mỗi cycle)
- data/orders.xlsx  (lịch sử lệnh)
Chạy: streamlit run dashboard/app.py
Bot phải chạy song song: python main.py
"""
import streamlit as st
import yaml
from pathlib import Path
import pandas as pd
from datetime import datetime
import time
import json

st.set_page_config(page_title="Binance Futures Bot", layout="wide", page_icon="📈")

ROOT = Path(__file__).parent.parent
STATUS_PATH = ROOT / "data" / "status.json"
ORDERS_XLSX = ROOT / "data" / "orders.xlsx"
ORDERS_CSV = ROOT / "data" / "orders.csv"
CFG = ROOT / "config"

with open(CFG / "settings.yaml") as f:
    settings = yaml.safe_load(f)
with open(CFG / "risk.yaml") as f:
    risk = yaml.safe_load(f)


def load_status():
    if not STATUS_PATH.exists():
        return {}
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_orders(n=50):
    try:
        if ORDERS_XLSX.exists():
            df = pd.read_excel(ORDERS_XLSX)
            return df.tail(n).iloc[::-1]  # mới nhất trên
        if ORDERS_CSV.exists():
            df = pd.read_csv(ORDERS_CSV)
            return df.tail(n).iloc[::-1]
    except Exception as e:
        st.warning(f"Không đọc được orders: {e}")
    return pd.DataFrame()


st.title("🤖 Binance Futures Multi-Agent Bot")
st.caption("Dữ liệu thật từ bot · status.json + orders.xlsx")

# Sidebar
st.sidebar.header("⚙️ Control")
auto_refresh = st.sidebar.checkbox("Auto-refresh (10s)", value=True)
st.sidebar.markdown("---")
st.sidebar.subheader("Config (read-only)")
st.sidebar.write(f"Mode file: **{settings.get('mode')}**")
st.sidebar.write(f"Top N: **{settings.get('top_n')}**")
st.sidebar.write(f"Scan interval: **{settings.get('scan_interval_seconds')}s**")

status = load_status()
bot_running = bool(status.get("updated_at"))

if not bot_running:
    st.error(
        "⚠️ Chưa có dữ liệu từ bot. Hãy chạy `python main.py` trong terminal khác, "
        "đợi 1 cycle xong rồi refresh trang này."
    )
else:
    age = status.get("updated_at", "?")
    st.success(f"Bot đang chạy · Cập nhật lần cuối: **{age}**")

# Metrics
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Mode", str(status.get("mode", settings.get("mode", "?"))).upper())
ks = status.get("kill_switch", False)
m2.metric("Kill Switch", "🔴 ON" if ks else "🟢 OFF")
m3.metric("Equity", f"{status.get('equity', 0):,.2f}")
m4.metric("Daily PnL", f"{status.get('daily_pnl', 0):,.2f}")
m5.metric("Open Pos", status.get("open_positions", 0))
m6.metric("Executed (last)", status.get("executed", 0))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw signals", status.get("raw_signals", 0))
c2.metric("Filtered", status.get("filtered_signals", 0))
c3.metric("Skip symbols", status.get("skip_symbols", 0))
c4.metric("Consecutive losses", status.get("consecutive_losses", 0))

if status.get("last_error"):
    st.error(f"Last error: {status['last_error']}")

tab1, tab2, tab3, tab4 = st.tabs(["📡 Signals", "📊 Positions", "📋 Orders (Excel)", "ℹ️ Status raw"])

with tab1:
    st.subheader("Recent signals (từ bot)")
    sigs = status.get("recent_signals") or []
    if sigs:
        df_sig = pd.DataFrame(sigs)
        cols = [c for c in ["timestamp", "symbol", "side", "strategy", "confidence", "entry", "sl", "tp", "rr", "reason"] if c in df_sig.columns]
        st.dataframe(df_sig[cols] if cols else df_sig, use_container_width=True, hide_index=True)
    else:
        st.info("Chưa có signal trong cycle gần nhất.")

with tab2:
    st.subheader("Open positions")
    pos = status.get("positions") or []
    if pos:
        st.dataframe(pd.DataFrame(pos), use_container_width=True, hide_index=True)
    else:
        st.write("Không có position mở (hoặc bot chưa report).")

with tab3:
    st.subheader("Lịch sử lệnh (data/orders.xlsx)")
    orders = load_orders(80)
    if orders is not None and len(orders) > 0:
        st.dataframe(orders, use_container_width=True, hide_index=True)
        st.caption(f"Hiển thị {len(orders)} dòng mới nhất")
    else:
        st.info("Chưa có lệnh nào được ghi. Sau khi bot EXECUTED sẽ xuất hiện ở đây.")

with tab4:
    st.subheader("Raw status.json")
    st.json(status if status else {"note": "empty"})

st.markdown("---")
st.caption("⚠️ Không phải lời khuyên tài chính. Bot và Dashboard chạy 2 process riêng.")

if auto_refresh:
    time.sleep(10)
    st.rerun()
