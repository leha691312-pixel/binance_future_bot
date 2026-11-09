"""
Streamlit Dashboard - Real-time style (auto-refresh + status)
Chạy: streamlit run dashboard/app.py
"""
import streamlit as st
import yaml
from pathlib import Path
import pandas as pd
from datetime import datetime
import time

st.set_page_config(page_title="Binance Futures Bot", layout="wide", page_icon="📈")

# Auto refresh every 15s
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

REFRESH_SEC = 15

cfg_path = Path(__file__).parent.parent / "config"
with open(cfg_path / "settings.yaml") as f:
    settings = yaml.safe_load(f)
with open(cfg_path / "risk.yaml") as f:
    risk = yaml.safe_load(f)

st.title("🤖 Binance Futures Multi-Agent Bot")
st.caption("Paper / Testnet / Live | ICT-SMC + FVG/BOS/CHoCH/OI | Risk Gate | Kill Switch | ML Filter")

# Sidebar
st.sidebar.header("⚙️ Control Panel")
mode = st.sidebar.selectbox("Mode", ["paper", "testnet", "live"], index=0)
top_n = st.sidebar.number_input("Top N coins", 5, 50, settings.get("top_n", 20))
scan_interval = st.sidebar.slider("Scan interval (s)", 30, 300, 60)
auto_refresh = st.sidebar.checkbox("Auto-refresh dashboard", value=True)

st.sidebar.subheader("Risk Parameters")
max_pos = st.sidebar.slider("Max position %", 1, 20, int(risk["max_position_pct"] * 100))
max_daily_loss = st.sidebar.slider("Max daily loss %", 1, 10, int(risk["max_daily_loss_pct"] * 100))
min_rr = st.sidebar.slider("Min RR", 1.0, 4.0, float(risk["min_risk_reward"]), 0.1)

col_ks1, col_ks2 = st.sidebar.columns(2)
if col_ks1.button("🛑 KILL SWITCH", type="primary"):
    st.session_state.kill_switch = True
    st.sidebar.error("Kill Switch ACTIVATED")
if col_ks2.button("Reset KS"):
    st.session_state.kill_switch = False
    st.sidebar.success("Kill Switch reset")

if "kill_switch" not in st.session_state:
    st.session_state.kill_switch = False

# Header metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Mode", mode.upper())
m2.metric("Kill Switch", "🔴 ON" if st.session_state.kill_switch else "🟢 OFF")
m3.metric("Top N", top_n)
m4.metric("Strategies", 7)
m5.metric("Last refresh", datetime.now().strftime("%H:%M:%S"))

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📡 Live Signals", "📊 Positions", "📈 Equity", "🧪 Backtest", "🧠 ML", "ℹ️ Status"
])

with tab1:
    st.subheader("Latest Signals (mock – connect bot for live)")
    mock_signals = pd.DataFrame([
        {"time": datetime.now().strftime("%H:%M:%S"), "symbol": "BTC/USDT", "side": "LONG", "strategy": "FVG", "confidence": 0.74, "rr": 2.2, "status": "PENDING"},
        {"time": datetime.now().strftime("%H:%M:%S"), "symbol": "ETH/USDT", "side": "SHORT", "strategy": "BOS_CHOCH", "confidence": 0.71, "rr": 2.4, "status": "REJECTED (RR)"},
        {"time": datetime.now().strftime("%H:%M:%S"), "symbol": "SOL/USDT", "side": "LONG", "strategy": "OI_Funding", "confidence": 0.68, "rr": 2.0, "status": "EXECUTED"},
        {"time": datetime.now().strftime("%H:%M:%S"), "symbol": "BNB/USDT", "side": "SHORT", "strategy": "LiquiditySweep", "confidence": 0.70, "rr": 1.9, "status": "PENDING"},
    ])
    st.dataframe(mock_signals, use_container_width=True, hide_index=True)
    st.info("Khi chạy `python main.py`, signals thật sẽ được ghi log. Dashboard có thể đọc từ Redis/shared state để real-time.")

with tab2:
    st.subheader("Open Positions")
    st.write("Chưa có position mở (paper mode).")
    st.progress(0.0, text="Exposure 0%")

with tab3:
    st.subheader("Equity Curve (demo)")
    eq = pd.Series(
        [10000, 10120, 10080, 10350, 10290, 10520, 10480, 10710, 10650, 10890],
        index=pd.date_range("2025-01-01", periods=10, freq="D"),
        name="Equity",
    )
    st.line_chart(eq)
    c1, c2, c3 = st.columns(3)
    c1.metric("Return", "+8.9%")
    c2.metric("Max DD", "-2.1%")
    c3.metric("Sharpe (est)", "1.4")

with tab4:
    st.subheader("Backtest & Walk-Forward & Stress")
    st.markdown("""
    ```bash
    python -m backtest.run_backtest
    ```
    - Commission + Slippage  
    - Walk-Forward Optimization  
    - Stress: fees 2x/3x, slippage cao, adverse, extreme  
    """)
    if st.button("Run Quick Backtest (demo)"):
        with st.spinner("Running..."):
            time.sleep(1.5)
        st.success("Demo done. Xem terminal để chạy backtest thật.")
        st.json({
            "total_trades": 47,
            "win_rate": 53.2,
            "profit_factor": 1.41,
            "max_drawdown_pct": 8.7,
            "total_return_pct": 12.4,
        })

with tab5:
    st.subheader("ML Agent")
    st.markdown("""
    **Features**: confidence, RR, side, strategy_id, body_ratio, ATR%, volume_ratio, RSI, dist high/low, trend strength.
    
    Train demo model:
    ```python
    from agents.ml_agent import MLAgent
    ml = MLAgent()
    print(ml.train_demo())
    ```
    """)
    if st.button("Train Demo ML Model"):
        st.info("Chạy trong terminal: python -c \"from agents.ml_agent import MLAgent; print(MLAgent().train_demo())\"")

with tab6:
    st.subheader("System Status")
    st.json({
        "mode": mode,
        "kill_switch": st.session_state.kill_switch,
        "risk_gate": {
            "max_position_pct": max_pos / 100,
            "max_daily_loss_pct": max_daily_loss / 100,
            "min_rr": min_rr,
        },
        "scanner": {"top_n": top_n, "interval": scan_interval},
        "strategies": ["ORB", "LiquiditySweep", "OrderBlock", "SupplyDemand", "FVG", "BOS_CHOCH", "OI_Funding"],
        "ml_enabled": True,
        "note": "True WebSocket real-time cần FastAPI + frontend riêng. Streamlit dùng polling/auto-refresh.",
    })

st.markdown("---")
st.caption("⚠️ Không phải lời khuyên tài chính. Giao dịch Futures có rủi ro mất vốn cao.")

# Auto refresh
if auto_refresh:
    time.sleep(0.1)
    # Streamlit experimental rerun on interval
    if time.time() - st.session_state.last_refresh > REFRESH_SEC:
        st.session_state.last_refresh = time.time()
        st.rerun()
