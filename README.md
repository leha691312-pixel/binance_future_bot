# Binance Futures Multi-Agent Trading Bot

Bot giao dịch Futures hỗ trợ **Paper / Testnet (demo.binance.com) / Live**, multi-agent, Risk Gate bắt buộc, chống lệnh trùng, Kill Switch, quét Top N, auto TP/SL, ML filter, và **backtest nghiêm ngặt + walk-forward + stress test**.

## ⚠️ CẢNH BÁO

- Giao dịch Futures có rủi ro **mất toàn bộ vốn**.
- ICT / SMC / Liquidity Sweep / Order Block là các khái niệm chủ quan. Code hiện tại chỉ là **xấp xỉ hợp lý**, không phải “full ICT” hoàn hảo.
- Luôn test kỹ trên **Paper → Testnet** ít nhất vài tuần trước khi nghĩ tới Live.
- Không phải lời khuyên tài chính.

## Tính năng

| Tính năng                    | Trạng thái      |
|-----------------------------|-----------------|
| Paper / Testnet / Live      | ✅              |
| Multi-agent                 | ✅              |
| Risk Gate bắt buộc          | ✅              |
| Chống lệnh trùng            | ✅              |
| Kill Switch                 | ✅              |
| Quét Top N coin             | ✅              |
| Auto TP / SL                | ✅ (simplified) |
| Strategies (ORB, Liq Sweep, OB, S&D, FVG, BOS/CHoCH, OI_Funding) | ✅ |
| ML filter (feature eng + train) | ✅ |
| Dashboard Streamlit         | ✅              |
| Backtest + Walk-Forward + Stress | ✅        |

## Cài đặt

```bash
cd binance_futures_bot
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tạo file `.env`:

```
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
BINANCE_TESTNET_API_KEY=your_testnet_key
BINANCE_TESTNET_API_SECRET=your_testnet_secret
```

## Chạy Bot

```bash
# Paper (mặc định)
python main.py

# Đổi mode trong config/settings.yaml → testnet hoặc live
```

## Dashboard

```bash
streamlit run dashboard/app.py
```

## Backtest nghiêm ngặt

```bash
python -m backtest.run_backtest
```

Sẽ chạy:
1. Base backtest (commission + slippage)
2. Stress test (fees 2x, slippage 3x, adverse, extreme)
3. Walk-forward optimization windows

## Cấu trúc

```
binance_futures_bot/
├── agents/
│   ├── risk_gate.py          # BẮT BUỘC
│   ├── scanner.py
│   ├── execution.py
│   ├── ml_agent.py
│   └── strategies/           # ORB, LiquiditySweep, OrderBlock, SupplyDemand, FVG, BOS_CHOCH
├── backtest/
│   ├── engine.py             # Walk-forward + Stress
│   └── run_backtest.py
├── core/
│   ├── exchange.py           # 3 modes
│   └── models.py
├── dashboard/app.py
├── config/
├── main.py
└── requirements.txt
```

## Strategies hiện có (ICT/SMC approximations)

| Strategy     | Mô tả |
|--------------|-------|
| ORB          | Opening Range Breakout |
| LiquiditySweep | Equal highs/lows → sweep → reclaim |
| OrderBlock   | Impulse → Order Block retest |
| SupplyDemand | Base nhỏ + impulse zone |
| **FVG**      | Fair Value Gap (3-candle imbalance) retest |
| **BOS_CHOCH**| Break of Structure + Change of Character |
| **OI_Funding** | Open Interest + Funding rate divergence |

Có thể mở rộng thêm: Premium/Discount zones, Killzones (London/NY), Multi-timeframe confluence.

Thêm class mới trong `agents/strategies/` và đăng ký trong `__init__.py`.

## Risk Gate logic chính

- Max position size
- Max total exposure
- Min Risk:Reward
- Daily loss limit → Kill Switch
- Consecutive losses → Kill Switch
- Duplicate order window
- API error rate

## Unit Tests
```bash
pytest tests/ -v
```

## Train ML demo
```bash
python scripts/train_ml.py
```

## Khuyến nghị sử dụng

1. Chạy backtest + stress test trước.
2. Paper trading ≥ 2–4 tuần.
3. Testnet ≥ 2 tuần.
4. Live chỉ với size rất nhỏ + monitoring liên tục.
5. Luôn giữ Kill Switch sẵn sàng.
