# Universal Ticker Advisor (UTA) 📈

**Universal Ticker Advisor (UTA)** is a multi-asset financial evaluation engine built with Streamlit and `yfinance`. It transitions beyond single-asset ETF scoring into a comprehensive evaluation platform capable of analyzing stocks, broad market ETFs, fixed-income funds, and hybrid assets through a unified, hybrid evaluation architecture.

---

## 🌟 Key Features

* **Multi-Asset Auto-Calibration:** Automatically identifies asset classes (Equity, ETF, Bond/Fixed Income) and applies appropriate benchmark rules and baselines.
* **Dual-Engine Scoring Architecture:**
  * **Weighted Quality Scorecard (40%):** Evaluates assets across a 10-rule fundamental framework including momentum, expense ratio/fees, valuation multiples, drawdown stability, and volume liquidity.
  * **Multi-Scenario Intrinsic Valuation (60%):** Computes discounted baseline, optimistic, and conservative intrinsic values to estimate target margin-of-safety entry points.
* **Interactive Rule Configurator:** Adjust rule weights, thresholds, and scenario assumptions live via sidebar controls.
* **Dynamic Bookmark & State Persistence:** Deep-link directly into specific asset analyses and custom scenarios using URL query parameters (`?ticker=XYZ`).

---

## 📁 Repository Structure

```text
Universal-Ticker-Advisor/
├── .streamlit/
│   └── config.toml          # Custom theme and UI preferences
├── app.py                   # Main Streamlit application entry point
├── requirements.txt         # Package dependencies for cloud deployment
├── LICENSE                  # MIT License
└── README.md                # Project documentation
