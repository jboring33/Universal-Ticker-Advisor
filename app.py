import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. PAGE & URL QUERY PARAMETER SETUP ---
st.set_page_config(
    page_title="Universal Asset Advisor & Intrinsic Valuation Model",
    page_icon="⚖️",
    layout="wide"
)

# Read ticker from browser URL query parameter if available (e.g., ?ticker=VFLO)
query_params = st.query_params
default_ticker = query_params.get("ticker", "VFLO").upper()


# --- 2. DATA FETCHING & ASSET CALIBRATION ---
def get_calibrated_inputs(ticker_symbol: str) -> dict:
    """Fetches ticker metadata via yfinance and derives baseline growth/metrics."""
    ticker = yf.Ticker(ticker_symbol)
    
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    quote_type = info.get("quoteType", "EQUITY")
    
    current_price = (
        info.get("currentPrice") 
        or info.get("navPrice") 
        or info.get("previousClose")
        or info.get("regularMarketPrice")
    )
    
    if not current_price:
        try:
            hist = ticker.history(period="5d")
            if not hist.empty:
                current_price = float(hist["Close"].iloc[-1])
            else:
                current_price = 100.0
        except Exception:
            current_price = 100.0

    growth_rate = 0.05
    exit_multiple = 15.0
    asset_class = "Stock"

    if quote_type == "ETF":
        asset_class = "ETF"
        try:
            hist = ticker.history(period="5y")
            if len(hist) > 250:
                start_price = float(hist["Close"].iloc[0])
                end_price = float(hist["Close"].iloc[-1])
                years = len(hist) / 252.0
                if start_price > 0:
                    growth_rate = (end_price / start_price) ** (1 / years) - 1.0
        except Exception:
            growth_rate = 0.06
        
        exit_multiple = info.get("trailingPE") or 18.0

    elif quote_type in ["MUTUALFUND", "MONEYMARKET"]:
        asset_class = "Fund / Bond"
        growth_rate = info.get("yield") or info.get("threeYearAverageReturn") or 0.04
        exit_multiple = 1.0

    else:
        asset_class = "Stock"
        growth_rate = info.get("earningsGrowth") or info.get("revenueGrowth") or 0.07
        exit_multiple = info.get("forwardPE") or info.get("trailingPE") or 20.0

    # Sanity Clamping
    growth_rate = max(-0.25, min(float(growth_rate), 0.50))
    if asset_class == "Stock":
        exit_multiple = max(3.0, min(float(exit_multiple), 80.0))

    return {
        "asset_class": asset_class,
        "current_price": float(current_price),
        "growth_rate": float(growth_rate),
        "exit_multiple": float(exit_multiple),
        "long_name": info.get("longName") or info.get("shortName") or ticker_symbol,
        "trailing_pe": info.get("trailingPE"),
        "info": info
    }


# --- 3. RULE ENGINE (ETF & ASSET SCORING) ---
def evaluate_weekly_rules(data: dict, rule_weights: dict) -> tuple[float, list]:
    """
    Evaluates quality/technical rules against ticker metadata.
    Returns total score (0-100) and line-item breakdown.
    """
    info = data["info"]
    asset_class = data["asset_class"]
    results = []
    total_score = 0.0
    max_possible = sum(rule_weights.values())

    # Example 10-Rule Ruleset (Tailor bounds as needed)
    rules_check = [
        ("AUM Size (> $100M)", info.get("totalAssets", 0) > 100_000_000, rule_weights["aum"]),
        ("Expense Ratio (< 0.50%)", info.get("expenseRatio", 0.003) <= 0.005, rule_weights["expense"]),
        ("Positive 5Y Growth/Return", data["growth_rate"] > 0.02, rule_weights["growth"]),
        ("Low Tracking Error / Beta (< 1.2)", info.get("beta", 1.0) <= 1.2 if info.get("beta") else True, rule_weights["beta"]),
        ("Healthy Dividend/Yield", (info.get("yield") or 0) > 0.01, rule_weights["yield"]),
        ("Active Trading Volume", info.get("volume", 500000) > 100000, rule_weights["volume"]),
        ("P/E Value Safety (< 25x)", (info.get("trailingPE") or 20) < 25.0, rule_weights["pe_ratio"]),
        ("Low Short Interest / Risk", info.get("shortPercentOfFloat", 0) < 0.05, rule_weights["risk"]),
        ("52-Week Price Position (> 30% Low)", data["current_price"] > (info.get("fiftyTwoWeekLow", data["current_price"]*0.8) * 1.1), rule_weights["momentum"]),
        ("Institutional Backing (> 20%)", info.get("heldPercentInstitutions", 0.5) > 0.2, rule_weights["inst_hold"])
    ]

    for label, passed, weight in rules_check:
        points = weight if passed else 0.0
        total_score += points
        results.append({
            "Rule": label,
            "Passed": "✅ Pass" if passed else "❌ Fail",
            "Points Awarded": f"{points} / {weight}"
        })

    normalized_score = (total_score / max_possible) * 100 if max_possible > 0 else 50.0
    return round(normalized_score, 1), results


# --- 4. VALUATION ENGINE ---
def calculate_scenario_valuation(init_price, growth_rate, exit_multiple, proj_years, trailing_pe, asset_class, discount_rate=0.09):
    compounded_val = init_price * ((1 + growth_rate) ** proj_years)
    
    if asset_class == "Stock" and trailing_pe:
        pe_expansion = exit_multiple / max(trailing_pe, 1.0)
        target_price = compounded_val * pe_expansion
    else:
        target_price = compounded_val

    intrinsic_value = target_price / ((1 + discount_rate) ** proj_years)
    cagr = (((target_price / init_price) ** (1 / proj_years)) - 1) * 100

    return {
        "target_price": target_price,
        "intrinsic_value": intrinsic_value,
        "cagr": cagr
    }


# --- 5. HYBRID DECISION ENGINE ---
def get_unified_recommendation(rule_score: float, med_cagr: float, margin_of_safety: float) -> tuple[str, str, float]:
    """Combines Rule Quality Score and Valuation Returns into a single decision."""
    val_score = min(max((med_cagr / 15.0) * 100, 0), 100) # 15% CAGR = 100
    composite_index = (0.40 * rule_score) + (0.60 * val_score)

    if composite_index >= 78 and margin_of_safety >= 0:
        return "🟢 STRONG BUY", "success", composite_index
    elif composite_index >= 65:
        return "🟢 BUY / ACCUMULATE", "success", composite_index
    elif composite_index >= 50:
        return "🟡 HOLD / NEUTRAL", "warning", composite_index
    elif composite_index >= 35:
        return "🟠 TRIM / OVERVALUED", "warning", composite_index
    else:
        return "🔴 SELL / HIGH RISK", "error", composite_index


# --- 6. STREAMLIT UI ---

st.title("⚖️ Universal Asset Advisor & Valuation Model")

# Top Search Bar & Favorite URL Sync
col_search, col_url = st.columns([2, 1])

with col_search:
    ticker_input = st.text_input("Enter Ticker Symbol:", value=default_ticker).strip().upper()

# Sync URL query parameters so user can bookmark/favorite URL
if ticker_input != query_params.get("ticker"):
    st.query_params["ticker"] = ticker_input

if ticker_input:
    try:
        with st.spinner(f"Running rules engine & valuation models for {ticker_input}..."):
            data = get_calibrated_inputs(ticker_input)

        st.subheader(f"{data['long_name']} ({ticker_input})")
        st.caption(f"Asset Class: **{data['asset_class']}** | Current Price: **${data['current_price']:,.2f}**")

        # SIDEBAR: Configurable Weekly Points & Valuation Sliders
        st.sidebar.header("⚙️ Weekly Points Configurator")
        st.sidebar.caption("Adjust weightings for quality rules (0–10 pts):")
        
        rule_weights = {
            "aum": st.sidebar.slider("AUM Size Weight", 0, 10, 10),
            "expense": st.sidebar.slider("Expense Ratio Weight", 0, 10, 10),
            "growth": st.sidebar.slider("Growth Trajectory Weight", 0, 10, 10),
            "beta": st.sidebar.slider("Volatility/Beta Weight", 0, 10, 5),
            "yield": st.sidebar.slider("Yield Weight", 0, 10, 5),
            "volume": st.sidebar.slider("Liquidity Weight", 0, 10, 5),
            "pe_ratio": st.sidebar.slider("Valuation Multiple Weight", 0, 10, 10),
            "risk": st.sidebar.slider("Short Interest Weight", 0, 10, 5),
            "momentum": st.sidebar.slider("52-Week Trend Weight", 0, 10, 5),
            "inst_hold": st.sidebar.slider("Institutional Support Weight", 0, 10, 5),
        }

        st.sidebar.markdown("---")
        st.sidebar.header("🎯 Valuation Parameters")
        proj_years = st.sidebar.slider("Projection Horizon (Years)", 1, 20, 5)
        discount_rate = st.sidebar.slider("Discount Rate (%)", 4.0, 15.0, 9.0, step=0.5) / 100.0

        base_growth = round(data["growth_rate"] * 100, 2)
        base_multiple = round(data["exit_multiple"], 1)

        growth_low = st.sidebar.slider("Low Growth Rate (%)", -20.0, 30.0, float(round(base_growth * 0.6, 2)), step=0.5) / 100.0
        growth_med = st.sidebar.slider("Medium Growth Rate (%)", -20.0, 40.0, float(base_growth), step=0.5) / 100.0
        growth_high = st.sidebar.slider("High Growth Rate (%)", -20.0, 60.0, float(round(base_growth * 1.4, 2)), step=0.5) / 100.0

        if data["asset_class"] == "Stock":
            mult_low = st.sidebar.slider("Low Exit P/E", 3.0, 60.0, float(round(base_multiple * 0.75, 1)), step=0.5)
            mult_med = st.sidebar.slider("Medium Exit P/E", 3.0, 70.0, float(base_multiple), step=0.5)
            mult_high = st.sidebar.slider("High Exit P/E", 3.0, 80.0, float(round(base_multiple * 1.25, 1)), step=0.5)
        else:
            mult_low = mult_med = mult_high = data["exit_multiple"]

        # Run Engines
        rule_score, rule_breakdown = evaluate_weekly_rules(data, rule_weights)
        
        init_price = data["current_price"]
        trailing_pe = data["trailing_pe"]
        asset_class = data["asset_class"]

        scenarios = {
            "Low Case": calculate_scenario_valuation(init_price, growth_low, mult_low, proj_years, trailing_pe, asset_class, discount_rate),
            "Medium (Base)": calculate_scenario_valuation(init_price, growth_med, mult_med, proj_years, trailing_pe, asset_class, discount_rate),
            "High Case": calculate_scenario_valuation(init_price, growth_high, mult_high, proj_years, trailing_pe, asset_class, discount_rate)
        }

        med_iv = scenarios["Medium (Base)"]["intrinsic_value"]
        med_cagr = scenarios["Medium (Base)"]["cagr"]
        margin_of_safety = ((med_iv - init_price) / init_price) * 100

        unified_signal, signal_type, composite_index = get_unified_recommendation(rule_score, med_cagr, margin_of_safety)

        # EXECUTIVE TOP BAR (Unified Verdict)
        st.write("### 🏆 Executive Decision Dashboard")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("10-Rule Quality Score", f"{rule_score} / 100")
        m2.metric("Medium Intrinsic Value", f"${med_iv:,.2f}", f"{margin_of_safety:+.1f}% Margin")
        m3.metric("Projected 5Y CAGR", f"{med_cagr:.1f}%")
        m4.metric("Composite Index", f"{composite_index:.1f} / 100")

        if signal_type == "success":
            st.success(f"**Unified Recommendation:** {unified_signal} | High Quality + Strong Intrinsic Return upside.")
        elif signal_type == "warning":
            st.warning(f"**Unified Recommendation:** {unified_signal} | Fairly valued or quality score imposes limits.")
        else:
            st.error(f"**Unified Recommendation:** {unified_signal} | Fails quality rules or lacks sufficient margin of safety.")

        st.divider()

        # TWO-COLUMN DETAILED DRILLDOWN
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("📋 10-Rule Scorecard Breakdown")
            df_rules = pd.DataFrame(rule_breakdown)
            st.dataframe(df_rules, use_container_width=True, hide_index=True)

        with col_right:
            st.subheader("🏛️ Valuation Scenario Matrix")
            summary_df = pd.DataFrame([
                {
                    "Scenario": k,
                    "Intrinsic Value": v["intrinsic_value"],
                    "Target Price": v["target_price"],
                    "Implied CAGR": v["cagr"]
                }
                for k, v in scenarios.items()
            ]).set_index("Scenario")
            
            st.dataframe(
                summary_df.style.format({
                    "Intrinsic Value": "${:,.2f}",
                    "Target Price": "${:,.2f}",
                    "Implied CAGR": "{:,.2f}%"
                }),
                use_container_width=True
            )

        # TRAJECTORY CHART
        st.subheader("📈 Multi-Scenario Trajectory")
        years_seq = list(range(0, proj_years + 1))
        chart_data = {"Year": years_seq}

        for sc_name, g_rate, m_val in [
            ("Low Case", growth_low, mult_low),
            ("Medium (Base)", growth_med, mult_med),
            ("High Case", growth_high, mult_high)
        ]:
            prices = []
            for y in years_seq:
                base_val = init_price * ((1 + g_rate) ** y)
                if asset_class == "Stock" and trailing_pe:
                    pe_step = trailing_pe + (m_val - trailing_pe) * (y / proj_years)
                    price = base_val * (pe_step / max(trailing_pe, 1.0))
                else:
                    price = base_val
                prices.append(price)
            chart_data[sc_name] = prices

        st.line_chart(pd.DataFrame(chart_data).set_index("Year"))

    except Exception as e:
        st.error(f"Unable to process ticker '{ticker_input}'. Please check the symbol and try again.")
        st.caption(f"Error details: {e}")
