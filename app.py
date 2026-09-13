import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. PAGE CONFIGURATION & URL QUERY PARAMETERS ---
st.set_page_config(
    page_title="Universal Ticker Advisor (UTA)",
    page_icon="⚖️",
    layout="wide"
)

# Read ticker from browser URL query parameter if present (e.g., ?ticker=VFLO)
query_params = st.query_params
default_ticker = query_params.get("ticker", "VFLO").upper()


# --- 2. DATA FETCHING & ASSET CALIBRATION ENGINE ---
def get_calibrated_inputs(ticker_symbol: str) -> dict:
    """Fetches ticker metadata via yfinance and derives baseline growth & metrics."""
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


# --- 3. RULE EVALUATION ENGINE ---
def evaluate_weekly_rules(data: dict, rule_weights: dict) -> tuple[float, list]:
    """Evaluates 10 quality/risk rules against asset metadata."""
    info = data["info"]
    results = []
    total_score = 0.0
    max_possible = sum(rule_weights.values())

    rules_check = [
        ("AUM Size (> $100M)", info.get("totalAssets", 0) > 100_000_000, rule_weights["aum"]),
        ("Expense Ratio (< 0.50%)", info.get("expenseRatio", 0.003) <= 0.005, rule_weights["expense"]),
        ("Positive Trajectory (> 2% Growth)", data["growth_rate"] > 0.02, rule_weights["growth"]),
        ("Low Volatility (Beta <= 1.2)", info.get("beta", 1.0) <= 1.2 if info.get("beta") else True, rule_weights["beta"]),
        ("Healthy Income/Yield (> 1%)", (info.get("yield") or 0) > 0.01, rule_weights["yield"]),
        ("Active Trading Volume (> 100k)", info.get("volume", 500000) > 100000, rule_weights["volume"]),
        ("Valuation Safety (P/E < 25x)", (info.get("trailingPE") or 20) < 25.0, rule_weights["pe_ratio"]),
        ("Low Short Interest Risk (< 5%)", info.get("shortPercentOfFloat", 0) < 0.05, rule_weights["risk"]),
        ("52-Week Price Position (> 10% Low)", data["current_price"] > (info.get("fiftyTwoWeekLow", data["current_price"] * 0.8) * 1.1), rule_weights["momentum"]),
        ("Institutional Support (> 20%)", info.get("heldPercentInstitutions", 0.5) > 0.2, rule_weights["inst_hold"])
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
def calculate_scenario_valuation(init_price, growth_rate, exit_multiple, proj_years, trailing_pe, asset_class, discount_rate):
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
    """Blends Quality Score (40%) and Valuation CAGR (60%) into a final rating."""
    val_score = min(max((med_cagr / 15.0) * 100, 0), 100)  # 15% CAGR maps to 100
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


# --- 6. STREAMLIT APPLICATION INTERFACE ---

st.title("⚖️ Universal Ticker Advisor (UTA)")

# Ticker Search & URL Parameter Sync
ticker_input = st.text_input("Enter Ticker Symbol:", value=default_ticker).strip().upper()

if ticker_input != query_params.get("ticker"):
    st.query_params["ticker"] = ticker_input

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("🌐 Macro Environment Presets")
outlook = st.sidebar.selectbox(
    "Select Market Outlook:",
    options=["Neutral (Baseline)", "Bear / Defensive 🐻", "Bull / Risk-On 🐂"],
    index=0,
    help="Adjusts discount rates, valuation haircuts, and baseline rule defaults based on macro conditions."
)

# 1. Macro Outlook Preset Calibration
if outlook == "Bear / Defensive 🐻":
    default_aum, default_expense, default_growth = 10, 10, 5
    default_beta, default_yield, default_volume = 10, 8, 8
    default_pe, default_risk, default_momentum, default_inst = 10, 8, 2, 5
    preset_discount_rate = 11.0
    growth_scale = 0.70
    multiple_scale = 0.80

elif outlook == "Bull / Risk-On 🐂":
    default_aum, default_expense, default_growth = 5, 5, 10
    default_beta, default_yield, default_volume = 3, 2, 5
    default_pe, default_risk, default_momentum, default_inst = 4, 3, 10, 8
    preset_discount_rate = 8.0
    growth_scale = 1.20
    multiple_scale = 1.15

else:  # Neutral (Baseline)
    default_aum, default_expense, default_growth = 10, 10, 10
    default_beta, default_yield, default_volume = 5, 5, 5
    default_pe, default_risk, default_momentum, default_inst = 10, 5, 5, 5
    preset_discount_rate = 9.0
    growth_scale = 1.0
    multiple_scale = 1.0

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Strategy Profile Presets")

# Fly-over Tooltips Dictionary
PROFILE_HELP = {
    "Macro Alignment (Default)": (
        "Synchronizes weights directly with your chosen Macro Outlook. "
        "Best for general analysis aligned with prevailing market conditions."
    ),
    "Capital Preservation 🛡️": (
        "Prioritizes low-volatility, large-cap stability, high liquidity, and low short risk. "
        "Best for conservative investors seeking drawdown protection during market highs."
    ),
    "Dividend & Income Focus 💰": (
        "Emphasizes strong cash yield, low expense ratios, and valuation safety. "
        "Best for income-focused portfolios prioritizing steady cash flow."
    ),
    "Deep Value & Safety 🔍": (
        "Focuses heavily on low P/E multiples, expense efficiency, and minimal short risk. "
        "Best for finding underpriced, mispriced, or out-of-favor assets."
    ),
    "Growth & Momentum 🚀": (
        "Weights 52-week momentum trends, growth trajectory, and institutional backing. "
        "Best for bull markets and aggressive capital appreciation strategies."
    )
}

# Interactive Decision Assistant Expander
with st.sidebar.expander("💡 Help Me Choose a Strategy"):
    st.markdown("**Quick Selector Guide:**")
    investment_goal = st.radio(
        "What is your primary investment goal?",
        options=[
            "Protect Capital (Conservative)",
            "Generate Cash Flow / Yield",
            "Bargain Hunting / Value",
            "Maximize Growth & Momentum"
        ],
        index=0
    )
    
    if "Protect Capital" in investment_goal:
        recommended_profile = "Capital Preservation 🛡️"
    elif "Generate Cash Flow" in investment_goal:
        recommended_profile = "Dividend & Income Focus 💰"
    elif "Bargain Hunting" in investment_goal:
        recommended_profile = "Deep Value & Safety 🔍"
    else:
        recommended_profile = "Growth & Momentum 🚀"
        
    st.info(f"**Recommended Profile:**\n\n`{recommended_profile}`")

profile_options = list(PROFILE_HELP.keys())
default_index = profile_options.index(recommended_profile) if 'recommended_profile' in locals() else 0

rule_preset = st.sidebar.selectbox(
    "Select Strategy Profile:",
    options=profile_options,
    index=default_index,
    help="Select a profile to pre-fill rule weights based on your investment strategy."
)

st.sidebar.caption(f"ℹ️ **Profile Focus:** {PROFILE_HELP[rule_preset]}")

# 2. Strategy Weight Presets Mapping
if rule_preset == "Capital Preservation 🛡️":
    w_aum, w_exp, w_growth = 10, 10, 3
    w_beta, w_yield, w_vol = 10, 6, 10
    w_pe, w_risk, w_mom, w_inst = 8, 10, 2, 6

elif rule_preset == "Dividend & Income Focus 💰":
    w_aum, w_exp, w_growth = 8, 10, 4
    w_beta, w_yield, w_vol = 8, 10, 6
    w_pe, w_risk, w_mom, w_inst = 8, 6, 2, 4

elif rule_preset == "Deep Value & Safety 🔍":
    w_aum, w_exp, w_growth = 6, 8, 5
    w_beta, w_yield, w_vol = 6, 6, 6
    w_pe, w_risk, w_mom, w_inst = 10, 10, 3, 5

elif rule_preset == "Growth & Momentum 🚀":
    w_aum, w_exp, w_growth = 4, 4, 10
    w_beta, w_yield, w_vol = 2, 1, 5
    w_pe, w_risk, w_mom, w_inst = 3, 4, 10, 8

else:  # Macro Alignment (Default)
    w_aum, w_exp, w_growth = default_aum, default_expense, default_growth
    w_beta, w_yield, w_vol = default_beta, default_yield, default_volume
    w_pe, w_risk, w_mom, w_inst = default_pe, default_risk, default_momentum, default_inst

# Rule Weight Sliders
rule_weights = {
    "aum": st.sidebar.slider("AUM Size Weight", 0, 10, w_aum),
    "expense": st.sidebar.slider("Expense Ratio Weight", 0, 10, w_exp),
    "growth": st.sidebar.slider("Growth Trajectory Weight", 0, 10, w_growth),
    "beta": st.sidebar.slider("Low Volatility Weight", 0, 10, w_beta),
    "yield": st.sidebar.slider("Income/Yield Weight", 0, 10, w_yield),
    "volume": st.sidebar.slider("Liquidity Weight", 0, 10, w_vol),
    "pe_ratio": st.sidebar.slider("Valuation Multiple Weight", 0, 10, w_pe),
    "risk": st.sidebar.slider("Short Interest Weight", 0, 10, w_risk),
    "momentum": st.sidebar.slider("Momentum Weight", 0, 10, w_mom),
    "inst_hold": st.sidebar.slider("Institutional Backing Weight", 0, 10, w_inst),
}

st.sidebar.markdown("---")
st.sidebar.header("🎯 Valuation Parameters")

proj_years = st.sidebar.slider("Projection Horizon (Years)", 1, 20, 5)
discount_rate = st.sidebar.slider("Discount Rate (%)", 4.0, 15.0, preset_discount_rate, step=0.5) / 100.0


# --- MAIN ENGINE PROCESSING ---
if ticker_input:
    try:
        with st.spinner(f"Processing evaluation engines for {ticker_input}..."):
            data = get_calibrated_inputs(ticker_input)

        st.subheader(f"{data['long_name']} ({ticker_input})")
        st.caption(
            f"Asset Class: **{data['asset_class']}** | Current Price: **${data['current_price']:,.2f}** | "
            f"Macro Outlook: **{outlook}** | Strategy Profile: **{rule_preset}**"
        )

        base_growth = data["growth_rate"] * growth_scale
        base_multiple = data["exit_multiple"] * multiple_scale

        growth_low = st.sidebar.slider("Low Growth Rate (%)", -20.0, 30.0, float(round(base_growth * 0.6 * 100, 2)), step=0.5) / 100.0
        growth_med = st.sidebar.slider("Medium Growth Rate (%)", -20.0, 40.0, float(round(base_growth * 100, 2)), step=0.5) / 100.0
        growth_high = st.sidebar.slider("High Growth Rate (%)", -20.0, 60.0, float(round(base_growth * 1.4 * 100, 2)), step=0.5) / 100.0

        if data["asset_class"] == "Stock":
            mult_low = st.sidebar.slider("Low Exit P/E", 3.0, 60.0, float(round(base_multiple * 0.75, 1)), step=0.5)
            mult_med = st.sidebar.slider("Medium Exit P/E", 3.0, 70.0, float(round(base_multiple, 1)), step=0.5)
            mult_high = st.sidebar.slider("High Exit P/E", 3.0, 80.0, float(round(base_multiple * 1.25, 1)), step=0.5)
        else:
            mult_low = mult_med = mult_high = base_multiple

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

        # EXECUTIVE DASHBOARD DISPLAY
        st.write("### 🏆 Executive Decision Dashboard")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("10-Rule Quality Score", f"{rule_score} / 100")
        m2.metric("Medium Intrinsic Value", f"${med_iv:,.2f}", f"{margin_of_safety:+.1f}% Margin")
        m3.metric("Projected 5Y CAGR", f"{med_cagr:.1f}%")
        m4.metric("Composite Conviction Index", f"{composite_index:.1f} / 100")

        if signal_type == "success":
            st.success(f"**Unified Recommendation:** {unified_signal} | Strong fundamental quality and valuation upside.")
        elif signal_type == "warning":
            st.warning(f"**Unified Recommendation:** {unified_signal} | Fair valuation or restricted by rule penalties.")
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

        # MULTI-SCENARIO TRAJECTORY CHART
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
        st.error(f"Unable to process ticker '{ticker_input}'. Please verify the symbol.")
        st.caption(f"Error details: {e}")
