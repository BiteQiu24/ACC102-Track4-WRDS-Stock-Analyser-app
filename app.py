"""
ACC102 Mini Assignment – Track 4
WRDS Stock Performance Analyser
Enhanced version with richer academic data analysis and evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import wrds
from datetime import date, timedelta

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(page_title="WRDS Stock Analyser | ACC102", layout="wide")
st.title("📊 WRDS Stock Performance Analyser (ACC102 Track 4)")
st.caption("Data Source: WRDS CRSP | Dual Stock Comparison with Market Benchmark")

# ── Core Functions ────────────────────────────────────────────────────────────
@st.cache_resource
def connect_wrds(username: str, password: str):
    try:
        return wrds.Connection(wrds_username=username, wrds_password=password)
    except Exception as e:
        st.error(f"WRDS connection failed: {e}")
        return None

@st.cache_data
def get_crsp_max_date(username: str):
    db = wrds.Connection(wrds_username=username)
    try:
        result = db.raw_sql("SELECT MAX(date) AS max_date FROM crsp.dsf")
        return pd.to_datetime(result["max_date"].iloc[0]).date()
    finally:
        db.close()

def get_permno_by_ticker(db, ticker: str):
    ticker = ticker.upper().strip()
    query = f"""
    SELECT permno
    FROM crsp.stocknames
    WHERE ticker = '{ticker}'
    ORDER BY namedt DESC
    LIMIT 1
    """
    result = db.raw_sql(query)
    if result.empty:
        return None
    return int(result["permno"].iloc[0])

@st.cache_data
def get_price_data(username: str, permno: int, start: str, end: str):
    db = wrds.Connection(wrds_username=username)
    try:
        query = f"""
        SELECT date, prc, cfacpr
        FROM crsp.dsf
        WHERE permno = {permno}
          AND date >= '{start}'
          AND date <= '{end}'
        ORDER BY date
        """
        df = db.raw_sql(query)
    finally:
        db.close()

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df["prc"] = pd.to_numeric(df["prc"], errors="coerce").abs()
    df["cfacpr"] = pd.to_numeric(df["cfacpr"], errors="coerce").replace(0, np.nan)
    df["adj_prc"] = df["prc"] / df["cfacpr"]
    df = df.dropna(subset=["adj_prc"])
    df = df[df["adj_prc"] > 0]

    if df.empty:
        return pd.DataFrame()

    df["daily_return"] = np.log(df["adj_prc"] / df["adj_prc"].shift(1))
    df["simple_return"] = df["adj_prc"].pct_change()
    df["cum_return"] = np.exp(df["daily_return"].fillna(0).cumsum()) - 1
    df["wealth_index"] = (1 + df["simple_return"].fillna(0)).cumprod()
    df["running_max"] = df["wealth_index"].cummax()
    df["drawdown"] = df["wealth_index"] / df["running_max"] - 1

    return df

def compute_metrics(price_series: pd.Series, return_series: pd.Series):
    p = price_series.dropna()
    r = return_series.dropna()

    if len(p) < 2 or len(r) < 1:
        return {
            "Total Return (%)": np.nan,
            "Annual Return (%)": np.nan,
            "Annual Volatility (%)": np.nan,
            "Sharpe Ratio": np.nan,
            "Final Price ($)": np.nan
        }

    total_ret = (p.iloc[-1] / p.iloc[0] - 1) * 100
    ann_ret = ((p.iloc[-1] / p.iloc[0]) ** (252 / len(p)) - 1) * 100
    ann_vol = r.std() * np.sqrt(252) * 100
    sharpe = (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() != 0 else np.nan

    return {
        "Total Return (%)": round(total_ret, 2),
        "Annual Return (%)": round(ann_ret, 2),
        "Annual Volatility (%)": round(ann_vol, 2),
        "Sharpe Ratio": round(sharpe, 3) if pd.notna(sharpe) else np.nan,
        "Final Price ($)": round(p.iloc[-1], 2)
    }

def compute_advanced_metrics(asset_df: pd.DataFrame, bench_df: pd.DataFrame, risk_free_rate: float = 0.0):
    merged = pd.concat(
        [
            asset_df["simple_return"].rename("asset"),
            bench_df["simple_return"].rename("bench"),
            asset_df["drawdown"].rename("drawdown")
        ],
        axis=1
    ).dropna()

    if merged.empty or len(merged) < 2:
        return {
            "Max Drawdown (%)": np.nan,
            "Correlation with Benchmark": np.nan,
            "Beta vs Benchmark": np.nan,
            "Alpha vs Benchmark (%)": np.nan,
            "Tracking Error (%)": np.nan,
            "Positive-Day Ratio (%)": np.nan
        }

    asset_r = merged["asset"]
    bench_r = merged["bench"]
    rf_daily = risk_free_rate / 252

    covariance = np.cov(asset_r, bench_r, ddof=1)[0, 1]
    variance_b = np.var(bench_r, ddof=1)
    beta = covariance / variance_b if variance_b != 0 else np.nan

    ann_asset = asset_r.mean() * 252
    ann_bench = bench_r.mean() * 252
    alpha = (ann_asset - risk_free_rate) - beta * (ann_bench - risk_free_rate) if pd.notna(beta) else np.nan
    tracking_error = (asset_r - bench_r).std() * np.sqrt(252)
    positive_day_ratio = (asset_r > 0).mean() * 100
    max_drawdown = merged["drawdown"].min() * 100
    correlation = asset_r.corr(bench_r)

    return {
        "Max Drawdown (%)": round(max_drawdown, 2),
        "Correlation with Benchmark": round(correlation, 3) if pd.notna(correlation) else np.nan,
        "Beta vs Benchmark": round(beta, 3) if pd.notna(beta) else np.nan,
        "Alpha vs Benchmark (%)": round(alpha * 100, 2) if pd.notna(alpha) else np.nan,
        "Tracking Error (%)": round(tracking_error * 100, 2),
        "Positive-Day Ratio (%)": round(positive_day_ratio, 2)
    }

def winner_text(metric_name: str, val1, val2, higher_is_better=True):
    if pd.isna(val1) or pd.isna(val2):
        return f"{metric_name}: comparison unavailable because of insufficient data."
    if np.isclose(val1, val2):
        return f"{metric_name}: both stocks are broadly similar over the selected period."
    if higher_is_better:
        better = "Stock 1" if val1 > val2 else "Stock 2"
    else:
        better = "Stock 1" if val1 < val2 else "Stock 2"
    return f"{metric_name}: {better} performs better on this indicator over the selected period."

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Connect to Database")
    wrds_user = st.text_input("WRDS Username")
    wrds_pwd = st.text_input("WRDS Password", type="password")
    connect_btn = st.button("Connect to Database")

    db_conn = None
    if connect_btn and wrds_user and wrds_pwd:
        db_conn = connect_wrds(wrds_user, wrds_pwd)
        if db_conn:
            st.session_state["db_conn"] = db_conn
            st.session_state["username"] = wrds_user
            st.session_state["password"] = wrds_pwd
            try:
                st.session_state["crsp_max_date"] = get_crsp_max_date(wrds_user)
            except Exception:
                st.session_state["crsp_max_date"] = None
            st.success("Connected successfully.")
    elif "db_conn" in st.session_state:
        db_conn = st.session_state["db_conn"]
        wrds_user = st.session_state["username"]

    st.markdown("---")
    st.subheader("Stock Configuration")
    ticker1 = st.text_input("Stock 1 (e.g. AAPL)", value="AAPL")
    ticker2 = st.text_input("Stock 2 (e.g. MSFT)", value="MSFT")
    benchmark_ticker = st.text_input("Market Benchmark (e.g. SPY)", value="SPY")

    st.subheader("Date Range")
    max_date = st.session_state.get("crsp_max_date", date.today())
    default_end = max_date if max_date else date.today()
    default_start = default_end - timedelta(days=365 * 3)
    date_range = st.date_input("Select Time Period", value=(default_start, default_end))

    if max_date:
        st.caption(f"Latest CRSP daily data available in this environment: {max_date}")

    st.subheader("Evaluation Settings")
    risk_free_rate_pct = st.number_input(
        "Annual risk-free rate (%) for alpha analysis",
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=0.1
    )
    rolling_window = st.slider("Rolling volatility window (trading days)", 20, 120, 60, 10)

    query_btn = st.button("Query All Stock Data")

# ── Main logic ────────────────────────────────────────────────────────────────
if not db_conn:
    st.info("Please log in to WRDS in the sidebar first.")
    st.stop()

if len(date_range) != 2:
    st.warning("Please select both a start date and an end date.")
    st.stop()

start_date, end_date = date_range
crsp_max_date = st.session_state.get("crsp_max_date")
if crsp_max_date and end_date > crsp_max_date:
    st.warning(
        f"The selected end date ({end_date}) is later than the latest CRSP date available "
        f"({crsp_max_date}). The app will use {crsp_max_date} instead."
    )
    end_date = crsp_max_date

if start_date > end_date:
    st.error("The start date must be earlier than or equal to the end date.")
    st.stop()

if query_btn:
    with st.spinner("Retrieving stock identifiers..."):
        permno1 = get_permno_by_ticker(db_conn, ticker1)
        permno2 = get_permno_by_ticker(db_conn, ticker2)
        permno_bench = get_permno_by_ticker(db_conn, benchmark_ticker)

    missing_permnos = []
    if permno1 is None:
        missing_permnos.append(ticker1.upper())
    if permno2 is None:
        missing_permnos.append(ticker2.upper())
    if permno_bench is None:
        missing_permnos.append(benchmark_ticker.upper())

    if missing_permnos:
        st.error("Could not find CRSP identifiers for: " + ", ".join(missing_permnos))
        st.stop()

    st.session_state["permnos"] = {
        "ticker1": ticker1.upper(), "permno1": permno1,
        "ticker2": ticker2.upper(), "permno2": permno2,
        "benchmark": benchmark_ticker.upper(), "permno_bench": permno_bench
    }
    st.success("All stock identifiers retrieved successfully.")

if "permnos" not in st.session_state:
    st.info("Set your tickers and click 'Query All Stock Data'.")
    st.stop()

permnos = st.session_state["permnos"]
ticker1, permno1 = permnos["ticker1"], permnos["permno1"]
ticker2, permno2 = permnos["ticker2"], permnos["permno2"]
bench_ticker, permno_bench = permnos["benchmark"], permnos["permno_bench"]

start_str = str(start_date)
end_str = str(end_date)

with st.spinner("Loading data from WRDS CRSP..."):
    df1 = get_price_data(wrds_user, permno1, start_str, end_str)
    df2 = get_price_data(wrds_user, permno2, start_str, end_str)
    df_bench = get_price_data(wrds_user, permno_bench, start_str, end_str)

empty_labels = []
if df1.empty:
    empty_labels.append(ticker1)
if df2.empty:
    empty_labels.append(ticker2)
if df_bench.empty:
    empty_labels.append(bench_ticker)

if empty_labels:
    st.error(
        "No price data were returned for: "
        + ", ".join(empty_labels)
        + f". Selected range: {start_str} to {end_str}."
    )
    st.info(
        "This usually means either: "
        "1) the selected dates are beyond the latest CRSP data available, "
        "2) the ticker/identifier mapping is not valid for that period, or "
        "3) the chosen period is too narrow for one of the securities."
    )
    st.stop()

# ── Summary Metrics ───────────────────────────────────────────────────────────
metrics1 = compute_metrics(df1["adj_prc"], df1["daily_return"])
metrics2 = compute_metrics(df2["adj_prc"], df2["daily_return"])
metrics_bench = compute_metrics(df_bench["adj_prc"], df_bench["daily_return"])

adv1 = compute_advanced_metrics(df1, df_bench, risk_free_rate_pct / 100)
adv2 = compute_advanced_metrics(df2, df_bench, risk_free_rate_pct / 100)

st.subheader("Performance Metrics Summary")
metrics_df = pd.DataFrame({
    ticker1: metrics1,
    ticker2: metrics2,
    f"{bench_ticker} (Benchmark)": metrics_bench
}).T
st.dataframe(metrics_df, use_container_width=True)

st.subheader("Advanced Evaluation Metrics")
advanced_df = pd.DataFrame({
    ticker1: adv1,
    ticker2: adv2
}).T
st.dataframe(advanced_df, use_container_width=True)

# ── Charts ────────────────────────────────────────────────────────────────────
st.subheader("Adjusted Price Comparison")
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=df1.index, y=df1["adj_prc"], mode="lines", name=ticker1))
fig_price.add_trace(go.Scatter(x=df2.index, y=df2["adj_prc"], mode="lines", name=ticker2))
fig_price.add_trace(go.Scatter(x=df_bench.index, y=df_bench["adj_prc"], mode="lines", name=bench_ticker))
fig_price.update_layout(xaxis_title="Date", yaxis_title="Adjusted Price")
st.plotly_chart(fig_price, use_container_width=True)

st.subheader("Normalised Price Comparison (Base = 100)")
for df in [df1, df2, df_bench]:
    df["norm_prc"] = df["adj_prc"] / df["adj_prc"].iloc[0] * 100

fig_norm = go.Figure()
fig_norm.add_trace(go.Scatter(x=df1.index, y=df1["norm_prc"], mode="lines", name=ticker1))
fig_norm.add_trace(go.Scatter(x=df2.index, y=df2["norm_prc"], mode="lines", name=ticker2))
fig_norm.add_trace(go.Scatter(x=df_bench.index, y=df_bench["norm_prc"], mode="lines", name=bench_ticker))
fig_norm.update_layout(xaxis_title="Date", yaxis_title="Normalised Price")
st.plotly_chart(fig_norm, use_container_width=True)

st.subheader("Cumulative Return Comparison (%)")
fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(x=df1.index, y=df1["cum_return"] * 100, mode="lines", name=ticker1))
fig_cum.add_trace(go.Scatter(x=df2.index, y=df2["cum_return"] * 100, mode="lines", name=ticker2))
fig_cum.add_trace(go.Scatter(x=df_bench.index, y=df_bench["cum_return"] * 100, mode="lines", name=bench_ticker))
fig_cum.update_layout(xaxis_title="Date", yaxis_title="Cumulative Return (%)")
st.plotly_chart(fig_cum, use_container_width=True)

st.subheader("Risk vs Return")
scatter_data = pd.DataFrame({
    "Stock": [ticker1, ticker2, bench_ticker],
    "Annual Return (%)": [
        metrics1["Annual Return (%)"],
        metrics2["Annual Return (%)"],
        metrics_bench["Annual Return (%)"]
    ],
    "Annual Volatility (%)": [
        metrics1["Annual Volatility (%)"],
        metrics2["Annual Volatility (%)"],
        metrics_bench["Annual Volatility (%)"]
    ]
})
fig_scatter = px.scatter(
    scatter_data,
    x="Annual Volatility (%)",
    y="Annual Return (%)",
    color="Stock",
    text="Stock",
    title="Risk (Volatility) vs Return"
)
fig_scatter.update_traces(textposition="top center")
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("Drawdown Comparison (%)")
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=df1.index, y=df1["drawdown"] * 100, mode="lines", name=ticker1))
fig_dd.add_trace(go.Scatter(x=df2.index, y=df2["drawdown"] * 100, mode="lines", name=ticker2))
fig_dd.add_trace(go.Scatter(x=df_bench.index, y=df_bench["drawdown"] * 100, mode="lines", name=bench_ticker))
fig_dd.update_layout(xaxis_title="Date", yaxis_title="Drawdown (%)")
st.plotly_chart(fig_dd, use_container_width=True)

st.subheader("Rolling Volatility Comparison (%)")
rolling_df = pd.DataFrame(index=df1.index.union(df2.index).union(df_bench.index)).sort_index()
rolling_df[f"{ticker1} Rolling Vol"] = df1["simple_return"].rolling(rolling_window).std() * np.sqrt(252) * 100
rolling_df[f"{ticker2} Rolling Vol"] = df2["simple_return"].rolling(rolling_window).std() * np.sqrt(252) * 100
rolling_df[f"{bench_ticker} Rolling Vol"] = df_bench["simple_return"].rolling(rolling_window).std() * np.sqrt(252) * 100
fig_roll = go.Figure()
for col in rolling_df.columns:
    fig_roll.add_trace(go.Scatter(x=rolling_df.index, y=rolling_df[col], mode="lines", name=col))
fig_roll.update_layout(xaxis_title="Date", yaxis_title="Annualised Rolling Volatility (%)")
st.plotly_chart(fig_roll, use_container_width=True)

# ── Academic Analysis and Evaluation ──────────────────────────────────────────
st.subheader("Academic Data Analysis and Evaluation")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### Plain-Language Investor Interpretation")
    beginner_points = [
        winner_text("Return", metrics1["Annual Return (%)"], metrics2["Annual Return (%)"], higher_is_better=True),
        winner_text("Volatility", metrics1["Annual Volatility (%)"], metrics2["Annual Volatility (%)"], higher_is_better=False),
        winner_text("Sharpe Ratio", metrics1["Sharpe Ratio"], metrics2["Sharpe Ratio"], higher_is_better=True),
        winner_text("Maximum Drawdown", adv1["Max Drawdown (%)"], adv2["Max Drawdown (%)"], higher_is_better=False),
        winner_text("Positive-Day Ratio", adv1["Positive-Day Ratio (%)"], adv2["Positive-Day Ratio (%)"], higher_is_better=True)
    ]
    for point in beginner_points:
        st.write(f"- {point}")

    st.markdown(
        """
        **How to read this section**
        - Higher return is attractive, but not enough on its own.
        - Lower volatility and smaller drawdown indicate a smoother holding experience.
        - A higher Sharpe ratio suggests better reward per unit of risk.
        - A higher positive-day ratio can indicate more consistent short-run performance, although it does not guarantee superior long-run return.
        """
    )

with col_b:
    st.markdown("### Professional Evaluation")
    st.markdown(
        f"""
        This section evaluates the two stocks relative to the benchmark **{bench_ticker}** using both descriptive and risk-adjusted indicators.

        **Return dimension:** Annual return summarises growth over the selected period.  
        **Risk dimension:** Annual volatility and maximum drawdown capture both fluctuation intensity and worst peak-to-trough loss.  
        **Risk-adjusted dimension:** Sharpe ratio evaluates excess reward per unit of total risk.  
        **Market-relative dimension:** Beta, correlation, alpha, and tracking error assess each stock's co-movement and relative behaviour against the benchmark.  

        A rigorous investment assessment should avoid relying on one single metric. A stock may rank first in return but still be unattractive if it achieves that return through materially higher volatility, deeper drawdowns, or weak benchmark-adjusted performance.
        """
    )

st.markdown("### Comparative Evaluation Table")
comparison_eval = pd.DataFrame({
    "Indicator": [
        "Annual Return",
        "Annual Volatility",
        "Sharpe Ratio",
        "Maximum Drawdown",
        "Beta vs Benchmark",
        "Alpha vs Benchmark",
        "Correlation with Benchmark",
        "Tracking Error",
        "Positive-Day Ratio"
    ],
    ticker1: [
        metrics1["Annual Return (%)"],
        metrics1["Annual Volatility (%)"],
        metrics1["Sharpe Ratio"],
        adv1["Max Drawdown (%)"],
        adv1["Beta vs Benchmark"],
        adv1["Alpha vs Benchmark (%)"],
        adv1["Correlation with Benchmark"],
        adv1["Tracking Error (%)"],
        adv1["Positive-Day Ratio (%)"]
    ],
    ticker2: [
        metrics2["Annual Return (%)"],
        metrics2["Annual Volatility (%)"],
        metrics2["Sharpe Ratio"],
        adv2["Max Drawdown (%)"],
        adv2["Beta vs Benchmark"],
        adv2["Alpha vs Benchmark (%)"],
        adv2["Correlation with Benchmark"],
        adv2["Tracking Error (%)"],
        adv2["Positive-Day Ratio (%)"]
    ],
    "Interpretation Focus": [
        "Growth outcome",
        "Total fluctuation risk",
        "Risk-adjusted efficiency",
        "Worst historical loss",
        "Sensitivity to market movements",
        "Return beyond benchmark-adjusted expectation",
        "Degree of co-movement with market",
        "Relative deviation from benchmark",
        "Short-run consistency"
    ]
})
st.dataframe(comparison_eval, use_container_width=True)

st.markdown("### Objective Analytical Conclusion")
return_leader = ticker1 if metrics1["Annual Return (%)"] > metrics2["Annual Return (%)"] else ticker2
risk_leader = ticker1 if metrics1["Annual Volatility (%)"] < metrics2["Annual Volatility (%)"] else ticker2
sharpe_leader = ticker1 if metrics1["Sharpe Ratio"] > metrics2["Sharpe Ratio"] else ticker2
drawdown_leader = ticker1 if adv1["Max Drawdown (%)"] > adv2["Max Drawdown (%)"] else ticker2

st.write(
    f"""
    Over the selected period, **{return_leader}** delivered the stronger annual return, while **{risk_leader}** showed the lower overall volatility.
    In risk-adjusted terms, **{sharpe_leader}** achieved the stronger Sharpe ratio.
    From a downside-risk perspective, **{drawdown_leader}** experienced the smaller maximum drawdown.

    Therefore, the preferred stock depends on the user's decision rule:
    - a **return-seeking investor** may prioritise the stock with the stronger annual return,
    - a **risk-sensitive investor** may prefer the stock with lower volatility and smaller drawdown,
    - a **more professional evaluator** should place stronger weight on Sharpe ratio, alpha, beta, and tracking error rather than headline return alone.

    This multi-metric framework improves the objectivity of the tool because it avoids a one-dimensional judgement and makes the analysis readable for both non-specialist and more advanced investors.
    """
)

# ── Export table ──────────────────────────────────────────────────────────────
st.subheader("Raw Comparison Table")
raw_data = pd.DataFrame({
    f"{ticker1}_Adj_Price": df1["adj_prc"],
    f"{ticker2}_Adj_Price": df2["adj_prc"],
    f"{bench_ticker}_Adj_Price": df_bench["adj_prc"],
    f"{ticker1}_Daily_Return": df1["daily_return"],
    f"{ticker2}_Daily_Return": df2["daily_return"],
    f"{bench_ticker}_Daily_Return": df_bench["daily_return"]
}).round(6)
st.dataframe(raw_data, use_container_width=True)

csv = raw_data.to_csv().encode("utf-8")
st.download_button(
    "Download comparison data as CSV",
    data=csv,
    file_name="stock_comparison.csv",
    mime="text/csv"
)
