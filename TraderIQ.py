import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="TraderIQ: MT5 Strategy Optimizer", layout="centered", page_icon="🧠")

st.title("🧠 TraderIQ: MT5 Backtest Analyzer & Optimizer")
st.subheader("Analyze, Optimize, and Export Smarter Bot Settings Automatically.")

# --- Helper to robustly extract trades table from any MT5 report ---
def extract_trades_from_mt5_report(file):
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    lines = content.splitlines()
    # Flexible: look for header row that has Profit and (Ticket or Order)
    trade_table_start = None
    for idx, line in enumerate(lines):
        if "Profit" in line and ("Ticket" in line or "Order" in line):
            trade_table_start = idx
            break
    if trade_table_start is None:
        for idx, line in enumerate(lines):
            if "Profit" in line:
                trade_table_start = idx
                break
    if trade_table_start is None:
        raise ValueError("Could not find trades table header (with 'Profit') in uploaded file.")

    # Find table end (blank line or next section/summary)
    trade_table_end = None
    for idx in range(trade_table_start + 1, len(lines)):
        if lines[idx].strip() == "" or any(x in lines[idx] for x in ["Summary", "Report", "[", "input"]):
            trade_table_end = idx
            break
    if trade_table_end is None:
        trade_table_end = len(lines)

    table_lines = lines[trade_table_start:trade_table_end]
    from io import StringIO
    trades_df = pd.read_csv(StringIO("\n".join(table_lines)))
    return trades_df

# --- File Uploaders ---
uploaded_csv = st.file_uploader("Step 1: Upload your MT5 Backtest CSV or Report", type=["csv"])
uploaded_set = st.file_uploader(
    "Step 2: Upload your EA's .set or .ini file", 
    type=["set", "ini"]
)
st.caption("CSV: Either a trade log or a full MT5 report. .set/.ini: Your bot settings.")

# --- Load and parse CSV, handling both raw and report formats ---
df = None
if uploaded_csv:
    # Try to read as plain trade log first
    try:
        df = pd.read_csv(uploaded_csv)
        if "Profit" not in df.columns:
            raise Exception("No 'Profit' column found; attempting report extract...")
    except Exception:
        uploaded_csv.seek(0)
        try:
            df = extract_trades_from_mt5_report(uploaded_csv)
            st.success("Extracted trade table from report!")
        except Exception as e:
            st.error(f"Failed to extract trades from uploaded CSV/report: {e}")
            df = None

# --- CSV Analysis (only if df loaded) ---
if df is not None:
    profit_col = next((c for c in df.columns if "profit" in c.lower()), None)
    if not profit_col:
        st.error("Profit column not found. Please upload a standard MT5 results CSV or report with trade table.")
        st.stop()
    # Ensure 'Profit' is numeric!
    profits = pd.to_numeric(df[profit_col], errors="coerce")
    total_trades = len(profits)
    win_rate = (profits > 0).sum() / total_trades * 100 if total_trades else 0
    total_profit = profits.sum()
    avg_win = profits[profits > 0].mean() if (profits > 0).sum() else 0
    avg_loss = profits[profits < 0].mean() if (profits < 0).sum() else 0
    profit_factor = profits[profits > 0].sum() / abs(profits[profits < 0].sum()) if (profits < 0).sum() else float('inf')
    expectancy = ((profits > 0).sum() * avg_win + (profits < 0).sum() * avg_loss) / total_trades if total_trades else 0

    st.markdown(f"""
    ### Backtest Metrics:
    - **Total Trades:** {total_trades}
    - **Win Rate:** {win_rate:.2f}%
    - **Total Profit:** {total_profit:.2f}
    - **Profit Factor:** {profit_factor:.2f}
    - **Expectancy:** {expectancy:.2f}
    """)

    # Show equity curve
    st.subheader("📈 Equity Curve")
    balance = profits.cumsum()
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(balance.values)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Cumulative Profit")
    ax.grid(True)
    st.pyplot(fig)

# --- .set/.ini parameter parsing and download (add this block below for bot parameter editing) ---
# [Insert your existing .set/.ini parameter code block here if needed.]

st.markdown("---")
st.caption("TraderIQ: Now supports raw CSVs or full MT5 report files for ultimate flexibility!")
