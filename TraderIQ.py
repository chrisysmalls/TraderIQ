import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import os
from PIL import Image

# --- PAGE CONFIG (must be the very first Streamlit command) ---
st.set_page_config(page_title="TraderIQ: MT5 Strategy Optimizer", layout="centered", page_icon="🧠")

# --- LOGO: Safe loading (only if file exists) ---
logo_path = "/mnt/data/TradeIQ.png"
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, width=150)
else:
    st.info("Upload your logo as TradeIQ.png to show it here.")

st.title("🧠 TraderIQ: MT5 Backtest Analyzer & Optimizer")
st.subheader("Analyze, Optimize, and Export Smarter Bot Settings Automatically.")

# --- Helper to robustly extract trades table from any MT5 report ---
def extract_trades_from_mt5_report(file):
    file.seek(0)  # Always start at beginning!
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    lines = content.splitlines()
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

# --- Helper for .set/.ini parsing and UI ---
def parse_ini_setfile(file):
    file.seek(0)
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    lines = content.splitlines()
    sections = {}
    current_section = None
    output_lines = []
    for line in lines:
        output_lines.append(line)
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            current_section = stripped.strip('[]')
            sections[current_section] = []
        elif current_section is not None:
            sections[current_section].append(line)
    return sections, output_lines

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
    try:
        uploaded_csv.seek(0)
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

# --- DEBUG SECTION: show preview and columns ---
if df is not None:
    st.markdown("#### 🐞 DEBUG: Data Preview & Columns")
    st.write(df.head())
    st.write("Columns detected:", list(df.columns))

    profit_col = next((c for c in df.columns if "profit" in c.lower()), None)
    if not profit_col:
        st.error("Profit column not found. Please upload a standard MT5 results CSV or report with trade table.")
        st.stop()
    
    def clean_profit(val):
        if pd.isnull(val):
            return np.nan
        val = str(val).replace(" ", "").replace(",", "")
        if val in ['', '-', '--']:
            return 0.0
        if val.startswith('-.'):
            val = '-0.' + val[2:]
        try:
            return float(val)
        except Exception:
            val = val.replace('--', '-')
            try:
                return float(val)
            except:
                return np.nan

    profits = df[profit_col].apply(clean_profit)
    total_trades = len(profits.dropna())
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

    st.subheader("📈 Equity Curve")
    balance = profits.cumsum()
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(balance.values)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Cumulative Profit")
    ax.grid(True)
    st.pyplot(fig)

# --- .set/.ini parameter parsing and download ---
editable_params = {}
full_output_lines = []

if uploaded_set:
    sections, full_output_lines = parse_ini_setfile(uploaded_set)
    st.markdown("### EA Parameters Detected (Edit as Needed)")
    for section, lines in sections.items():
        st.markdown(f"**[{section}]**")
        for line in lines:
            if line.strip().startswith(";") or '=' not in line:
                st.write(line)
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            main_val = value.split('||')[0].strip()
            if main_val.lower() in ['true', 'false']:
                val_bool = True if main_val.lower() == "true" else False
                val = st.checkbox(key, val_bool)
                editable_params[key] = f"{str(val).lower()}{value[len(main_val):]}"
            elif re.match(r'^-?\d+(\.\d+)?$', main_val):
                try:
                    parts = value.split('||')
                    min_val = float(parts[2])
                    max_val = float(parts[3])
                    val_num = st.number_input(key, value=float(main_val), min_value=min_val, max_value=max_val, step=0.1)
                    editable_params[key] = f"{val_num}{'||' + '||'.join(parts[1:]) if len(parts)>1 else ''}"
                except Exception:
                    val_num = st.number_input(key, value=float(main_val))
                    editable_params[key] = str(val_num)
            else:
                val_str = st.text_input(key, main_val)
                editable_params[key] = f"{val_str}{value[len(main_val):]}"
        st.markdown("---")

if editable_params:
    st.success("Download your optimized set/ini file for MT5 below:")
    output_lines = []
    for line in full_output_lines:
        if '=' in line and not line.strip().startswith(';'):
            key = line.split('=',1)[0].strip()
            if key in editable_params:
                output_lines.append(f"{key}={editable_params[key]}")
            else:
                output_lines.append(line)
        else:
            output_lines.append(line)
    st.download_button("📥 Download Optimized Settings File", "\n".join(output_lines), "TraderIQ_Optimized.set")

st.markdown("---")
st.caption("TraderIQ: Debug enabled! See data details above to troubleshoot uploads. If anything is empty, check your input files.")
