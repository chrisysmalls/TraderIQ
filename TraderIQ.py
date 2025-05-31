import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import os
from PIL import Image

# --- PAGE CONFIG (must be first Streamlit command) ---
st.set_page_config(page_title="TraderIQ: MT5 Strategy Optimizer", layout="centered", page_icon="🧠")

# --- LOGO: Safe loading (only if file exists) ---
logo_path = "/mnt/data/TradeIQ.png"
if os.path.exists(logo_path):
    try:
        logo = Image.open(logo_path)
        st.image(logo, width=150)
    except Exception as e:
        st.warning("Logo found but could not be opened.")
else:
    st.info("Upload your logo as TradeIQ.png to show it here.")

st.title("🧠 TraderIQ: MT5 Backtest Analyzer & Optimizer")
st.subheader("Analyze, Optimize, and Export Smarter Bot Settings Automatically.")

# --- File Uploaders (MUST come before usage!) ---
uploaded_csv = st.file_uploader("Step 1: Upload your MT5 Backtest CSV or Report", type=["csv"])
uploaded_set = st.file_uploader(
    "Step 2: Upload your EA's .set or .ini file",
    type=["set", "ini"]
)
st.caption("CSV: Either a trade log or a full MT5 report. .set/.ini: Your bot settings.")

# --- Helper to robustly extract trades table from any MT5 report ---
def extract_trades_from_mt5_report(file):
    file.seek(0)
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

# --- UNIVERSAL .set/.ini parser with robust encoding (handles MT5 .set) ---
def parse_ini_setfile(file):
    file.seek(0)
    content = file.read()
    tried_decodings = []
    for encoding in ("utf-16", "utf-16le", "utf-8", "latin-1"):
        try:
            if isinstance(content, bytes):
                decoded = content.decode(encoding)
            else:
                decoded = content
            # Remove null bytes if present
            if '\x00' in decoded:
                decoded = decoded.replace('\x00', '')
            # Only keep if it actually finds parameter-like lines
            if sum(1 for l in decoded.splitlines() if '=' in l or l.strip().startswith('[')) > 2:
                break
        except Exception:
            tried_decodings.append(encoding)
            continue
    else:
        decoded = content if isinstance(content, str) else ""
    lines = decoded.splitlines()
    sections = {}
    current_section = "Parameters"
    output_lines = []
    sections[current_section] = []
    for line in lines:
        output_lines.append(line)
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            current_section = stripped.strip('[]')
            sections[current_section] = []
        elif '=' in line and not stripped.startswith(";") and stripped != "":
            sections.setdefault(current_section, []).append(line)
    return sections, output_lines

# --- Load and parse CSV, handling both raw and report formats ---
df = None
profit_col = None
profits = None
total_trades = 0
profit_factor = 1.0
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

# --- CSV analysis and metrics ---
if df is not None:
    st.markdown("#### 🐞 CSV DEBUG: Data Preview & Columns")
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

# --- .set/.ini parameter parsing, analysis, and optimizer ---
editable_params = {}
full_output_lines = []
optimized_params = {}

if uploaded_set:
    sections, full_output_lines = parse_ini_setfile(uploaded_set)
    st.markdown("#### 🐞 SET FILE DEBUG: Raw Lines")
    st.code("\n".join(full_output_lines) if full_output_lines else "No lines read from file.")

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
            editable_params[key] = main_val
        st.markdown("---")

    # Show optimization button only if both files uploaded and parsed
    if df is not None and profit_col is not None:
        if st.button("🔍 Analyze & Optimize Settings Automatically"):
            st.success("Optimization complete! Scroll down for the proposed new settings and download.")

            # --- Example Simple Optimization Logic ---
            optimized_params = editable_params.copy()
            messages = []
            if "TakeProfit" in optimized_params:
                try:
                    avg_win = profits[profits > 0].mean()
                    new_tp = round(avg_win, 2)
                    if new_tp > 0:
                        old = float(optimized_params["TakeProfit"])
                        optimized_params["TakeProfit"] = str(new_tp)
                        messages.append(f"- Increased TakeProfit from {old} to {new_tp} (based on average winning trade).")
                except Exception:
                    pass
            if "StopLoss" in optimized_params:
                try:
                    avg_loss = abs(profits[profits < 0].mean())
                    new_sl = round(avg_loss, 2)
                    if new_sl > 0:
                        old = float(optimized_params["StopLoss"])
                        optimized_params["StopLoss"] = str(new_sl)
                        messages.append(f"- Adjusted StopLoss from {old} to {new_sl} (based on average losing trade).")
                except Exception:
                    pass
            if "RiskPercent" in optimized_params:
                try:
                    if profit_factor < 1.5:
                        old = float(optimized_params["RiskPercent"])
                        new_risk = max(0.5, old * 0.75)
                        optimized_params["RiskPercent"] = str(round(new_risk, 2))
                        messages.append(f"- Reduced RiskPercent from {old} to {new_risk} to lower drawdown.")
                except Exception:
                    pass
            if not messages:
                messages.append("No automatic improvements found. Review parameters manually.")

            st.markdown("#### 🛠️ Optimization Suggestions")
            for msg in messages:
                st.write(msg)

            st.markdown("#### 🚀 Optimized Settings (Download Below)")
            # Compose new setfile
            output_lines = []
            for line in full_output_lines:
                if '=' in line and not line.strip().startswith(';'):
                    key = line.split('=', 1)[0].strip()
                    if key in optimized_params:
                        output_lines.append(f"{key}={optimized_params[key]}")
                    else:
                        output_lines.append(line)
                else:
                    output_lines.append(line)
            st.download_button("📥 Download Optimized Set File", "\n".join(output_lines), "TraderIQ_Optimized.set")

# --- Manual editing section as fallback ---
if editable_params and not optimized_params:
    st.markdown("#### Or adjust parameters below manually:")
    for key, main_val in editable_params.items():
        val = st.text_input(key, main_val)
        editable_params[key] = val
    # Download manual edit set file
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
    st.download_button("Download Edited Set File", "\n".join(output_lines), "TraderIQ_ManualEdit.set")

st.markdown("---")
st.caption("TraderIQ: Upload both files, analyze, optimize, and download your improved set file for MT5. For custom rules, just ask!")
