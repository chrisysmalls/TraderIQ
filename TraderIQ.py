import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import io
import matplotlib as mpl
import time

# --- 1. SET PAGE CONFIG FIRST ---
st.set_page_config(page_title="TraderIQ: MT5 Strategy Optimizer", layout="wide", page_icon="🧠")

# --- 2. CSS Styling ---
st.markdown("""
<style>
div.stButton > button:first-child {
    font-size: 18px;
    padding: 10px 20px;
    min-width: 200px;
    border-radius: 10px;
}
body, .main, .block-container {
    background-color: #0f111a;
    color: #e0e6f8;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.css-1d391kg, .css-1v3fvcr {
    background: rgba(20, 25, 50, 0.8);
    backdrop-filter: blur(15px);
    border-radius: 12px;
    padding: 20px;
}
h1, h2, h3, h4, h5, h6 {
    color: #68c0ff !important;
    text-shadow: 0 0 8px #68c0ff;
}
input, textarea {
    background-color: #1b2038 !important;
    color: #e0e6f8 !important;
    border-radius: 10px !important;
    border: 1px solid #3a4a75 !important;
    padding: 10px !important;
}
thead tr th {
    background: #1c2a53 !important;
    color: #68c0ff !important;
    font-weight: 700 !important;
    text-align: center !important;
    text-shadow: 0 0 4px #68c0ff;
}
tbody tr {
    background: #0f111a !important;
    border-bottom: 1px solid #2c3e75 !important;
}
tbody tr:hover {
    background: #1a2a59 !important;
}
.stPlotlyChart > div > div > div {
    background-color: #0f111a !important;
}
footer {visibility: hidden;}
.css-1d391kg {
    scrollbar-width: thin;
    scrollbar-color: #68c0ff #0f111a;
}
.css-1d391kg::-webkit-scrollbar {
    width: 8px;
}
.css-1d391kg::-webkit-scrollbar-thumb {
    background-color: #68c0ff;
    border-radius: 10px;
}
.css-1avcm0n {
    color: #6e7caa !important;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# --- 3. Logo + Info panel ---
col1, col2 = st.columns([1, 3])
with col1:
    logo_path = "TradeIQ.png"
    if os.path.exists(logo_path):
        logo_img = Image.open(logo_path)
        st.image(logo_img, width=180)
    else:
        st.warning("Logo file 'TradeIQ.png' not found in the app folder.")
with col2:
    st.markdown("")

# --- 4. File uploaders ---
uploaded_csv = st.sidebar.file_uploader("Upload MT5 Backtest CSV or Report", type=["csv"], key="csv_uploader")
uploaded_set = st.sidebar.file_uploader("Upload EA Set File (.set/.ini)", type=["set", "ini"], key="set_uploader")
st.sidebar.caption("Supported CSVs: trade logs or full MT5 reports. Supported EA files: .set or .ini")

if uploaded_csv is None and uploaded_set is None:
    with col2:
        st.markdown("""
            # Welcome to TraderIQ 🧠
            ### Your MT5 Strategy Optimizer & Analyzer

            Upload your MT5 backtest CSV/report and EA .set/.ini files using the sidebar.
            
            Once uploaded, analyze performance, optimize parameters automatically, and download improved .set files.

            **Features:**
            - Intelligent auto parameter tuning
            - Backtest metrics visualization
            - Manual parameter editing
            - Clean, futuristic UI

            Start by uploading files in the sidebar to the left!
        """)

# --- 5. Helper functions ---
def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))

def clean_profit_value(val):
    try:
        if pd.isna(val):
            return np.nan
        s = str(val).replace(" ", "").replace(",", "")
        if s in ['', '-', '--']:
            return 0.0
        if s.startswith('-.'):
            s = '-0.' + s[2:]
        return float(s)
    except:
        return np.nan

def calculate_max_drawdown(profits_series):
    equity_curve = profits_series.cumsum()
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max)
    max_dd = drawdown.min() if len(drawdown) > 0 else 0
    return abs(max_dd)

def calculate_metrics(profits_series):
    total_trades = profits_series.count()
    wins = profits_series[profits_series > 0]
    losses = profits_series[profits_series < 0]
    win_rate = len(wins) / total_trades * 100 if total_trades else 0
    total_profit = profits_series.sum()
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else float('inf')
    expectancy = ((len(wins) * avg_win) + (len(losses) * avg_loss)) / total_trades if total_trades else 0
    sharpe_ratio = profits_series.mean() / profits_series.std() * np.sqrt(252) if profits_series.std() != 0 else 0
    max_drawdown = calculate_max_drawdown(profits_series)
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def decode_file(file):
    content = file.read()
    for encoding in ['utf-8', 'utf-16', 'latin-1']:
        try:
            return content.decode(encoding)
        except Exception:
            continue
    return content.decode('utf-8', errors='replace')

def parse_mt5_report(file):
    file.seek(0)
    text = decode_file(file)
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if ("Profit" in line) and ("Ticket" in line or "Order" in line):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find trades table header with 'Profit'.")
    end_idx = None
    for i in range(header_idx+1, len(lines)):
        if lines[i].strip() == "" or any(x in lines[i] for x in ["Summary", "Report", "[", "input"]):
            end_idx = i
            break
    if end_idx is None:
        end_idx = len(lines)
    table_lines = lines[header_idx:end_idx]
    df = pd.read_csv(io.StringIO("\n".join(table_lines)))
    return df

def parse_set_file(file):
    file.seek(0)
    raw = file.read()
    for encoding in ['utf-16', 'utf-8', 'latin-1']:
        try:
            if isinstance(raw, bytes):
                content = raw.decode(encoding)
            else:
                content = raw
            if '\x00' in content:
                content = content.replace('\x00', '')
            break
        except Exception:
            continue
    else:
        content = raw.decode('utf-8', errors='replace') if isinstance(raw, bytes) else raw

    lines = content.splitlines()
    sections = {}
    current_section = "Parameters"
    sections[current_section] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped.strip("[]")
            sections[current_section] = []
        elif '=' in line and not stripped.startswith(";"):
            sections[current_section].append(line)
        else:
            sections.setdefault(current_section, []).append(line)
    return sections, lines

def generate_equity_curve_plot(profits_series):
    mpl.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 3))
    equity = profits_series.cumsum()
    trades_count = len(equity)
    ax.plot(equity.index, equity.values, color='#00f0ff', linewidth=1.5, label='Equity Curve')
    ax.fill_between(equity.index, equity.values, equity.cummax(), color='#004466', alpha=0.3, label='Drawdown')
    ax.set_title("Equity Curve with Drawdown", color='#68c0ff', fontsize=14)
    ax.set_xlabel("Trade Number", color='#68c0ff', fontsize=12)
    ax.set_ylabel("Cumulative Profit", color='#68c0ff', fontsize=12)
    ax.tick_params(colors='#68c0ff', labelsize=10)
    ax.legend(facecolor='#0f111a', edgecolor='#68c0ff', labelcolor='#68c0ff', fontsize=10)
    ax.grid(True, color='#1a2a59')
    ax.set_xlim(0, trades_count if trades_count > 0 else 1)
    plt.tight_layout()
    return fig

def generate_optimized_setfile_text(full_output_lines, optimized_params):
    output_lines = []
    for line in full_output_lines:
        stripped = line.strip()
        if '=' in line and not stripped.startswith(';'):
            key = line.split('=', 1)[0].strip()
            if key in optimized_params:
                parts = line.split('=', 1)[1].split('||', 1)
                comment = f"||{parts[1]}" if len(parts) > 1 else ""
                new_line = f"{key}={optimized_params[key]}{comment}"
                output_lines.append(new_line)
            else:
                output_lines.append(line)
        else:
            output_lines.append(line)
    return "\n".join(output_lines)

# --- Advanced optimizer function ---
def advanced_optimizer(editable_params, metrics):
    optimized_params = editable_params.copy()
    messages = []

    max_dd = metrics['max_drawdown']
    profit_factor = metrics['profit_factor']
    win_rate = metrics['win_rate']
    expectancy = metrics['expectancy']
    sharpe = metrics['sharpe_ratio']
    avg_win = metrics['avg_win']
    avg_loss = abs(metrics['avg_loss'])

    # Adjust RiskPercent based on drawdown and profit factor
    if "RiskPercent" in optimized_params:
        try:
            old_risk = float(optimized_params["RiskPercent"])
            risk_factor = 1.0
            if max_dd > 15:
                risk_factor *= 0.5
            elif max_dd > 10:
                risk_factor *= 0.75
            if profit_factor < 1.5:
                risk_factor *= 0.7
            new_risk = clamp(old_risk * risk_factor, 0.1, old_risk)
            optimized_params["RiskPercent"] = str(round(new_risk, 3))
            messages.append(f"Adjusted RiskPercent from {old_risk} to {new_risk} due to drawdown and profit factor.")
        except:
            pass

    # Adjust TakeProfit and StopLoss to maintain good RR ratio and limit drawdown
    if "TakeProfit" in optimized_params and "StopLoss" in optimized_params:
        try:
            old_tp = float(optimized_params["TakeProfit"])
            old_sl = float(optimized_params["StopLoss"])

            vol_factor = 1.0 if max_dd < 10 else 0.7
            new_sl = clamp(min(old_sl, avg_loss * 1.1) * vol_factor, 2, 500)
            rr_ratio = 2.0 if expectancy > 0 else 1.5
            new_tp = clamp(new_sl * rr_ratio, 5, 1000)

            optimized_params["StopLoss"] = str(round(new_sl, 2))
            optimized_params["TakeProfit"] = str(round(new_tp, 2))
            messages.append(f"Set StopLoss to {new_sl} and TakeProfit to {new_tp} to improve reward:risk ratio.")
        except:
            pass

    # Tune RSI parameters if win rate low
    if win_rate < 50 and "RSIPeriod" in optimized_params and "RSIOverbought" in optimized_params and "RSIOversold" in optimized_params:
        try:
            rsi_period = int(optimized_params["RSIPeriod"])
            overbought = int(optimized_params["RSIOverbought"])
            oversold = int(optimized_params["RSIOversold"])

            rsi_period = clamp(rsi_period + 1, 7, 21)
            overbought = clamp(overbought - 2, 70, 90)
            oversold = clamp(oversold + 2, 10, 30)

            optimized_params["RSIPeriod"] = str(rsi_period)
            optimized_params["RSIOverbought"] = str(overbought)
            optimized_params["RSIOversold"] = str(oversold)
            messages.append(f"Tuned RSI parameters for better signal filtering.")
        except:
            pass

    # Adjust Moving Average periods if Sharpe low
    if sharpe < 0.5 and "MovingAveragePeriodShort" in optimized_params and "MovingAveragePeriodLong" in optimized_params:
        try:
            short_ma = int(optimized_params["MovingAveragePeriodShort"])
            long_ma = int(optimized_params["MovingAveragePeriodLong"])

            short_ma = clamp(short_ma + 3, 5, 50)
            long_ma = clamp(long_ma + 5, short_ma + 5, 200)

            optimized_params["MovingAveragePeriodShort"] = str(short_ma)
            optimized_params["MovingAveragePeriodLong"] = str(long_ma)
            messages.append(f"Adjusted MA periods to reduce noise and improve stability.")
        except:
            pass

    return optimized_params, messages

# --- Main logic ---

editable_params = {}
full_output_lines = []
metrics = {}
set_file_loaded = False
df = None

if uploaded_set is not None:
    try:
        sections, full_output_lines = parse_set_file(uploaded_set)
        set_file_loaded = True
    except Exception as e:
        st.error(f"Failed to parse set file: {e}")

if set_file_loaded:
    st.sidebar.markdown("### EA Parameters Detected (Edit if needed)")
    editable_params.clear()
    for sec, lines in sections.items():
        st.sidebar.markdown(f"**[{sec}]**")
        for line in lines:
            if '=' in line and not line.strip().startswith(";"):
                key, val = line.split('=', 1)
                editable_params[key.strip()] = val.split('||')[0].strip()

if uploaded_csv is not None:
    try:
        uploaded_csv.seek(0)
        df = pd.read_csv(uploaded_csv)
        if "Profit" not in df.columns:
            raise Exception("No 'Profit' column found; attempting MT5 report parsing.")
    except Exception:
        uploaded_csv.seek(0)
        try:
            df = parse_mt5_report(uploaded_csv)
            st.success("Extracted trade table from MT5 report.")
        except Exception as e:
            st.error(f"Error parsing CSV/report: {e}")
            df = None

    if df is not None:
        profit_col = next((c for c in df.columns if "profit" in c.lower()), None)
        if profit_col is None:
            st.error("Profit column missing in data.")
            st.stop()
        profits = df[profit_col].apply(clean_profit_value).dropna()
        metrics = calculate_metrics(profits)

        st.subheader("Backtest Metrics")
        st.write(metrics)

        fig = generate_equity_curve_plot(profits)
        st.pyplot(fig)

if uploaded_csv is not None and uploaded_set is not None:
    if st.sidebar.button("🔍 Analyze & Optimize Settings Automatically"):
        st.session_state.optimize_clicked = True

if st.session_state.get("optimize_clicked", False) and editable_params and metrics:
    with st.spinner("Optimizing your settings..."):
        time.sleep(1)
        optimized_params, messages = advanced_optimizer(editable_params, metrics)

        st.subheader("Optimization Suggestions & Changes")
        for msg in messages:
            st.write("- " + msg)

        st.subheader("Parameter Comparison")
        keys = sorted(set(editable_params.keys()) | set(optimized_params.keys()))
        comp_data = []
        for k in keys:
            comp_data.append({
                "Parameter": k,
                "Original": editable_params.get(k, ""),
                "Optimized": optimized_params.get(k, editable_params.get(k, ""))
            })
        st.table(comp_data)

        new_setfile_text = generate_optimized_setfile_text(full_output_lines, optimized_params)

        st.markdown("### Download Optimized Set File")
        st.download_button(
            label="📥 Download Updated .set File",
            data=new_setfile_text,
            file_name="TraderIQ_Optimized.set",
            mime="text/plain"
        )

# Manual editing fallback
if editable_params and not st.session_state.get("optimize_clicked", False):
    st.subheader("Manual Parameter Editor")
    for key, val in editable_params.items():
        new_val = st.text_input(key, val)
        editable_params[key] = new_val

    output_lines = []
    for line in full_output_lines:
        if '=' in line and not line.strip().startswith(";"):
            key = line.split('=',1)[0].strip()
            val = editable_params.get(key, None)
            if val is not None:
                output_lines.append(f"{key}={val}")
            else:
                output_lines.append(line)
        else:
            output_lines.append(line)
    new_setfile_text = "\n".join(output_lines)
    st.markdown("### Download Edited .set File")
    st.download_button(
        label="📥 Download Edited .set File",
        data=new_setfile_text,
        file_name="TraderIQ_ManualEdit.set",
        mime="text/plain"
    )

st.markdown("---")
st.caption("TraderIQ: Upload backtest CSV/report and EA set files to analyze, optimize, and export improved MT5 settings.")
