import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import io
import json
import base64
import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="TraderIQ: MT5 Strategy Optimizer", layout="wide", page_icon="🧠")

# --- LOGO ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(BASE_DIR, "TradeIQ.png")
if os.path.exists(logo_path):
    try:
        logo = Image.open(logo_path)
        st.sidebar.image(logo, width=150)
    except Exception:
        st.sidebar.warning("Logo found but could not be opened.")
else:
    st.sidebar.info(f"Logo missing: Place your logo as {logo_path}")

st.title("🧠 TraderIQ: Advanced MT5 Backtest Analyzer & Optimizer")
st.sidebar.markdown("## Upload your MT5 Backtest Data and EA Set File")

# --- FILE UPLOAD ---
uploaded_csv = st.sidebar.file_uploader("Upload MT5 Backtest CSV or Report", type=["csv"])
uploaded_set = st.sidebar.file_uploader("Upload EA Set File (.set/.ini)", type=["set", "ini"])

# --- HELPER FUNCTIONS ---

def decode_file(file):
    content = file.read()
    for encoding in ['utf-8', 'utf-16', 'latin-1']:
        try:
            return content.decode(encoding)
        except Exception:
            continue
    return content.decode('utf-8', errors='replace')

def parse_mt5_report(file):
    """Extract trade table from MT5 report format."""
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
    # Find end of table (empty line or summary)
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
    """Parse .set/.ini file robustly, supporting multiple encodings and sections."""
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
            # comments or blank lines kept in main section to preserve file integrity
            sections.setdefault(current_section, []).append(line)
    return sections, lines

def clean_profit_value(val):
    """Sanitize and convert profit values to float."""
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

def calculate_max_drawdown(profits_series):
    """Calculate max drawdown from equity curve."""
    equity_curve = profits_series.cumsum()
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max)
    max_dd = drawdown.min() if len(drawdown) > 0 else 0
    return abs(max_dd)

def generate_equity_curve_plot(profits_series):
    equity = profits_series.cumsum()
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(equity, label='Equity Curve')
    ax.fill_between(equity.index, equity, equity.cummax(), color='red', alpha=0.3, label='Drawdown')
    ax.set_title("Equity Curve with Drawdown")
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Cumulative Profit")
    ax.legend()
    ax.grid(True)
    return fig

def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))

def optimize_parameters(params, metrics):
    messages = []
    optimized = params.copy()

    # Basic statistics
    avg_win = metrics['avg_win']
    avg_loss = abs(metrics['avg_loss'])
    rr = avg_win / avg_loss if avg_loss > 0 else 2.0
    profit_factor = metrics['profit_factor']
    max_dd = metrics['max_drawdown']
    win_rate = metrics['win_rate']

    # Adjust TakeProfit and StopLoss to target RR ~2 with clamping
    if "TakeProfit" in optimized and "StopLoss" in optimized:
        try:
            old_tp = float(optimized["TakeProfit"])
            old_sl = float(optimized["StopLoss"])

            # Tighten stops if drawdown high or volatility (estimate by max_dd)
            vol_factor = 1.0 if max_dd < 10 else 0.7
            new_sl = clamp(min(old_sl, avg_loss * 1.1) * vol_factor, 2, 500)
            new_tp = clamp(new_sl * 2.0, 5, 1000)

            optimized["TakeProfit"] = str(round(new_tp, 2))
            optimized["StopLoss"] = str(round(new_sl, 2))
            messages.append(f"Set TakeProfit to {new_tp} and StopLoss to {new_sl} to maintain ~2:1 risk-reward considering volatility.")
        except:
            pass

    # RiskPercent adaptive tuning
    if "RiskPercent" in optimized:
        try:
            old_risk = float(optimized["RiskPercent"])
            risk_factor = 1.0
            if max_dd > 15:
                risk_factor *= 0.5
            elif max_dd > 10:
                risk_factor *= 0.7
            if profit_factor < 1.5:
                risk_factor *= 0.7
            new_risk = clamp(old_risk * risk_factor, 0.1, old_risk)
            optimized["RiskPercent"] = str(round(new_risk, 3))
            messages.append(f"Adjusted RiskPercent from {old_risk} to {new_risk} based on drawdown and profit factor.")
        except:
            pass

    # Moving averages smoothing adjustments
    if "MovingAveragePeriodShort" in optimized and "MovingAveragePeriodLong" in optimized:
        try:
            short_ma = int(optimized["MovingAveragePeriodShort"])
            long_ma = int(optimized["MovingAveragePeriodLong"])
            short_ma = clamp(short_ma, 5, 50)
            long_ma = clamp(long_ma, short_ma + 5, 200)
            optimized["MovingAveragePeriodShort"] = str(short_ma)
            optimized["MovingAveragePeriodLong"] = str(long_ma)
            messages.append(f"Set MA periods to Short={short_ma}, Long={long_ma} for noise reduction.")
        except:
            pass

    # RSI tuning
    if "RSIPeriod" in optimized and "RSIOverbought" in optimized and "RSIOversold" in optimized:
        try:
            rsi_period = int(optimized["RSIPeriod"])
            overbought = int(optimized["RSIOverbought"])
            oversold = int(optimized["RSIOversold"])
            rsi_period = clamp(rsi_period, 7, 21)
            overbought = clamp(overbought, 70, 90)
            oversold = clamp(oversold, 10, 30)
            optimized["RSIPeriod"] = str(rsi_period)
            optimized["RSIOverbought"] = str(overbought)
            optimized["RSIOversold"] = str(oversold)
            messages.append(f"Tuned RSI parameters: Period={rsi_period}, Overbought={overbought}, Oversold={oversold}.")
        except:
            pass

    # Overfitting detection heuristic
    try:
        if "TakeProfit" in optimized:
            tp_val = float(optimized["TakeProfit"])
            if tp_val > 5 * avg_win and avg_win > 0:
                messages.append("Warning: TakeProfit unusually high compared to average wins — possible overfitting.")
    except:
        pass

    return optimized, messages

def download_link(content: str, filename: str, link_text: str):
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# --- MAIN APP LOGIC ---

# Parse set file
editable_params = {}
full_output_lines = []
optimized_params = {}
metrics = {}

if uploaded_set:
    sections, full_output_lines = parse_set_file(uploaded_set)
    st.sidebar.markdown("### EA Parameters Detected")
    for sec, lines in sections.items():
        st.sidebar.markdown(f"**[{sec}]**")
        for line in lines:
            if '=' in line and not line.strip().startswith(";"):
                key, val = line.split('=', 1)
                editable_params[key.strip()] = val.split('||')[0].strip()

# Process CSV and calculate metrics
if uploaded_csv:
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

        # Show metrics
        st.subheader("Backtest Metrics")
        st.write(metrics)

        # Equity curve plot
        fig = generate_equity_curve_plot(profits)
        st.pyplot(fig)

# OPTIMIZATION UI
if editable_params and metrics:
    if st.button("Run Advanced Optimization"):
        optimized_params, messages = optimize_parameters(editable_params, metrics)
        st.subheader("Optimization Suggestions & Changes")
        for msg in messages:
            st.write("- " + msg)

        # Display old vs new parameters side by side
        st.subheader("Parameter Comparison")
        param_keys = sorted(set(editable_params.keys()) | set(optimized_params.keys()))
        comp_data = []
        for k in param_keys:
            comp_data.append({
                "Parameter": k,
                "Original": editable_params.get(k, ""),
                "Optimized": optimized_params.get(k, editable_params.get(k, ""))
            })
        st.table(comp_data)

        # Generate new setfile text
        output_lines = []
        for line in full_output_lines:
            if '=' in line and not line.strip().startswith(";"):
                key = line.split('=',1)[0].strip()
                val = optimized_params.get(key, None)
                if val is not None:
                    output_lines.append(f"{key}={val}")
                else:
                    output_lines.append(line)
            else:
                output_lines.append(line)
        new_setfile_text = "\n".join(output_lines)
        st.markdown("### Download Optimized Set File")
        st.markdown(download_link(new_setfile_text, "TraderIQ_Optimized.set", "📥 Click here to download"), unsafe_allow_html=True)

# Manual parameter editing fallback
if editable_params and not optimized_params:
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
    st.markdown("### Download Edited Set File")
    st.markdown(download_link(new_setfile_text, "TraderIQ_ManualEdit.set", "📥 Click here to download"), unsafe_allow_html=True)

st.markdown("---")
st.caption("TraderIQ: Upload backtest CSV/report and EA set files to analyze, optimize, and export improved MT5 settings.")

