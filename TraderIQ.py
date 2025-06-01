import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import io
import matplotlib as mpl
import time
import textwrap

# --- Page config ---
st.set_page_config(page_title="TraderIQ: AI-Powered MT5 Strategy Optimizer", layout="wide", page_icon="🤖")

# --- CSS Styling for polished dark futuristic UI ---
st.markdown("""
<style>
div.stButton > button:first-child {
    font-size: 18px;
    padding: 12px 30px;
    min-width: 220px;
    border-radius: 12px;
    font-weight: 700;
    background: linear-gradient(90deg, #0ff 0%, #68f 100%);
    color: #001f3f;
}
body, .main, .block-container {
    background-color: #0c1225;
    color: #a0b9ff;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    color: #76b0f1 !important;
    text-shadow: 0 0 6px #76b0f1;
}
.stMarkdown, .stText, .stTable {
    color: #cfdcff !important;
}
.stTable thead tr th {
    background: #142345 !important;
    color: #76b0f1 !important;
    font-weight: 700 !important;
    text-align: center !important;
}
.stTable tbody tr {
    background: #0c1225 !important;
    border-bottom: 1px solid #243a78 !important;
}
.stTable tbody tr:hover {
    background: #1f2d5f !important;
}
</style>
""", unsafe_allow_html=True)

# --- Load logo ---
col1, col2 = st.columns([1, 3])
with col1:
    logo_path = "TradeIQ.png"
    if os.path.exists(logo_path):
        logo_img = Image.open(logo_path)
        st.image(logo_img, width=180)
    else:
        st.warning("Logo file 'TradeIQ.png' not found in app folder.")

with col2:
    st.markdown("# 🤖 TraderIQ AI-Powered MT5 Strategy Optimizer")
    st.markdown("Upload your backtest CSV and EA .set/.ini files on the left to begin an advanced AI analysis and optimization session.")

# --- File upload ---
uploaded_csv = st.sidebar.file_uploader("Upload MT5 Backtest CSV or Report", type=["csv"])
uploaded_set = st.sidebar.file_uploader("Upload EA .set or .ini File", type=["set","ini"])

# --- Helper functions ---
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

def calculate_max_drawdown(equity):
    roll_max = equity.cummax()
    drawdowns = equity - roll_max
    return abs(drawdowns.min())

def advanced_metrics(profits):
    equity = profits.cumsum()
    max_dd = calculate_max_drawdown(equity)
    total_trades = profits.count()
    wins = profits[profits > 0]
    losses = profits[profits < 0]
    win_rate = len(wins) / total_trades * 100 if total_trades else 0
    total_profit = profits.sum()
    avg_win = wins.mean() if len(wins) else 0
    avg_loss = losses.mean() if len(losses) else 0
    profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else float('inf')
    expectancy = ((len(wins) * avg_win) + (len(losses) * avg_loss)) / total_trades if total_trades else 0
    sharpe = profits.mean() / profits.std() * np.sqrt(252) if profits.std() != 0 else 0
    volatility = profits.std() * np.sqrt(252)
    # Additional: Calculate consecutive losses max streak:
    streak = 0
    max_streak = 0
    for p in profits:
        if p < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'volatility_annualized': volatility,
        'max_consecutive_losses': max_streak
    }

def decode_file(file):
    content = file.read()
    for enc in ['utf-8','utf-16','latin-1']:
        try:
            return content.decode(enc)
        except:
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
        raise ValueError("Could not find trades table header.")
    end_idx = None
    for i in range(header_idx+1, len(lines)):
        if lines[i].strip() == "" or any(x in lines[i] for x in ["Summary","Report","[","input"]):
            end_idx = i
            break
    if end_idx is None:
        end_idx = len(lines)
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:end_idx])))
    return df

def parse_set_file(file):
    file.seek(0)
    raw = file.read()
    for enc in ['utf-16','utf-8','latin-1']:
        try:
            if isinstance(raw, bytes):
                content = raw.decode(enc)
            else:
                content = raw
            if '\x00' in content:
                content = content.replace('\x00','')
            break
        except:
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

def generate_equity_curve_plot(profits):
    mpl.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6,3))
    equity = profits.cumsum()
    ax.plot(equity.index, equity.values, color='#00ffff', linewidth=2, label='Equity Curve')
    ax.fill_between(equity.index, equity.values, equity.cummax(), color='#004466', alpha=0.3, label='Drawdown')
    ax.set_title("Equity Curve with Drawdown", color='#76b0f1', fontsize=14)
    ax.set_xlabel("Trade #", color='#76b0f1')
    ax.set_ylabel("Cumulative Profit", color='#76b0f1')
    ax.tick_params(colors='#76b0f1')
    ax.legend(facecolor='#0c1225', edgecolor='#76b0f1', labelcolor='#76b0f1')
    ax.grid(True, color='#1f2e5e')
    plt.tight_layout()
    return fig

def generate_optimized_setfile_text(full_output_lines, optimized_params):
    output_lines = []
    for line in full_output_lines:
        stripped = line.strip()
        if '=' in line and not stripped.startswith(';'):
            key = line.split('=', 1)[0].strip()
            if key in optimized_params:
                parts = line.split('=', 1)[1].split('||',1)
                comment = f"||{parts[1]}" if len(parts)>1 else ""
                new_line = f"{key}={optimized_params[key]}{comment}"
                output_lines.append(new_line)
            else:
                output_lines.append(line)
        else:
            output_lines.append(line)
    return "\n".join(output_lines)

def super_intelligent_optimizer(editable_params, metrics):
    optimized_params = editable_params.copy()
    messages = []

    max_dd = metrics['max_drawdown']
    profit_factor = metrics['profit_factor']
    win_rate = metrics['win_rate']
    expectancy = metrics['expectancy']
    sharpe = metrics['sharpe_ratio']
    volatility = metrics['volatility_annualized']
    max_streak = metrics['max_consecutive_losses']
    avg_win = metrics['avg_win']
    avg_loss = abs(metrics['avg_loss'])

    # Dynamic Risk Management: Lower risk if drawdown high or volatility high
    if "RiskPercent" in optimized_params:
        try:
            old_risk = float(optimized_params["RiskPercent"])
            risk_factor = 1.0
            if max_dd > 20 or volatility > 0.05:
                risk_factor *= 0.4
            elif max_dd > 10 or volatility > 0.03:
                risk_factor *= 0.7
            if profit_factor < 1.5:
                risk_factor *= 0.6
            new_risk = clamp(old_risk * risk_factor, 0.05, old_risk)
            optimized_params["RiskPercent"] = str(round(new_risk, 3))
            messages.append(f"Adaptive RiskPercent set to {new_risk} based on drawdown and volatility.")
        except:
            pass

    # Reward:Risk Adjustments with expectancy feedback
    if "TakeProfit" in optimized_params and "StopLoss" in optimized_params:
        try:
            old_tp = float(optimized_params["TakeProfit"])
            old_sl = float(optimized_params["StopLoss"])

            vol_factor = 1.0 if max_dd < 10 else 0.6
            new_sl = clamp(min(old_sl, avg_loss * 1.2) * vol_factor, 2, 500)

            # Adjust RR ratio by expectancy and win rate (min 1.5, max 3)
            base_rr = clamp(1.5 + (expectancy / 100) + (win_rate / 100), 1.5, 3.0)
            new_tp = clamp(new_sl * base_rr, 5, 1000)

            optimized_params["StopLoss"] = str(round(new_sl, 2))
            optimized_params["TakeProfit"] = str(round(new_tp, 2))
            messages.append(f"TakeProfit set to {new_tp} and StopLoss to {new_sl} based on expectancy and win rate.")
        except:
            pass

    # RSI and Momentum filter tuning
    if "RSIPeriod" in optimized_params and "RSIOverbought" in optimized_params and "RSIOversold" in optimized_params:
        try:
            rsi_period = int(optimized_params["RSIPeriod"])
            ob = int(optimized_params["RSIOverbought"])
            os = int(optimized_params["RSIOversold"])

            # Increase period and tighten bands if sharpe low or max streak high
            if sharpe < 0.5 or max_streak > 3:
                rsi_period = clamp(rsi_period + 2, 7, 21)
                ob = clamp(ob - 5, 70, 90)
                os = clamp(os + 5, 10, 30)
            else:
                # Relax bands a bit for more trades if sharpe high
                ob = clamp(ob + 3, 70, 95)
                os = clamp(os - 3, 5, 30)

            optimized_params["RSIPeriod"] = str(rsi_period)
            optimized_params["RSIOverbought"] = str(ob)
            optimized_params["RSIOversold"] = str(os)
            messages.append(f"RSI tuned: Period {rsi_period}, Overbought {ob}, Oversold {os}.")
        except:
            pass

    # Moving Average noise filter adjustments
    if "MovingAveragePeriodShort" in optimized_params and "MovingAveragePeriodLong" in optimized_params:
        try:
            short_ma = int(optimized_params["MovingAveragePeriodShort"])
            long_ma = int(optimized_params["MovingAveragePeriodLong"])

            if sharpe < 0.7:
                short_ma = clamp(short_ma + 5, 5, 50)
                long_ma = clamp(long_ma + 10, short_ma + 5, 200)
            else:
                short_ma = clamp(short_ma, 5, 50)
                long_ma = clamp(long_ma, short_ma + 5, 200)

            optimized_params["MovingAveragePeriodShort"] = str(short_ma)
            optimized_params["MovingAveragePeriodLong"] = str(long_ma)
            messages.append(f"MA periods adjusted: Short {short_ma}, Long {long_ma}.")
        except:
            pass

    # Add additional AI-driven heuristics here if you want to expand

    # Provide holistic summary advice in natural language
    summary = f"""
    Based on your backtest:
    - Total trades: {metrics['total_trades']}
    - Win rate: {win_rate:.2f}%
    - Profit factor: {profit_factor:.2f}
    - Max drawdown: {max_dd:.2f}
    - Sharpe ratio: {sharpe:.2f}
    - Volatility (annualized): {volatility:.4f}
    - Max consecutive losses: {max_streak}

    Suggested parameter adjustments aim to optimize reward:risk ratio,
    reduce drawdown and volatility exposure, and tune momentum filters for higher stability.
    """

    return optimized_params, messages, summary

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
        metrics = advanced_metrics(profits)

        st.subheader("Backtest Metrics")
        st.write(metrics)

        fig = generate_equity_curve_plot(profits)
        st.pyplot(fig)

if uploaded_csv is not None and uploaded_set is not None:
    if st.sidebar.button("🤖 Analyze & Optimize Settings Automatically"):
        st.session_state.optimize_clicked = True

if st.session_state.get("optimize_clicked", False) and editable_params and metrics:
    with st.spinner("Running AI-powered optimization..."):
        time.sleep(1)
        optimized_params, messages, summary = super_intelligent_optimizer(editable_params, metrics)

        st.subheader("🤖 AI Optimization Summary")
        st.info(textwrap.fill(summary, width=80))

        st.subheader("⚙️ Optimization Suggestions & Changes")
        for msg in messages:
            st.write("- " + msg)

        st.subheader("📊 Parameter Comparison")
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

        st.markdown("### Download Your AI-Optimized Set File")
        st.download_button(
            label="📥 Download Optimized .set File",
            data=new_setfile_text,
            file_name="TraderIQ_AI_Optimized.set",
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
st.caption("TraderIQ: AI-driven MT5 Strategy Optimizer — Upload your backtest CSV and EA .set/.ini files to get started.")
