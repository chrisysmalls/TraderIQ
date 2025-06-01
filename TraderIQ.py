import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import textwrap
from PIL import Image
from bs4 import BeautifulSoup
import re

# Optional OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Must be first Streamlit command!
st.set_page_config(
    page_title="TraderIQ 🤖 AI-Powered MT5 Optimizer",
    layout="wide",
    page_icon="🤖"
)

st.markdown("""
<style>
body, .main, .block-container {background-color: #0b0e1d; color: #cfdcff;}
h1, h2, h3, h4, h5, h6 {color: #68c0ff !important;}
div.stButton > button:first-child {font-size: 18px; padding: 12px 30px; min-width: 220px; border-radius: 10px; font-weight: 700; background: linear-gradient(90deg, #0ff 0%, #68f 100%); color: #001f3f;}
input, textarea {background-color: #1b2038 !important; color: #cfdcff !important; border-radius: 8px !important; border: 1px solid #3a4a75 !important; padding: 8px !important;}
.stTable thead tr th {background: #142345 !important; color: #68c0ff !important;}
.stTable tbody tr {background: #0b0e1d !important; border-bottom: 1px solid #243a78 !important;}
.stTable tbody tr:hover {background: #1a2c54 !important;}
footer {visibility: hidden;}
hr {border: 1px solid #243a78;}
</style>
""", unsafe_allow_html=True)

# Helper function to deduplicate DataFrame columns
def deduplicate_columns(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_idx = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dup_idx):
            if i == 0:
                continue  # Keep first as is
            cols[idx] = f"{dup}_{i}"
    df.columns = cols
    return df

# --- LOGO ROW (top of page) ---
logo_col, title_col = st.columns([1, 6])
with logo_col:
    if os.path.exists("TradeIQ.png"):
        st.image("TradeIQ.png", width=130)
    else:
        st.write("")
with title_col:
    st.markdown("# TraderIQ 🤖 AI-Powered MT5 Optimizer")
    st.caption("Upload your backtest CSV/HTML and EA file on the left to begin.")

# --- SIDEBAR ---
st.sidebar.markdown("## Start Here")
uploaded_set = st.sidebar.file_uploader("Upload EA (.set or .ini)", type=["set", "ini"])
uploaded_report = st.sidebar.file_uploader("Upload Backtest CSV or HTML", type=["csv", "html"])

# --- STEP 1: EA SETFILE ---
st.markdown("## 1️⃣ Step 1: EA Setfile (Upload & Preview)")
editable_params = {}
full_output_lines = []

def parse_set_file(file):
    file.seek(0)
    raw = file.read()
    for enc in ("utf-16", "utf-16le", "utf-8", "latin-1"):
        try:
            content = raw.decode(enc)
            break
        except:
            continue
    else:
        content = raw.decode("utf-8", errors="replace")
    lines = content.splitlines()
    sections = {}
    current = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped.strip("[]")
            sections[current] = []
        elif current is not None:
            sections.setdefault(current, []).append(line)
        else:
            current = "Parameters"
            sections.setdefault(current, []).append(line)
    return sections, lines

if uploaded_set:
    try:
        sections, full_output_lines = parse_set_file(uploaded_set)
        editable_params.clear()
        for sec, lines in sections.items():
            for line in lines:
                if "=" in line and not line.strip().startswith(";"):
                    k, v = line.split("=", 1)
                    editable_params[k.strip()] = v.split("||")[0].strip()
        st.code("\n".join(full_output_lines), language="ini")
        st.success("Setfile uploaded and parsed.")
    except Exception as e:
        st.error(f"Error parsing setfile: {e}")
else:
    st.info("Upload your EA setfile (.set or .ini) in the sidebar.")

st.markdown("---")

# --- STEP 2: BACKTEST RESULTS ---
st.markdown("## 2️⃣ Step 2: Backtest Report (Upload, Table Pick, Profit Column Pick)")
metrics = {}
df = None

def to_float(val):
    if isinstance(val, str):
        cleaned = re.sub(r'[^\d\-,\.]', '', val)
        try:
            if ',' in cleaned and '.' not in cleaned:
                cleaned = cleaned.replace(',', '.')
            return float(cleaned)
        except:
            return np.nan
    try:
        return float(val)
    except:
        return np.nan

profit_col = None
numeric_cols = []
profits = None
if uploaded_report:
    filetype = os.path.splitext(uploaded_report.name)[1].lower()
    if filetype == ".html":
        html_bytes = uploaded_report.read()
        soup = BeautifulSoup(html_bytes, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            st.warning("❗️ No tables found in HTML report.")
        else:
            # Only keep tables with 3+ columns (likely real trade tables)
            dfs = []
            table_options = []
            for idx, table in enumerate(tables):
                rows = table.find_all("tr")
                if not rows:
                    continue
                headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
                data = []
                for row in rows[1:]:
                    tds = row.find_all("td")
                    if len(tds) == len(headers):
                        data.append([td.get_text(strip=True) for td in tds])
                # Skip empty, single-col, or "report info" tables
                if data and len(headers) >= 3:
                    df_try = pd.DataFrame(data, columns=headers)
                    df_try = deduplicate_columns(df_try)  # Deduplicate columns here
                    dfs.append(df_try)
                    table_label = f"Table {idx+1}: {headers}"
                    table_options.append(table_label)
            if not dfs:
                st.warning("No valid trade tables found in your report.")
            else:
                st.info(f"Found {len(dfs)} valid table(s) in your HTML report.")
                selected_idx = st.selectbox(
                    "Select which table looks like your Deals/trade history table:",
                    range(len(dfs)), format_func=lambda i: table_options[i])
                df = dfs[selected_idx]
                st.subheader("Preview of selected table (first 5 rows):")
                st.write(df.head())
    else:
        uploaded_report.seek(0)
        try:
            df = pd.read_csv(uploaded_report)
            df = deduplicate_columns(df)  # Deduplicate columns here too
            st.write(df.head())
        except Exception as e:
            st.error("Error parsing CSV report: " + str(e))
            df = None

    # Numeric column auto-detect and metrics
    if df is not None and not df.empty:
        numeric_cols = []
        for col in df.columns:
            cleaned_col = df[col].apply(to_float)
            if cleaned_col.notna().sum() > 0:
                numeric_cols.append(col)
                df[col + " (num)"] = cleaned_col

        profit_col = None
        profit_names = ["profit", "p/l", "result", "pnl"]
        for col in numeric_cols:
            for name in profit_names:
                if name in col.lower():
                    profit_col = col
                    break
            if profit_col:
                break
        if not profit_col and numeric_cols:
            profit_col = numeric_cols[0]

        if numeric_cols:
            profit_col = st.selectbox(
                "Select the profit/result column for metrics calculation:",
                options=numeric_cols,
                index=numeric_cols.index(profit_col) if profit_col in numeric_cols else 0
            )
            profits = df[profit_col].apply(to_float).dropna()
            if profits.empty:
                st.warning("⚠️ The selected column does not contain usable numbers for profit calculation. Please try another column above.")
            else:
                def calculate_advanced_metrics(profits):
                    equity = profits.cumsum()
                    total_trades = len(profits.dropna())
                    wins = profits[profits > 0]
                    losses = profits[profits < 0]
                    win_rate = (len(wins) / total_trades * 100) if total_trades else 0
                    total_profit = profits.sum()
                    avg_win = wins.mean() if not wins.empty else 0.0
                    avg_loss = losses.mean() if not losses.empty else 0.0
                    profit_factor = (wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else float('inf')
                    expectancy = ((len(wins)*avg_win) + (len(losses)*avg_loss)) / total_trades if total_trades else 0
                    sharpe = (profits.mean() / profits.std() * np.sqrt(252)) if profits.std() != 0 else 0
                    roll_max = equity.cummax()
                    drawdowns = equity - roll_max
                    max_dd = abs(drawdowns.min()) if not drawdowns.empty else 0.0
                    volatility = profits.std() * np.sqrt(252) if profits.std() != 0 else 0.0
                    streak = 0
                    max_consec = 0
                    for p in profits:
                        if p < 0:
                            streak += 1
                            max_consec = max(max_consec, streak)
                        else:
                            streak = 0
                    return {
                        "total_trades": total_trades,
                        "win_rate": round(win_rate, 2),
                        "total_profit": round(total_profit, 2),
                        "avg_win": round(avg_win, 2),
                        "avg_loss": round(avg_loss, 2),
                        "profit_factor": round(profit_factor, 2),
                        "expectancy": round(expectancy, 2),
                        "sharpe_ratio": round(sharpe, 2),
                        "max_drawdown": round(max_dd, 2),
                        "volatility_annualized": round(volatility, 4),
                        "max_consecutive_losses": max_consec
                    }
                metrics = calculate_advanced_metrics(profits)
                st.subheader("📊 Backtest Metrics")
                st.write(metrics)

                # Equity curve with clear deep blue and teal drawdown
                st.subheader("📈 Equity Curve")
                fig, ax = plt.subplots(figsize=(6, 3), facecolor="#0b0e1d")
                eq = profits.cumsum()
                if len(eq) > 1:
                    ax.plot(eq.index, eq.values, color="#1853a0", linewidth=3, label="Equity Curve", zorder=3)
                    ax.fill_between(eq.index, eq.values, eq.cummax(), color="#22e6c5", alpha=0.25, label="Drawdown", zorder=2)
                else:
                    ax.hlines(eq.values[0], eq.index[0], eq.index[-1], color="#1853a0", linewidth=3, label="Equity Curve", zorder=3)
                ax.set_facecolor("#0b0e1d")
                ax.set_title("Equity Curve with Drawdown", color="#68c0ff", fontsize=14)
                ax.set_xlabel("Trade #", color="#68c0ff")
                ax.set_ylabel("Cum. Profit", color="#68c0ff")
                ax.tick_params(colors="#68c0ff")
                ax.legend(facecolor="#0b0e1d", edgecolor="#68c0ff", labelcolor="#68c0ff")
                ax.grid(True, color="#1f2e5e", linestyle="--", linewidth=0.6, alpha=0.8)
                fig.tight_layout()
                st.pyplot(fig)
else:
    st.info("Upload your MT5 backtest HTML/CSV in the sidebar.")

st.markdown("---")

# --- STEP 3: AI OPTIMIZATION & DOWNLOAD ---
st.markdown("## 3️⃣ Step 3: AI Optimization & Download")

def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))

def super_intelligent_optimizer(params, metrics):
    optimized = params.copy()
    messages = []
    dd = metrics["max_drawdown"]
    pf = metrics["profit_factor"]
    wr = metrics["win_rate"]
    expv = metrics["expectancy"]
    sr = metrics["sharpe_ratio"]
    vol = metrics["volatility_annualized"]
    streak = metrics["max_consecutive_losses"]
    avg_w = metrics["avg_win"]
    avg_l = abs(metrics["avg_loss"])
    if "RiskPercent" in optimized:
        try:
            r_old = float(optimized["RiskPercent"])
            rf = 1.0
            if dd > 20 or vol > 0.06:
                rf *= 0.4
            elif dd > 10 or vol > 0.04:
                rf *= 0.7
            if pf < 1.5:
                rf *= 0.6
            new_r = clamp(r_old * rf, 0.05, r_old)
            optimized["RiskPercent"] = str(round(new_r, 3))
            messages.append(f"RiskPercent → {new_r} (based on drawdown/volatility).")
        except:
            pass
    if "TakeProfit" in optimized and "StopLoss" in optimized:
        try:
            tp_old = float(optimized["TakeProfit"])
            sl_old = float(optimized["StopLoss"])
            vf = 1.0 if dd < 10 else 0.6
            new_sl = clamp(min(sl_old, avg_l * 1.2) * vf, 2, 500)
            base_rr = clamp(1.5 + (expv / 100) + (wr / 100), 1.5, 3.0)
            new_tp = clamp(new_sl * base_rr, 5, 1000)
            optimized["StopLoss"] = str(round(new_sl, 2))
            optimized["TakeProfit"] = str(round(new_tp, 2))
            messages.append(f"StopLoss → {new_sl}, TakeProfit → {new_tp} (RR={round(base_rr,2)}).")
        except:
            pass
    if all(k in optimized for k in ["RSI_Period", "RSI_Overbought", "RSI_Oversold"]):
        try:
            rspi = int(float(optimized["RSI_Period"]))
            ob = int(float(optimized["RSI_Overbought"]))
            osv = int(float(optimized["RSI_Oversold"]))
            if sr < 0.6 or streak > 3:
                rspi = clamp(rspi + 2, 7, 21)
                ob = clamp(ob - 5, 70, 90)
                osv = clamp(osv + 5, 10, 30)
            else:
                ob = clamp(ob + 3, 70, 95)
                osv = clamp(osv - 3, 5, 30)
            optimized["RSI_Period"] = str(rspi)
            optimized["RSI_Overbought"] = str(ob)
            optimized["RSI_Oversold"] = str(osv)
            messages.append(f"RSI → Period {rspi}, OB {ob}, OS {osv}.")
        except:
            pass
    if all(k in optimized for k in ["MovingAveragePeriodShort", "MovingAveragePeriodLong"]):
        try:
            sma = int(float(optimized["MovingAveragePeriodShort"]))
            lma = int(float(optimized["MovingAveragePeriodLong"]))
            if sr < 0.8:
                sma = clamp(sma + 5, 5, 50)
                lma = clamp(lma + 10, sma + 5, 200)
            optimized["MovingAveragePeriodShort"] = str(sma)
            optimized["MovingAveragePeriodLong"] = str(lma)
            messages.append(f"MA → Short {sma}, Long {lma}.")
        except:
            pass
    return optimized, messages

def generate_optimized_setfile_text(full_lines, optimized_params):
    out = []
    for line in full_lines:
        stripped = line.strip()
        if "=" in line and not stripped.startswith(";"):
            key = line.split("=", 1)[0].strip()
            if key in optimized_params:
                parts = line.split("=", 1)[1].split("||", 1)
                comment = f"||{parts[1]}" if len(parts) > 1 else ""
                out.append(f"{key}={optimized_params[key]}{comment}")
            else:
                out.append(line)
        else:
            out.append(line)
    return "\n".join(out)

if profits is not None and not profits.empty and metrics and uploaded_set and editable_params:
    with st.spinner("Running super-intelligent optimizer..."):
        time.sleep(1)
        opt_params, heuristic_msgs = super_intelligent_optimizer(editable_params, metrics)

    st.subheader("⚙️ Heuristic Adjustments")
    for m in heuristic_msgs:
        st.write(f"- {m}")

    st.subheader("🧠 GPT-Powered Suggestions")
    gpt_advice = ""
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        prompt = f"""
You are an MT5 strategy optimization expert.
Current EA parameters:
{ {k: opt_params[k] for k in opt_params} }

Backtest metrics:
- Trades: {metrics['total_trades']}
- Win Rate: {metrics['win_rate']}%
- Total Profit: {metrics['total_profit']}
- Avg Win: {metrics['avg_win']}, Avg Loss: {metrics['avg_loss']}
- Profit Factor: {metrics['profit_factor']}
- Expectancy: {metrics['expectancy']}
- Sharpe: {metrics['sharpe_ratio']}
- Max Drawdown: {metrics['max_drawdown']}
- Volatility: {metrics['volatility_annualized']}
- Max Consecutive Losses: {metrics['max_consecutive_losses']}

Give concise bullet suggestions for further tuning. Only refer to the above parameters.
"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a world-class MT5 optimizer AI."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            gpt_advice = response.choices[0].message.content.strip()
        except Exception as e:
            gpt_advice = f"(GPT advice failed: {e})"
    else:
        gpt_advice = "(AI suggestions are optional. Add your OpenAI API key to unlock GPT-powered tips!)"
    st.info(textwrap.fill(gpt_advice, width=90))

    st.subheader("📋 Parameter Comparison")
    comp_list = []
    keys = sorted(set(editable_params.keys()) | set(opt_params.keys()))
    for k in keys:
        comp_list.append({
            "Parameter": k,
            "Original": editable_params.get(k, ""),
            "Optimized": opt_params.get(k, editable_params.get(k, ""))
        })
    st.table(comp_list)

    new_text = generate_optimized_setfile_text(full_output_lines, opt_params)
    st.markdown("### 📥 Download Your AI-Optimized `.set` File")
    st.download_button(
        label="Download AI-Optimized Set File",
        data=new_text,
        file_name="TraderIQ_AI_Optimized.set",
        mime="text/plain"
    )
else:
    st.info("Upload your EA setfile (.set/.ini) and pick a profit/result column above to unlock AI optimization and downloads.")

st.markdown("---")
st.caption("TraderIQ: AI-Driven MT5 Strategy Optimizer. Upload backtest CSV/HTML and EA `.set`/`.ini` to get started.")
