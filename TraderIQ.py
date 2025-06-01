import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import io
import os
import time
from PIL import Image
import textwrap
from bs4 import BeautifulSoup

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(
    page_title="TraderIQ 🤖 AI-Powered MT5 Optimizer",
    layout="wide",
    page_icon="🤖"
)

st.markdown("""
<style>
body, .main, .block-container {
    background-color: #0b0e1d;
    color: #cfdcff;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    color: #68c0ff !important;
    text-shadow: 0 0 6px #68c0ff;
}
div.stButton > button:first-child {
    font-size: 18px;
    padding: 12px 30px;
    min-width: 220px;
    border-radius: 10px;
    font-weight: 700;
    background: linear-gradient(90deg, #0ff 0%, #68f 100%);
    color: #001f3f;
}
input, textarea {
    background-color: #1b2038 !important;
    color: #cfdcff !important;
    border-radius: 8px !important;
    border: 1px solid #3a4a75 !important;
    padding: 8px !important;
}
.stTable thead tr th {
    background: #142345 !important;
    color: #68c0ff !important;
    font-weight: 700 !important;
    text-align: center !important;
}
.stTable tbody tr {
    background: #0b0e1d !important;
    border-bottom: 1px solid #243a78 !important;
}
.stTable tbody tr:hover {
    background: #1a2c54 !important;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    if os.path.exists("TradeIQ.png"):
        st.image("TradeIQ.png", width=160)
    else:
        st.warning("Logo `TradeIQ.png` not found.")
with col2:
    st.markdown("# TraderIQ 🤖 AI-Powered MT5 Optimizer")
    st.markdown("Upload your backtest CSV or HTML and EA file on the left to begin super-intelligent optimization.")

# --- 2. FILE UPLOADS ---
st.sidebar.markdown(
    "🛈 <b>For <span style='color:#68c0ff'>MT5</span>, please upload the <span style='color:#68c0ff'>HTML report</span> of your backtest.<br>"
    "(Right-click in the Strategy Tester → Save as Report)</b>",
    unsafe_allow_html=True
)
uploaded_report = st.sidebar.file_uploader(
    "Upload MT5 Backtest CSV or HTML Report", type=["csv", "html"]
)
uploaded_set = st.sidebar.file_uploader("Upload EA", type=["set", "ini"])
st.sidebar.caption("CSV or HTML: MT5 trade log or full report. EA: .set or .ini")

if (uploaded_report is None) and (uploaded_set is None):
    st.info("Waiting for both files…\n• Upload backtest CSV/HTML report\n• Upload EA file")

# --- 3. HELPERS ---

def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))

def clean_profit(val):
    try:
        if pd.isna(val):
            return np.nan
        s = str(val).replace(",", "").strip()
        if s in ["", "-", "--"]:
            return 0.0
        if s.startswith("-."):
            s = "-0." + s[2:]
        return float(s)
    except:
        return np.nan

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

def extract_deals_table_from_mt5_html(html_bytes):
    soup = BeautifulSoup(html_bytes, "html.parser")
    tables = soup.find_all("table")
    deals_table = None

    for table in tables:
        if table.find(string="Deals"):
            deals_table = table
            break

    if deals_table is None:
        return None

    rows = deals_table.find_all("tr")
    for i, row in enumerate(rows):
        ths = row.find_all("th")
        if ths and any("Deal" in th.get_text() for th in ths):
            header_row = i
            break
    else:
        return None

    header = [th.get_text(strip=True) for th in rows[header_row].find_all(["th", "td"])]
    data = []
    for row in rows[header_row + 1:]:
        tds = row.find_all("td")
        if len(tds) == len(header):
            data.append([td.get_text(strip=True).replace('\xa0', ' ') for td in tds])

    df = pd.DataFrame(data, columns=header)
    if "Balance" in df.columns:
        df["Balance"] = df["Balance"].astype(str).str.replace(" ", "").str.replace(",", "").astype(float, errors="ignore")
    if "Profit" in df.columns:
        df["Profit"] = df["Profit"].astype(str).str.replace(" ", "").str.replace(",", "").astype(float, errors="ignore")
    return df

def parse_mt5_report(file):
    file.seek(0)
    raw = file.read()
    try:
        text = raw.decode("utf-8")
    except:
        text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if ("Profit" in line) and ("Ticket" in line or "Order" in line):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Cannot find `Profit` header.")
    end_idx = None
    for i in range(header_idx+1, len(lines)):
        if lines[i].strip() == "" or any(tok in lines[i] for tok in ["Summary", "Report", "[", "input"]):
            end_idx = i
            break
    if end_idx is None:
        end_idx = len(lines)
    table_text = "\n".join(lines[header_idx:end_idx])
    return pd.read_csv(io.StringIO(table_text))

def parse_set_file(file):
    file.seek(0)
    raw = file.read()
    for enc in ("utf-16", "utf-16le", "utf-8", "latin-1"):
        try:
            if isinstance(raw, bytes):
                content = raw.decode(enc)
            else:
                content = raw
            if "\x00" in content:
                content = content.replace("\x00", "")
            if "=" in content:
                break
        except:
            continue
    else:
        content = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw

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
    # Example heuristics:
    # 1) Dynamic Risk
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
    # 2) TP/SL Adjustment
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
    # 3) RSI Tuning
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
    # 4) MA Filter
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

# --- 4. MAIN LOGIC ---

editable_params = {}
full_output_lines = []
metrics = {}
df = None
parsed = False

# 4a) Parse `.set`
if uploaded_set:
    try:
        sections, full_output_lines = parse_set_file(uploaded_set)
        editable_params.clear()
        for sec, lines in sections.items():
            for line in lines:
                if "=" in line and not line.strip().startswith(";"):
                    k, v = line.split("=", 1)
                    editable_params[k.strip()] = v.split("||")[0].strip()
        st.subheader("📄 EA Set File Preview")
        st.code("\n".join(full_output_lines), language="ini")
    except Exception as e:
        st.error(f"Error parsing setfile: {e}")

# 4b) Parse CSV or HTML & show metrics/chart
df = None
if uploaded_report:
    filetype = os.path.splitext(uploaded_report.name)[1].lower()
    if filetype == ".html":
        html_bytes = uploaded_report.read()
        df = extract_deals_table_from_mt5_html(html_bytes)
        if df is not None and not df.empty:
            st.success("Extracted 'Deals' table from MT5 HTML report.")
            st.download_button(
                label="Download Cleaned Deals Table as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="MT5_Deals_Table.csv",
                mime="text/csv"
            )
        else:
            st.error("Could not find a valid 'Deals' table in the HTML report.")
            df = None
    else:
        uploaded_report.seek(0)
        try:
            df = pd.read_csv(uploaded_report)
        except:
            try:
                df = parse_mt5_report(uploaded_report)
                st.success("Extracted trades from MT5 report.")
            except Exception as e:
                st.error(f"Error parsing CSV/report: {e}")
                df = None

    if df is not None:
        profit_col = next((c for c in df.columns if "profit" in c.lower()), None)
        if not profit_col:
            st.error("`Profit` column not found.")
            st.stop()
        profits = df[profit_col].apply(clean_profit).dropna()
        metrics = calculate_advanced_metrics(profits)

        st.subheader("📊 Backtest Metrics")
        st.write(metrics)

        st.subheader("📈 Equity Curve")
        fig = plt.figure(figsize=(6, 3))
        eq = profits.cumsum()
        plt.plot(eq.index, eq.values, color="#00ffff", linewidth=2, label="Equity Curve")
        plt.fill_between(eq.index, eq.values, eq.cummax(), color="#004466", alpha=0.4, label="Drawdown")
        plt.title("Equity Curve with Drawdown", color="#68c0ff", fontsize=14)
        plt.xlabel("Trade #", color="#68c0ff")
        plt.ylabel("Cum. Profit", color="#68c0ff")
        plt.tick_params(colors="#68c0ff")
        plt.legend(facecolor="#0b0e1d", edgecolor="#68c0ff", labelcolor="#68c0ff")
        plt.grid(True, color="#1f2e5e")
        plt.tight_layout()
        st.pyplot(fig)

# 4c) If both files uploaded, show optimizer results
if uploaded_set and uploaded_report and metrics and editable_params:
    st.markdown("---")
    st.header("🚀 AI-Powered Optimization Results")
    with st.spinner("Running super-intelligent optimizer..."):
        time.sleep(1)
        opt_params, heuristic_msgs = super_intelligent_optimizer(editable_params, metrics)

    st.subheader("⚙️ Heuristic Adjustments")
    for m in heuristic_msgs:
        st.write(f"- {m}")

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

st.markdown("---")
st.caption("TraderIQ: AI-Driven MT5 Strategy Optimizer. Upload backtest CSV/HTML and EA `.set`/`.ini` to get started.")
