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
        df["Balance"] = df["Balance"].str.replace(" ", "").str.replace(",", "").astype(float, errors="ignore")
    if "Profit" in df.columns:
        df["Profit"] = df["Profit"].str.replace(" ", "").str.replace(",", "").astype(float, errors="ignore")
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

# --- 4. MAIN LOGIC ---

editable_params = {}
full_output_lines = []
metrics = {}
df = None
parsed = False

# 4a) Parse `.set` (now previews the setfile)
if uploaded_set:
    try:
        uploaded_set.seek(0)
        raw = uploaded_set.read()
        # Try common encodings
        for enc in ("utf-16", "utf-16le", "utf-8", "latin-1"):
            try:
                content = raw.decode(enc)
                break
            except:
                continue
        else:
            content = raw.decode("utf-8", errors="replace")
        st.subheader("📄 EA Set File Preview")
        st.code(content, language="ini")
    except Exception as e:
        st.error(f"Error reading setfile: {e}")

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

st.markdown("---")
st.caption("TraderIQ: AI-Driven MT5 Strategy Optimizer. Upload backtest CSV/HTML and EA `.set`/`.ini` to get started.")
