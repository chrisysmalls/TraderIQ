import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import time
import textwrap
from PIL import Image
from bs4 import BeautifulSoup
import re

# --- PAGE CONFIG & THEME ---
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

# --- SIDEBAR ---
st.sidebar.markdown("## Start Here")
uploaded_set = st.sidebar.file_uploader("Upload EA (.set or .ini)", type=["set", "ini"])
uploaded_report = st.sidebar.file_uploader("Upload Backtest CSV or HTML", type=["csv", "html"])

st.markdown("# TraderIQ 🤖 AI-Powered MT5 Optimizer")
st.markdown("Upload your backtest CSV or HTML and EA file on the left to begin super-intelligent optimization.")

editable_params = {}
full_output_lines = []
if uploaded_set:
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
st.markdown("## 2️⃣ Step 2: Backtest Results (MT5)")

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

if uploaded_report:
    filetype = os.path.splitext(uploaded_report.name)[1].lower()
    if filetype == ".html":
        html_bytes = uploaded_report.read()
        soup = BeautifulSoup(html_bytes, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            st.warning("❗️ No tables found in HTML report.")
        else:
            st.info(f"Found {len(tables)} table(s) in your HTML report. Previewing them all below.")
            table_options = []
            dfs = []
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
                if data:
                    df_try = pd.DataFrame(data, columns=headers)
                    dfs.append(df_try)
                    table_label = f"Table {idx+1}: {headers}"
                    table_options.append(table_label)
            if dfs:
                selected_idx = st.selectbox(
                    "Select which table looks like your Deals/trade history table:",
                    range(len(dfs)), format_func=lambda i: table_options[i])
                df = dfs[selected_idx]
                st.subheader(f"Preview of selected table (first 5 rows):")
                st.write(df.head())
            else:
                st.warning("Tables found, but none with data rows matching the header.")
    else:
        uploaded_report.seek(0)
        try:
            df = pd.read_csv(uploaded_report)
            st.write(df.head())
        except Exception as e:
            st.error("Error parsing CSV report: " + str(e))
            df = None

# ---- NUMERIC COLUMN LOGIC/DIAGNOSE ----
if df is not None and not df.empty:
    numeric_cols = []
    for col in df.columns:
        cleaned_col = df[col].apply(to_float)
        if cleaned_col.notna().sum() > 0:
            numeric_cols.append(col)
            df[col + " (num)"] = cleaned_col
    st.subheader("Columns with detected numbers:")
    st.write(numeric_cols)
    if numeric_cols:
        profit_col = st.selectbox(
            "Select the profit/result column for metrics calculation:",
            options=numeric_cols
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
    else:
        st.warning("❗️ No numeric columns found at all. Please check your report format or select another table.")

st.markdown("---")
st.caption("TraderIQ: AI-Driven MT5 Strategy Optimizer. Upload backtest CSV/HTML and EA `.set`/`.ini` to get started.")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import time
import textwrap
from PIL import Image
from bs4 import BeautifulSoup
import re

# --- PAGE CONFIG & THEME ---
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

# --- SIDEBAR ---
st.sidebar.markdown("## Start Here")
uploaded_set = st.sidebar.file_uploader("Upload EA (.set or .ini)", type=["set", "ini"])
uploaded_report = st.sidebar.file_uploader("Upload Backtest CSV or HTML", type=["csv", "html"])

st.markdown("# TraderIQ 🤖 AI-Powered MT5 Optimizer")
st.markdown("Upload your backtest CSV or HTML and EA file on the left to begin super-intelligent optimization.")

editable_params = {}
full_output_lines = []
if uploaded_set:
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
st.markdown("## 2️⃣ Step 2: Backtest Results (MT5)")

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

if uploaded_report:
    filetype = os.path.splitext(uploaded_report.name)[1].lower()
    if filetype == ".html":
        html_bytes = uploaded_report.read()
        soup = BeautifulSoup(html_bytes, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            st.warning("❗️ No tables found in HTML report.")
        else:
            st.info(f"Found {len(tables)} table(s) in your HTML report. Previewing them all below.")
            table_options = []
            dfs = []
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
                if data:
                    df_try = pd.DataFrame(data, columns=headers)
                    dfs.append(df_try)
                    table_label = f"Table {idx+1}: {headers}"
                    table_options.append(table_label)
            if dfs:
                selected_idx = st.selectbox(
                    "Select which table looks like your Deals/trade history table:",
                    range(len(dfs)), format_func=lambda i: table_options[i])
                df = dfs[selected_idx]
                st.subheader(f"Preview of selected table (first 5 rows):")
                st.write(df.head())
            else:
                st.warning("Tables found, but none with data rows matching the header.")
    else:
        uploaded_report.seek(0)
        try:
            df = pd.read_csv(uploaded_report)
            st.write(df.head())
        except Exception as e:
            st.error("Error parsing CSV report: " + str(e))
            df = None

# ---- NUMERIC COLUMN LOGIC/DIAGNOSE ----
if df is not None and not df.empty:
    numeric_cols = []
    for col in df.columns:
        cleaned_col = df[col].apply(to_float)
        if cleaned_col.notna().sum() > 0:
            numeric_cols.append(col)
            df[col + " (num)"] = cleaned_col
    st.subheader("Columns with detected numbers:")
    st.write(numeric_cols)
    if numeric_cols:
        profit_col = st.selectbox(
            "Select the profit/result column for metrics calculation:",
            options=numeric_cols
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
    else:
        st.warning("❗️ No numeric columns found at all. Please check your report format or select another table.")

st.markdown("---")
st.caption("TraderIQ: AI-Driven MT5 Strategy Optimizer. Upload backtest CSV/HTML and EA `.set`/`.ini` to get started.")
