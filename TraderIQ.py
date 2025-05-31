import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import os
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="TraderIQ: MT5 Strategy Optimizer", layout="centered", page_icon="🧠")

# --- DEBUG INFO FOR LOGO PATH ---
st.write("Current working directory:", os.getcwd())
st.write("Script absolute path:", os.path.abspath(__file__))
logo_path = "TradeIQ.png"
st.write("Looking for logo at:", os.path.abspath(logo_path))
st.write("Logo file exists:", os.path.exists(logo_path))

# --- LOGO: Load from same folder as script ---
if os.path.exists(logo_path):
    try:
        logo = Image.open(logo_path)
        st.image(logo, width=150)
    except Exception:
        st.warning("Logo found on disk but could not be opened.")
else:
    st.info("Logo missing: Place your logo file as TradeIQ.png in the same folder as this script.")

st.title("🧠 TraderIQ: MT5 Backtest Analyzer & Optimizer")
st.subheader("Analyze, Optimize, and Export Smarter Bot Settings Automatically.")

# --- File Uploaders (MUST come before usage!) ---
uploaded_csv = st.file_uploader("Step 1: Upload your MT5 Backtest CSV or Report", type=["csv"])
uploaded_set = st.file_uploader(
    "Step 2: Upload your EA's .set or .ini file",
    type=["set", "ini"]
)
st.caption("CSV: Either a trade log or a full MT5 report. .set/.ini: Your bot settings.")

# --- rest of your existing code unchanged ---
# ... (Keep all the rest of your code as is, starting from extract_trades_from_mt5_report)
