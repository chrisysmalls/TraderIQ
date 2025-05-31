import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

st.set_page_config(page_title="TraderIQ: The Ultimate MT5 Strategy Optimizer", layout="centered", page_icon="🧠")

# --- Sidebar / Branding ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4725/4725123.png", width=60)
    st.title("TraderIQ")
    st.markdown("**The Ultimate Trading Feedback Loop.**")
    st.info("""
    - Upload your MT5 CSV backtest and bot .set file.
    - Analyze, optimize, and export improved settings!
    """)
    st.download_button("📥 Sample CSV", "Ticket,Open Time,Type,Volume,Symbol,Price,S/L,T/P,Close Time,Close Price,Commission,Swap,Profit\n1,2024-01-01 10:00,buy,0.1,EURUSD,1.1000,1.0950,1.1050,2024-01-01 11:00,1.1025,-0.2,0,25\n2,2024-01-01 12:00,sell,0.1,EURUSD,1.1030,1.1080,1.0980,2024-01-01 13:00,1.1005,-0.2,0,25\n", "sample_mt5.csv")
    st.download_button("📥 Sample .set", "StopLoss=40\nTakeProfit=60\nRiskPerTrade=2.0\nMA_Period=14\n", "sample_bot.set")
    st.markdown("---")
    st.markdown("**Contact:** support@traderiq.com\n**Feedback:** [Google Form Link Here]")

# --- File Uploads ---
st.title("🧠 TraderIQ: The Ultimate MT5 Backtest Analyzer & Optimizer")
st.subheader("Analyze | Optimize | Export | Improve")

uploaded_csv = st.file_uploader("Step 1: Upload your MT5 Backtest CSV", type=["csv"])
uploaded_set = st.file_uploader("Step 2: Upload your MT5 EA's .set file", type=["set"])

session_history = st.session_state.setdefault("history", [])

set_params = {}
if uploaded_set:
    set_lines = uploaded_set.getvalue().decode("utf-8").splitlines()
    for line in set_lines:
        if "=" in line:
            k, v = line.split("=")
            k = k.strip()
            try:
                v = float(v.strip())
            except:
                v = v.strip()
            set_params[k] = v

if uploaded_csv and set_params:
    df = pd.read_csv(uploaded_csv, encoding='utf-8')
    profit_col = next((c for c in df.columns if "profit" in c.lower()), None)
    if not profit_col:
        st.error("Profit column not found. Please upload a standard MT5 results CSV.")
        st.stop()

    # --- 1. Key Metrics ---
    profits = df[profit_col]
    total_trades = len(profits)
    win_rate = (profits > 0).sum() / total_trades * 100 if total_trades else 0
    total_profit = profits.sum()
    avg_win = profits[profits > 0].mean() if (profits > 0).sum() else 0
    avg_loss = profits[profits < 0].mean() if (profits < 0).sum() else 0
    profit_factor = profits[profits > 0].sum() / abs(profits[profits < 0].sum()) if (profits < 0).sum() else float('inf')
    expectancy = ((profits > 0).sum() * avg_win + (profits < 0).sum() * avg_loss) / total_trades if total_trades else 0

    metrics = {
        "Total Trades": total_trades,
        "Win Rate (%)": f"{win_rate:.2f}",
        "Total Profit": f"{total_profit:.2f}",
        "Average Win": f"{avg_win:.2f}",
        "Average Loss": f"{avg_loss:.2f}",
        "Profit Factor": f"{profit_factor:.2f}",
        "Expectancy": f"{expectancy:.2f}"
    }

    with st.expander("🔑 Key Metrics", expanded=True):
        for k, v in metrics.items():
            st.markdown(f"- **{k}:** {v}")

    # --- 2. Parameter Detection & Editing ---
    st.markdown("### EA Parameters Detected")
    editable_params = {}
    param_suggestions = {}
    for k, v in set_params.items():
        if isinstance(v, float) or isinstance(v, int):
            new_val = st.number_input(f"{k}", value=v, key=k)
            editable_params[k] = new_val
            # Example suggestion: Tighten SL, reduce risk, lengthen MA, etc.
            if "stop" in k.lower():
                if new_val > 10:
                    param_suggestions[k] = new_val - 10
            if "risk" in k.lower() or "lot" in k.lower():
                if new_val > 1:
                    param_suggestions[k] = max(1, new_val - 0.5)
        else:
            editable_params[k] = st.text_input(f"{k}", value=str(v), key=k)

    # --- 3. Smart Suggestions ---
    st.markdown("### 🧠 TraderIQ Smart Suggestions")
    advice = []
    if win_rate < 40:
        advice.append("🔴 **Low win rate:** Try improving entry filters or exit logic.")
    if profit_factor < 1.2:
        advice.append("🟠 **Low profit factor:** Consider bigger TPs, smaller SLs, or filtering out bad trades.")
    if avg_loss < -avg_win:
        advice.append("🟡 **Average loss larger than win:** Try lowering stop loss or trailing stops.")
    if expectancy < 0:
        advice.append("🔴 **Negative expectancy:** Change your risk or improve strategy logic.")
    if not advice:
        advice.append("✅ Your strategy looks strong. Keep monitoring and tweaking!")

    for line in advice:
        st.markdown(line)
    if param_suggestions:
        st.markdown("**Parameter Optimization Suggestions:**")
        for k, v in param_suggestions.items():
            st.markdown(f"- Try `{k}` = **{v}**")

    # --- 4. Simulated Optimization (Demo) ---
    st.markdown("### Simulated Optimized Metrics")
    # Demo: Improving stop loss or risk slightly improves profit
    opt_profit = total_profit
    for k, v in param_suggestions.items():
        opt_profit += 10
    st.markdown(f"- **Estimated Profit after Optimization:** `{opt_profit:.2f}`")

    # --- 5. .set File Download ---
    st.success("Download your improved setfile for MT5 below:")
    new_set = "\n".join([f"{k}={editable_params[k]}" for k in editable_params])
    st.download_button("📥 Download Optimized .set File", new_set, "TraderIQ_Optimized.set")

    # --- 6. Charts & Visualization ---
    st.markdown("---")
    st.subheader("📈 Equity Curve & Distribution")
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    balance = profits.cumsum()
    ax[0].plot(balance.values)
    ax[0].set_title("Equity Curve")
    ax[1].hist(profits, bins=30)
    ax[1].set_title("Trade Profit Distribution")
    st.pyplot(fig)

    # --- 7. Session History ---
    if st.button("Save This Run To Session History"):
        session_history.append({
            "metrics": metrics,
            "editable_params": editable_params,
            "opt_profit": opt_profit
        })
        st.success("Run added to session! See below for comparison.")

    if session_history:
        st.subheader("📊 Session Comparison")
        for i, run in enumerate(session_history):
            st.markdown(f"#### Run {i+1}:")
            st.json(run)

    # --- 8. Download Full Report ---
    report_text = "\n".join([f"{k}: {v}" for k,v in metrics.items()]) + "\n" + "\n".join(advice)
    st.download_button("Download TXT Report", report_text, "TraderIQ_Report.txt")

st.markdown("---")
st.caption("Made with ❤️ by TraderIQ. Ready for Pro? Ask about cloud sync, Telegram bots, AI optimization and more!")
