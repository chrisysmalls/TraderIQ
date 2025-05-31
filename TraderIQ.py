import numpy as np

if df is not None:
    st.markdown("#### 🐞 DEBUG: Data Preview & Columns")
    st.write(df.head())
    st.write("Columns detected:", list(df.columns))

    profit_col = next((c for c in df.columns if "profit" in c.lower()), None)
    if not profit_col:
        st.error("Profit column not found. Please upload a standard MT5 results CSV or report with trade table.")
        st.stop()
    
    def clean_profit(val):
        if pd.isnull(val):
            return np.nan
        # Remove spaces, handle negatives, and commas
        val = str(val).replace(" ", "").replace(",", "")
        if val in ['', '-', '--']:
            return 0.0
        if val.startswith('-.'):
            val = '-0.' + val[2:]
        try:
            return float(val)
        except Exception:
            # Try to fix any odd dash issues
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
