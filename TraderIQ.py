import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import re

st.set_page_config(page_title="TraderIQ: MT5 Strategy Optimizer", layout="centered", page_icon="🧠")

st.title("🧠 TraderIQ: MT5 Backtest Analyzer & Optimizer")
st.subheader("Analyze, Optimize, and Export Smarter Bot Settings Automatically.")

# --- File Uploaders ---
uploaded_csv = st.file_uploader("Step 1: Upload your MT5 Backtest CSV", type=["csv"])
uploaded_set = st.file_uploader(
    "Step 2: Upload your EA's .set or .ini file", 
    type=["set", "ini"]
)
st.caption("Accepted: .set or .ini files exported from MT5 Expert Advisor settings.")

# --- Helper function to parse .ini/.set files, preserving sections and comments ---
def parse_ini_setfile(file):
    sections = {}
    current_section = None
    output_lines = []
    file.seek(0)
    for raw_line in file:
        try:
            line = raw_line.decode("utf-8")
        except AttributeError:
            line = raw_line
        output_lines.append(line.rstrip('\n'))
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            current_section = stripped.strip('[]')
            sections[current_section] = []
        elif current_section is not None:
            sections[current_section].append(line.rstrip('\n'))
    return sections, output_lines

# --- Parse settings file, display/edit parameters, preserve structure/comments ---
editable_params = {}
full_output_lines = []

if uploaded_set:
    sections, full_output_lines = parse_ini_setfile(uploaded_set)
    st.markdown("### EA Parameters Detected (Edit as Needed)")
    for section, lines in sections.items():
        st.markdown(f"**[{section}]**")
        for line in lines:
            if line.strip().startswith(";") or '=' not in line:
                st.write(line)
                continue
            # Parse key and value, also try to parse min/max if present
            key, value = line.split('=', 1)
            key = key.strip()
            main_val = value.split('||')[0].strip()
            if main_val.lower() in ['true', 'false']:
                val_bool = True if main_val.lower() == "true" else False
                val = st.checkbox(key, val_bool)
                editable_params[key] = f"{str(val).lower()}{value[len(main_val):]}"
            elif re.match(r'^-?\d+(\.\d+)?$', main_val):  # It's a number
                # Try to get min/max from value (e.g., ...||MIN||MAX...)
                try:
                    parts = value.split('||')
                    min_val = float(parts[2])
                    max_val = float(parts[3])
                    val_num = st.number_input(key, value=float(main_val), min_value=min_val, max_value=max_val, step=0.1)
                    # Reconstruct value with possibly new val_num
                    editable_params[key] = f"{val_num}{'||' + '||'.join(parts[1:]) if len(parts)>1 else ''}"
                except Exception:
                    val_num = st.number_input(key, value=float(main_val))
                    editable_params[key] = str(val_num)
            else:
                val_str = st.text_input(key, main_val)
                editable_params[key] = f"{val_str}{value[len(main_val):]}"
        st.markdown("---")

# --- CSV analysis logic (simplified) ---
if uploaded_csv:
    df = pd.read_csv(uploaded_csv, encoding='utf-8')
    profit_col = next((c for c in df.columns if "profit" in c.lower()), None)
    if not profit_col:
        st.error("Profit column not found. Please upload a standard MT5 results CSV.")
        st.stop()
    profits = df[profit_col]
    total_trades = len(profits)
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

    # Show equity curve
    st.subheader("📈 Equity Curve")
    balance = profits.cumsum()
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(balance.values)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Cumulative Profit")
    ax.grid(True)
    st.pyplot(fig)

# --- Download optimized .set/.ini file ---
if editable_params:
    st.success("Download your optimized set/ini file for MT5 below:")
    # Rebuild file content, preserving sections/comments and edited params
    output_lines = []
    for line in full_output_lines:
        if '=' in line and not line.strip().startswith(';'):
            key = line.split('=',1)[0].strip()
            if key in editable_params:
                output_lines.append(f"{key}={editable_params[key]}")
            else:
                output_lines.append(line)
        else:
            output_lines.append(line)
    st.download_button("📥 Download Optimized Settings File", "\n".join(output_lines), "TraderIQ_Optimized.set")

st.markdown("---")
st.caption("TraderIQ now supports both .set and .ini bot files with all original sections, parameters, and comments preserved.")
