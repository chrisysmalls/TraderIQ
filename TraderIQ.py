# --- .set/.ini parameter parsing and download ---
editable_params = {}
full_output_lines = []
optimized_params = {}

if uploaded_set:
    sections, full_output_lines = parse_ini_setfile(uploaded_set)
    st.markdown("#### 🐞 SET FILE DEBUG: Raw Lines")
    st.code("\n".join(full_output_lines) if full_output_lines else "No lines read from file.")

    st.markdown("### EA Parameters Detected (Edit as Needed)")
    for section, lines in sections.items():
        st.markdown(f"**[{section}]**")
        for line in lines:
            if line.strip().startswith(";") or '=' not in line:
                st.write(line)
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            main_val = value.split('||')[0].strip()
            editable_params[key] = main_val  # Store for use in optimizer UI
        st.markdown("---")

    # Show optimization button only if both files uploaded
    if df is not None:
        if st.button("🔍 Analyze & Optimize Settings Automatically"):
            st.success("Optimization complete! Scroll down for the proposed new settings and download.")

            # --- Example Simple Optimization Logic ---
            # (You can make this as complex as you like! This is a basic example)
            optimized_params = editable_params.copy()
            messages = []
            if "TakeProfit" in optimized_params:
                try:
                    avg_win = df[df[profit_col].astype(float) > 0][profit_col].astype(float).mean()
                    new_tp = round(avg_win, 2)
                    if new_tp > 0:
                        old = float(optimized_params["TakeProfit"])
                        optimized_params["TakeProfit"] = str(new_tp)
                        messages.append(f"- Increased TakeProfit from {old} to {new_tp} (based on average winning trade).")
                except Exception:
                    pass
            if "StopLoss" in optimized_params:
                try:
                    avg_loss = abs(df[df[profit_col].astype(float) < 0][profit_col].astype(float).mean())
                    new_sl = round(avg_loss, 2)
                    if new_sl > 0:
                        old = float(optimized_params["StopLoss"])
                        optimized_params["StopLoss"] = str(new_sl)
                        messages.append(f"- Adjusted StopLoss from {old} to {new_sl} (based on average losing trade).")
                except Exception:
                    pass
            if "RiskPercent" in optimized_params:
                try:
                    if profit_factor < 1.5:
                        old = float(optimized_params["RiskPercent"])
                        new_risk = max(0.5, old * 0.75)
                        optimized_params["RiskPercent"] = str(round(new_risk, 2))
                        messages.append(f"- Reduced RiskPercent from {old} to {new_risk} to lower drawdown.")
                except Exception:
                    pass
            # Add more rules as you wish!
            if not messages:
                messages.append("No automatic improvements found. Review parameters manually.")

            st.markdown("#### 🛠️ Optimization Suggestions")
            for msg in messages:
                st.write(msg)

            st.markdown("#### 🚀 Optimized Settings (Download Below)")
            # Compose new setfile
            output_lines = []
            for line in full_output_lines:
                if '=' in line and not line.strip().startswith(';'):
                    key = line.split('=', 1)[0].strip()
                    if key in optimized_params:
                        output_lines.append(f"{key}={optimized_params[key]}")
                    else:
                        output_lines.append(line)
                else:
                    output_lines.append(line)
            st.download_button("📥 Download Optimized Set File", "\n".join(output_lines), "TraderIQ_Optimized.set")

# Regular manual editing stays below (or you can hide it if optimizing)
if editable_params and not optimized_params:
    st.markdown("#### Or adjust parameters below manually:")
    for key, main_val in editable_params.items():
        val = st.text_input(key, main_val)
        editable_params[key] = val
    # Download original editable set file
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
    st.download_button("Download Edited Set File", "\n".join(output_lines), "TraderIQ_ManualEdit.set")
