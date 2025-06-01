import re

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

        # --- Editable parameter section ---
        # Parse lines that look like key=value
        param_lines = []
        for line in content.splitlines():
            # Only lines with = and not starting with ;
            if "=" in line and not line.strip().startswith(";"):
                param_lines.append(line)

        st.subheader("✏️ Edit Parameters")
        new_lines = []
        for line in content.splitlines():
            if "=" in line and not line.strip().startswith(";"):
                k, v = line.split("=", 1)
                k = k.strip()
                v_main = v.split("||")[0].strip()  # Only editable value
                # Try to cast to float for slider, else text
                try:
                    v_num = float(v_main)
                    new_val = st.number_input(k, value=v_num)
                    new_line = f"{k}={new_val}{v[len(v_main):]}"
                except:
                    new_val = st.text_input(k, value=v_main)
                    new_line = f"{k}={new_val}{v[len(v_main):]}"
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        edited_content = "\n".join(new_lines)

        st.download_button(
            label="Download Edited .set File",
            data=edited_content,
            file_name="Edited_SetFile.set",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"Error reading setfile: {e}")
