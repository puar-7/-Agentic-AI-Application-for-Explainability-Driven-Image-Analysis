import streamlit as st
from ui.shared import post_json


def render_workflow_ui():
    st.subheader("⚙️ Workflow Mode")

    st.markdown(
        """
        Enter the workflow configuration **in plain text**, following this format:

        ```
        dataset path - /data/example.csv
        model path - /models/example.pkl
        target variable - label
        spurious attribute - gender
        execution mode - white | black | both
        ```
        """
    )

    # -----------------------------
    # Workflow input
    # -----------------------------
    workflow_input_text = st.text_area(
        "Workflow Configuration",
        height=220,
        placeholder=(
            "dataset path - /data/adult.csv\n"
            "model path - /models/logreg.pkl\n"
            "target variable - income\n"
            "spurious attribute - gender\n"
            "execution mode - both"
        )
    )

    run_clicked = st.button("▶ Run Workflow")

    if not run_clicked:
        return

    if not workflow_input_text.strip():
        st.warning("Please provide workflow configuration before running.")
        return

    # -----------------------------
    # Execute workflow via backend
    # -----------------------------
    with st.spinner("Running workflow..."):
        resp = post_json(
            "/workflow",
            {"config_text": workflow_input_text}
        )

    if not resp.ok:
        st.error(resp.text)
        return

    result = resp.json()

    # -----------------------------
    # Error handling
    # -----------------------------
    if result.get("error"):
        st.error(result["error"])
        return

    # -----------------------------
    # Results rendering
    # -----------------------------
    st.success("Workflow completed successfully.")

    # White-box result
    if result.get("white_box_result"):
        with st.expander("🔍 White-box Analysis", expanded=True):
            st.json(result["white_box_result"])

    # Black-box result
    if result.get("black_box_result"):
        with st.expander("📦 Black-box Analysis", expanded=True):
            st.json(result["black_box_result"])

    # Report
    if result.get("report"):
        with st.expander("📝 Workflow Report", expanded=True):
            report_content = result["report"]
            
            if "human_readable" in report_content:
                # Render as rich text (Markdown)
                st.markdown(report_content["human_readable"])
            else:
                # Fallback: If format is different, show raw data
                st.json(report_content)

    
