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
        try:
            resp = post_json(
                "/workflow",
                {"config_text": workflow_input_text}
            )
        except Exception as e:
            st.write("**Failed to reach backend entirely:**")
            st.code(str(e))
            return

    if not resp.ok:
        st.write("**Status Code:**", resp.status_code)
        st.write("**Content-Type:**", resp.headers.get("Content-Type", "unknown"))
        st.code(resp.text[:2000], language="html")
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
            bb_data = result["black_box_result"].get("raw_output", {})
            matches = bb_data.get("api_matches", [])
            
            if matches:
                st.write("### Top Retrieval Matches")
                cols = st.columns(len(matches))
                for i, match in enumerate(matches):
                    with cols[i]:
                        # Assuming match["matched_path"] is accessible by the frontend
                        st.image(match["matched_path"], caption=f"Score: {match['score']:.4f}")
            else:
                st.json(result["black_box_result"]) # Fallback

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

    
