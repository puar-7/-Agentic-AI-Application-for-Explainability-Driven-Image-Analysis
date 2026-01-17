import streamlit as st

from backend.graph.workflow_graph import WorkflowGraph
from backend.graph.state import GraphState


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
    # Build initial state
    # -----------------------------
    state = GraphState(
        mode="workflow",
        user_message=workflow_input_text
    )

    workflow_graph = WorkflowGraph()

    # -----------------------------
    # Execute workflow
    # -----------------------------
    with st.spinner("Running workflow..."):
        try:
            final_state = workflow_graph.run(state)
        except Exception as e:
            st.error(f"Workflow execution failed: {e}")
            return

    # -----------------------------
    # Error handling
    # -----------------------------
    if final_state.get("error"):
        st.error(final_state["error"])
        return

    # -----------------------------
    # Results rendering
    # -----------------------------
    st.success("Workflow completed successfully.")

    # White-box result
    if final_state.get("white_box_result"):
        with st.expander("🔍 White-box Analysis", expanded=True):
            st.json(final_state["white_box_result"])

    # Black-box result
    if final_state.get("black_box_result"):
        with st.expander("📦 Black-box Analysis", expanded=True):
            st.json(final_state["black_box_result"])

    # Report
    if final_state.get("report"):
        with st.expander("📝 Workflow Report", expanded=True):
            st.json(final_state["report"])

    # Evaluation
    if final_state.get("evaluation"):
        with st.expander("📊 Evaluation Summary", expanded=True):
            st.json(final_state["evaluation"])
