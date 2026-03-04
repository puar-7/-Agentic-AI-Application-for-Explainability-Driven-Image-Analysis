import streamlit as st
from pathlib import Path

from ui.chat_ui import render_chat_ui
from ui.workflow_ui import render_workflow_ui


st.set_page_config(
    page_title="Agentic Framework",
    layout="wide"
)

# -----------------------------
# Session state initialization
# -----------------------------
if "mode" not in st.session_state:
    st.session_state.mode = "chat"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Sidebar: Mode selector
# -----------------------------
with st.sidebar:
    st.markdown("### Navigation")
    st.radio(
        "Select Mode",
        options=["chat", "workflow"],
        key="mode",                  # Streamlit owns the state — no index or manual assignment needed
        label_visibility="collapsed"
    )


# -----------------------------
# Mode routing (no title/header)
# -----------------------------
if st.session_state.mode == "chat":
    render_chat_ui()
else:
    render_workflow_ui()