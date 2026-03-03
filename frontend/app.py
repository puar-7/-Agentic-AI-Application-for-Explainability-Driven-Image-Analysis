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
    mode = st.radio(
        "Select Mode",
        options=["chat", "workflow"],
        index=0 if st.session_state.mode == "chat" else 1,
        label_visibility="collapsed"
    )

st.session_state.mode = mode

# -----------------------------
# Mode routing (no title/header)
# -----------------------------
if mode == "chat":
    render_chat_ui()
else:
    render_workflow_ui()