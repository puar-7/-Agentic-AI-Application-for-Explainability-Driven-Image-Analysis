import streamlit as st
from pathlib import Path

from ui.chat_ui import render_chat_ui
from ui.workflow_ui import render_workflow_ui
from ui.theme import init_theme, inject_theme_css, render_theme_toggle


st.set_page_config(
    page_title="Agentic Framework",
    layout="wide"
)

# -----------------------------
# Theme — must come before any rendering
# -----------------------------
init_theme()
inject_theme_css()

# -----------------------------
# Session state initialization
# -----------------------------
if "mode" not in st.session_state:
    st.session_state.mode = "chat"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Sidebar: Mode selector + Theme toggle
# -----------------------------
with st.sidebar:
    st.markdown("### Navigation")
    st.radio(
        "Select Mode",
        options=["chat", "workflow"],
        key="mode",
        label_visibility="collapsed"
    )

    st.divider()
    render_theme_toggle()


# -----------------------------
# Mode routing
# -----------------------------
if st.session_state.mode == "chat":
    render_chat_ui()
else:
    render_workflow_ui()