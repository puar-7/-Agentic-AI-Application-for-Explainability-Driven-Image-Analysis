import sys
import streamlit as st
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from ui.chat_ui import render_chat_ui
from ui.workflow_ui import render_workflow_ui




st.set_page_config(
    page_title="Agentic Framework",
    layout="wide"
)

st.title("🧠 Agentic Framework")

# -----------------------------
# Session state initialization
# -----------------------------
if "mode" not in st.session_state:
    st.session_state.mode = "chat"

if "document_store" not in st.session_state:
    st.session_state.document_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Mode selector
# -----------------------------
with st.sidebar:
    st.header("⚙️ Mode")

    mode = st.radio(
        "Select Mode",
        options=["chat", "workflow"],
        index=0 if st.session_state.mode == "chat" else 1
    )

st.session_state.mode = mode


st.divider()

# -----------------------------
# Mode routing
# -----------------------------
if mode == "chat":
    render_chat_ui()
else:
    render_workflow_ui()
