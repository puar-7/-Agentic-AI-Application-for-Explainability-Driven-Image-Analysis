import streamlit as st


# ------------------------------------------------------------------
# SHARED — always hidden regardless of theme
# ------------------------------------------------------------------
SHARED_CSS = """
<style>
/* ── Hide Streamlit's built-in Deploy toolbar (top-right corner) ── */
[data-testid="stToolbar"] {
    display: none !important;
}
/* Older Streamlit versions use this class name */
.stDeployButton {
    display: none !important;
}
/* Also hides the hamburger main-menu if present */
#MainMenu {
    visibility: hidden !important;
}
</style>
"""

# ------------------------------------------------------------------
# LIGHT THEME CSS
# ------------------------------------------------------------------
LIGHT_CSS = """
<style>
/* ── Root / body background — prevents white flash on edges ── */
html, body {
    background-color: #F7F9FB !important;
}

/* ── App background ── */
.stApp {
    background-color: #F7F9FB !important;
    color: #1E2A38 !important;
}

/* ── Bottom floating input container (outer shell) ── */
[data-testid="stBottom"] {
    background-color: #F7F9FB !important;
    border-top: 1px solid #E2E8F0 !important;
}
[data-testid="stBottom"] > div {
    background-color: #F7F9FB !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #EEF2F6 !important;
}
[data-testid="stSidebar"] * {
    color: #1E2A38 !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    color: #1E2A38 !important;
}

/* ── Chat input box ── */
[data-testid="stChatInput"] {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E0 !important;
}
[data-testid="stChatInput"] textarea {
    background-color: #FFFFFF !important;
    color: #1E2A38 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #8FA3B1 !important;
    opacity: 1 !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #FFFFFF !important;
    color: #1E2A38 !important;
    border: 1px solid #CBD5E0 !important;
}
.stButton > button:hover {
    background-color: #E2E8F0 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
}

/* ── Text inputs and text areas ── */
.stTextArea textarea, .stTextInput input {
    background-color: #FFFFFF !important;
    color: #1E2A38 !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {
    color: #8FA3B1 !important;
    opacity: 1 !important;
}

/* ── File uploader — Browse files button ── */
[data-testid="stFileUploaderDropzone"] button {
    background-color: #FFFFFF !important;
    color: #1E2A38 !important;
    border: 1px solid #CBD5E0 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background-color: #E2E8F0 !important;
}

/* ── Source citation line ── */
.source-line {
    color: #6B8FA3 !important;
}

/* ── Greeting text ── */
.greeting-text {
    color: #1E2A38 !important;
}
</style>
"""

# ------------------------------------------------------------------
# DARK THEME CSS
# ------------------------------------------------------------------
DARK_CSS = """
<style>
/* ── Root / body background — prevents white flash on edges ── */
html, body {
    background-color: #0F1923 !important;
}

/* ── App background ── */
.stApp {
    background-color: #0F1923 !important;
    color: #E2E8F0 !important;
}

/* ── Top header/toolbar bar ── */
header[data-testid="stHeader"] {
    background-color: #0F1923 !important;
    border-bottom: 1px solid #2D3F55 !important;
}
header[data-testid="stHeader"] * {
    color: #E2E8F0 !important;
    fill: #E2E8F0 !important;
}

/* ── Bottom floating input container (outer shell — was the white band) ── */
[data-testid="stBottom"] {
    background-color: #0F1923 !important;
    border-top: 1px solid #2D3F55 !important;
}
[data-testid="stBottom"] > div {
    background-color: #0F1923 !important;
}

/* ── Inner floating wrapper ── */
.stChatFloatingInputContainer {
    background-color: #0F1923 !important;
    border-top: none !important;    /* border is handled by stBottom */
    padding: 8px !important;
}
.stChatFloatingInputContainer > div {
    background-color: #0F1923 !important;
}

/* ── The rounded chat input box itself ── */
[data-testid="stChatInput"] {
    background-color: #1A2535 !important;
    border: 1px solid #2D3F55 !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] > div {
    background-color: #1A2535 !important;
    border-radius: 8px !important;
}

/* ── Chat input textarea text ── */
[data-testid="stChatInput"] textarea {
    background-color: #1A2535 !important;
    color: #E2E8F0 !important;
    border: none !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #5A7A94 !important;
    opacity: 1 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #1A2535 !important;
}
[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background-color: #1A2535 !important;
    border: 1px solid #2D3F55 !important;
    color: #E2E8F0 !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #1A2535 !important;
    color: #E2E8F0 !important;
    border: 1px solid #2D3F55 !important;
}
.stButton > button:hover {
    background-color: #2D3F55 !important;
}

/* ── Expander — header AND body ── */
[data-testid="stExpander"] {
    background-color: #1A2535 !important;
    border: 1px solid #2D3F55 !important;
}
[data-testid="stExpander"] summary {
    background-color: #1A2535 !important;
    color: #E2E8F0 !important;
}
[data-testid="stExpander"] summary:hover {
    background-color: #2D3F55 !important;
}
[data-testid="stExpander"] summary span {
    color: #E2E8F0 !important;
}
[data-testid="stExpander"] summary svg {
    fill: #E2E8F0 !important;
}

/* ── Text inputs and text areas (workflow config box) ── */
.stTextArea textarea, .stTextInput input {
    background-color: #1A2535 !important;
    color: #E2E8F0 !important;
    border: 1px solid #2D3F55 !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {
    color: #5A7A94 !important;
    opacity: 1 !important;
}

/* ── File uploader container and dropzone ── */
[data-testid="stFileUploader"] {
    background-color: #1A2535 !important;
    border: 1px dashed #2D3F55 !important;
}
[data-testid="stFileUploader"] * {
    color: #E2E8F0 !important;
}
[data-testid="stFileUploaderDropzone"] {
    background-color: #1A2535 !important;
}

/* ── Browse files button (was invisible — explicit override needed) ── */
[data-testid="stFileUploaderDropzone"] button {
    background-color: #2D3F55 !important;
    color: #E2E8F0 !important;
    border: 1px solid #3D5570 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background-color: #3D5570 !important;
    color: #FFFFFF !important;
}

/* ── Code blocks (st.code / st.markdown code) ── */
[data-testid="stCode"], .stCode {
    background-color: #1A2535 !important;
}
[data-testid="stCode"] pre, .stCode pre {
    background-color: #1A2535 !important;
    color: #E2E8F0 !important;
    border: 1px solid #2D3F55 !important;
}
[data-testid="stCode"] code, .stCode code {
    background-color: #1A2535 !important;
    color: #E2E8F0 !important;
}
/* Inline code in markdown */
code {
    background-color: #2D3F55 !important;
    color: #E2E8F0 !important;
}

/* ── st.metric — label, value, delta ── */
[data-testid="stMetric"] {
    background-color: #1A2535 !important;
    border-radius: 6px;
    padding: 8px !important;
}
[data-testid="stMetricLabel"] {
    color: #7A9BB5 !important;
}
[data-testid="stMetricValue"] {
    color: #E2E8F0 !important;
}
[data-testid="stMetricDelta"] {
    color: #7A9BB5 !important;
}

/* ── Markdown text ── */
.stMarkdown, .stMarkdown p, .stMarkdown li {
    color: #E2E8F0 !important;
}

/* ── Headers inside main content ── */
h1, h2, h3, h4, h5, h6 {
    color: #E2E8F0 !important;
}

/* ── Divider ── */
hr {
    border-color: #2D3F55 !important;
}

/* ── Source citation line ── */
.source-line {
    color: #7A9BB5 !important;
}

/* ── Greeting text ── */
.greeting-text {
    color: #E2E8F0 !important;
}

/* ── Captions and muted text ── */
[data-testid="stCaptionContainer"], .stCaptionContainer {
    color: #7A9BB5 !important;
}

/* ── Selectbox / radio ── */
[data-testid="stRadio"] label {
    color: #E2E8F0 !important;
}

/* ── st.table ── */
[data-testid="stTable"] table {
    background-color: #1A2535 !important;
    color: #E2E8F0 !important;
}
[data-testid="stTable"] th {
    background-color: #2D3F55 !important;
    color: #E2E8F0 !important;
}
[data-testid="stTable"] td {
    color: #E2E8F0 !important;
    border-color: #2D3F55 !important;
}

/* ── st.success / st.error / st.warning / st.info ── */
[data-testid="stAlert"] {
    background-color: #1A2535 !important;
    color: #E2E8F0 !important;
}

/* ── Spinner text ── */
[data-testid="stSpinner"] p {
    color: #E2E8F0 !important;
}

/* ── General fallback ── */
.element-container > div[class*="stBlock"] {
    background-color: transparent !important;
}
</style>
"""


def init_theme():
    """
    Initialise theme in session state if not already set.
    Call this once at the top of app.py before rendering anything.
    """
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False


def inject_theme_css():
    """
    Injects shared CSS (always) plus the correct theme CSS.
    Must be called on every rerender so styles are always applied.
    """
    # SHARED first — hides deploy button in both themes
    st.markdown(SHARED_CSS, unsafe_allow_html=True)

    if st.session_state.get("dark_mode", False):
        st.markdown(DARK_CSS, unsafe_allow_html=True)
    else:
        st.markdown(LIGHT_CSS, unsafe_allow_html=True)


def render_theme_toggle():
    """
    Renders the sun/moon toggle button in whichever container it is called from.
    Clicking it flips the dark_mode flag and triggers a rerun.
    """
    icon = "☀️" if st.session_state.get("dark_mode", False) else "🌙"
    label = f"{icon}  {'Light mode' if st.session_state.dark_mode else 'Dark mode'}"

    if st.button(label, key="theme_toggle_btn"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()


def get_greeting_color() -> str:
    """
    Returns the correct greeting text color for the current theme.
    Used by chat_ui.py to avoid a hardcoded color.
    """
    return "#E2E8F0" if st.session_state.get("dark_mode", False) else "#1E2A38"