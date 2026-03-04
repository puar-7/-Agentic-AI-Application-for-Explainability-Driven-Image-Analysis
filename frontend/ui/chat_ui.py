import streamlit as st
from ui.shared import post_json, post_files #helper functions for API calls
import requests #direct http calls
from datetime import datetime


def get_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning, how can I help?"
    elif hour < 17:
        return "Good afternoon, how can I help?"
    else:
        return "Good evening, how can I help?"


def render_chat_ui():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "known_files" not in st.session_state: #track which files we've already indexed to prevent duplicates and show in sidebar
        st.session_state.known_files = set()
        try: #fetch existing documents from backend on initial load to populate known_files (and show in sidebar) without needing user interaction
            from ui.shared import API_BASE
            response = requests.get(f"{API_BASE}/documents", timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.session_state.known_files = set(data.get("documents", []))
        except Exception:
            pass

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    greeting_placeholder = st.empty()
    # Handle pending query if any (from external triggers)
    if st.session_state.pending_query:
        user_query = st.session_state.pending_query
        st.session_state.pending_query = None
        st.chat_message("user").write(user_query)
        st.session_state.chat_history.append(("user", user_query))
        # We'll handle the query later in the input section
        # but for simplicity, we'll just process it now
        _handle_query(user_query, greeting_placeholder)

    # Create a placeholder for the greeting (will be at the top)
    

    # Show greeting only if no chat history exists yet
    if not st.session_state.chat_history:
        no_docs_hint = (
            '<p style="font-size:0.88rem; color:#9AACBB; margin:0; text-align:center;">'
            'Upload documents from the sidebar to get started.'
            '</p>'
            if not st.session_state.known_files else ""
        )
        greeting_html = f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: calc(100vh - 220px);
            gap: 10px;
        ">
            <p style="
                font-size: 1.85rem;
                font-weight: 600;
                color: #1E2A38;
                margin: 0;
                letter-spacing: -0.4px;
                text-align: center;
            ">{get_greeting()}</p>
            {no_docs_hint}
        </div>
        """
        greeting_placeholder.markdown(greeting_html, unsafe_allow_html=True)

    # Display existing chat messages (from history)
    for role, message in st.session_state.chat_history:
        st.chat_message(role).write(message)

    # --- SIDEBAR (unchanged) ---
    with st.sidebar:
        st.markdown("#### Document Management")

        MAX_FILE_SIZE_MB = 200

        uploaded_files = st.file_uploader(
            "Upload local documents (PDF / TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key=f"file_uploader_{st.session_state.uploader_key}"
        )

        upload_queue = []
        duplicate_files = []
        oversized_files = []

        if uploaded_files:
            for file in uploaded_files:
                file_size_mb = file.size / (1024 * 1024)
                if file_size_mb > MAX_FILE_SIZE_MB:
                    oversized_files.append((file.name, file_size_mb))
                    continue
                if file.name in st.session_state.known_files:
                    duplicate_files.append(file.name)
                    continue
                upload_queue.append(file)

        for name, size in oversized_files:
            st.error(f"{name}: Exceeds {MAX_FILE_SIZE_MB}MB limit.")
        for name in duplicate_files:
            st.warning(f"{name}: Already indexed.")

        if upload_queue:
            if st.button(f"Upload {len(upload_queue)} New File(s)"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, file in enumerate(upload_queue):
                    status_text.text(f"Uploading {file.name}...")
                    files_payload = [("files", (file.name, file.getvalue(), file.type))]

                    try:
                        resp = post_files("/upload-docs", files_payload)

                        if resp.status_code == 200:
                            data = resp.json()
                            if "new_files" in data:
                                st.success(f"{file.name}: Indexed.")
                                st.session_state.known_files.add(file.name)
                            else:
                                st.warning(f"{file.name}: Content already exists (Skipped).")
                                st.session_state.known_files.add(file.name)
                        elif resp.status_code == 400:
                            err_msg = resp.json().get("detail", "Unsupported format")
                            st.error(f"{file.name}: {err_msg}")
                        else:
                            st.error(f"{file.name}: Server error.")

                    except Exception as e:
                        st.error(f"{file.name}: Connection failed.")

                    progress_bar.progress((i + 1) / len(upload_queue))

                status_text.text("Processing complete.")
                st.session_state.uploader_key += 1
                st.rerun()

        st.divider()
        st.markdown("#### Indexed Documents")

        if not st.session_state.known_files:
            st.caption("No documents indexed yet.")
        else:
            for name in sorted(list(st.session_state.known_files)):
                display_name = name if len(name) <= 32 else name[:29] + "..."
                st.markdown(
                    f'<div class="doc-item">- {display_name}</div>',
                    unsafe_allow_html=True
                )

        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.caption("Removes all documents and resets the conversation.")

        st.markdown('<div class="danger-zone">', unsafe_allow_html=True)
        if st.button("Clear All Documents", key="clear_all_btn"):
            try:
                resp = post_json("/clear", {})
                if resp.ok:
                    st.session_state.chat_history = []
                    st.session_state.known_files = set()
                    st.session_state.uploader_key += 1
                    st.success("System reset.")
                    st.rerun()
                else:
                    st.error("Failed to clear system.")
            except Exception:
                st.error("Backend connection failed.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- CHAT INPUT ---
    user_query = st.chat_input("Ask a question based on uploaded documents")

    # Define query handler with access to the greeting placeholder
    def _handle_query(query, placeholder):
        with st.spinner("Generating answer..."):
            history_payload = [
                {"role": h_role, "content": h_msg}
                for h_role, h_msg in st.session_state.chat_history
            ]
            try:
                resp = post_json("/chat", {
                    "query": query,
                    "history": history_payload
                })

                if resp.ok:
                    data = resp.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])

                    st.chat_message("assistant").write(answer)
                    st.session_state.chat_history.append(("assistant", answer))

                    if sources:
                        with st.expander("Retrieved context"):
                            for i, src in enumerate(sources, 1):
                                st.markdown(f"**Chunk {i}:**")
                                st.write(src.get("content", ""))
                                st.divider()

                elif resp.status_code == 404:
                    st.error("No documents found. Please upload documents first.")
                elif resp.status_code == 500:
                    st.error("Server error while generating answer.")
                else:
                    st.error("Something went wrong while processing your question.")

            except Exception:
                st.error("Cannot connect to backend. Is server running?")
            finally:
                # Clear the greeting placeholder after the assistant responds
                placeholder.empty()

    if user_query:
        # Display user message immediately
        st.chat_message("user").write(user_query)
        st.session_state.chat_history.append(("user", user_query))
        # Process the query and clear the greeting afterwards
        _handle_query(user_query, greeting_placeholder)