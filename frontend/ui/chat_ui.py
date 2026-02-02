import streamlit as st
from ui.shared import post_json, post_files


def render_chat_ui():
    st.subheader("🗨️ Chat Mode (Local RAG)")

    # -----------------------------
    # Initialize session state
    # -----------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # -----------------------------
    # Display chat history
    # -----------------------------
    for role, message in st.session_state.chat_history:
        st.chat_message(role).write(message)

    # -----------------------------
    # Sidebar: Document management
    # -----------------------------
    with st.sidebar:
        st.header("📂 Document Management")

        uploaded_files = st.file_uploader(
            "Upload local documents (PDF / TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )

        if st.button("🗑️ Clear documents and reset"):
            with st.spinner("Clearing system..."):
                resp = post_json("/clear", {})

            if resp.ok:
                st.session_state.chat_history = []
                st.success("Cleared documents and chat history. Start fresh.")
                st.rerun()
            else:
                st.error(resp.text)

        if uploaded_files:
            files = [
                ("files", (f.name, f.getvalue(), f.type))
                for f in uploaded_files
            ]

            with st.spinner("Uploading and indexing documents..."):
                resp = post_files("/upload-docs", files)

            if resp.ok:
                st.success(resp.json().get("message", "Documents indexed and ready."))
            else:
                st.error(resp.text)

    # -----------------------------
    # Chat input
    # -----------------------------
    user_query = st.chat_input("Ask a question based on uploaded documents")

    if user_query:
        # Show user message
        st.chat_message("user").write(user_query)
        st.session_state.chat_history.append(("user", user_query))

        # Send query to backend
        with st.spinner("Generating answer..."):
            # Convert list of tuples [("user", "hi")] -> list of dicts [{"role": "user", "content": "hi"}]
            history_payload = [
                {"role": h_role, "content": h_msg} 
                for h_role, h_msg in st.session_state.chat_history
            ]
            
            # Send query AND history
            resp = post_json("/chat", {
                "query": user_query,
                "history": history_payload
            })

        if resp.ok:
            data = resp.json()

            answer = data.get("answer", "")
            sources = data.get("sources", [])

            st.chat_message("assistant").write(answer)
            st.session_state.chat_history.append(("assistant", answer))

            # Show retrieved local context (sources)
            if sources:
                with st.expander("📄 Retrieved context"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.write(src.get("content", ""))
                        st.divider()
