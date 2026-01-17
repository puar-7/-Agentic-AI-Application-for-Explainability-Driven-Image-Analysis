import streamlit as st
from typing import List
import tempfile
import os
import hashlib

def file_hash(file):
    return hashlib.md5(file.getbuffer()).hexdigest()

from backend.services.document_store import DocumentStore
from backend.graph.chat_graph import ChatGraph
from backend.graph.state import GraphState
from backend.nodes.chat.local_retriever_node import LocalRetrieverNode
from backend.nodes.chat.chat_llm_node import ChatLLMNode
from backend.llm.hf_client import get_chat_llm


def _init_document_store(uploaded_files: List):
    """
    Build or update the shared DocumentStore from uploaded files.
    """
    store = DocumentStore()

    file_paths = []
    for file in uploaded_files:
        suffix = os.path.splitext(file.name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.getbuffer())
            file_paths.append(tmp.name)

    documents = store.load_documents(file_paths)
    store.build_indexes(documents)

    return store


def render_chat_ui():
    st.subheader("🗨️ Chat Mode (Local RAG)")
    for role, message in st.session_state.chat_history:
        st.chat_message(role).write(message)


    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = None

    # -----------------------------
    # File upload
    # -----------------------------

    with st.sidebar:
        st.header("📂 Document Management")

        uploaded_files = st.file_uploader(
            "Upload local documents (PDF / TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )

        if st.button("🗑️ Clear documents and reset"):
            st.session_state.document_store = None
            st.session_state.indexed_files = None
            st.session_state.chat_history = []
            st.success("Cleared documents and chat history. Start fresh.")
            st.rerun()


    if uploaded_files:
        current_files = tuple((f.name, file_hash(f)) for f in uploaded_files)

        if st.session_state.indexed_files != current_files:
            with st.spinner("Building document index..."):
                st.session_state.document_store = _init_document_store(uploaded_files)
                st.session_state.indexed_files = current_files
            st.success("Documents indexed and ready for querying.")
    # -----------------------------
    # Chat input
    # -----------------------------
    user_query = st.chat_input("Ask a question based on uploaded documents")

    if user_query:
        if st.session_state.document_store is None:
            st.warning("Please upload documents before asking questions.")
            return

        # Display user message
        st.chat_message("user").write(user_query)

        # -----------------------------
        # Build ChatGraph
        # -----------------------------
        retriever_node = LocalRetrieverNode(st.session_state.document_store)
        llm = get_chat_llm()
        chat_node = ChatLLMNode(llm)

        chat_graph = ChatGraph(
            retriever_node=retriever_node,
            chat_llm_node=chat_node
        )

        # -----------------------------
        # Run graph
        # -----------------------------
        state = GraphState(
            mode="chat",
            user_message=user_query,
            chat_history=st.session_state.chat_history
        )

        with st.spinner("Generating answer..."):
            final_state = chat_graph.run(state)

        # Display assistant response
        st.chat_message("assistant").write(final_state.chat_response)


        # Save history
        st.session_state.chat_history.append(
            ("user", user_query)
        )
        st.session_state.chat_history.append(
            ("assistant", final_state.chat_response)
        )

        # -----------------------------
        # Optional: show retrieved docs
        # -----------------------------
        if final_state.retrieved_docs:
            with st.expander("📄 Retrieved context"):
                for i, doc in enumerate(final_state.retrieved_docs, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(doc.page_content)
                    st.divider()
