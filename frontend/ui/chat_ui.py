import streamlit as st
from ui.shared import post_json, post_files
import requests

def render_chat_ui():
    st.subheader("🗨️ Chat Mode (Local RAG)")

    # -----------------------------
    # Initialize session state
    # -----------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Store known files to prevent duplicates and display list
    if "known_files" not in st.session_state:
        st.session_state.known_files = set()    

        # FETCH ON LOAD: Get list of existing files from backend
        try:
            # We use a direct requests.get because shared.py only has post methods
            # assuming shared.API_BASE is imported or hardcoded, but better to reuse shared pattern if possible.
            # Since shared.py only has POST, we'll do a raw request here or add a get_json to shared.
            # For now, raw request using the same base url convention:
            from ui.shared import API_BASE
            response = requests.get(f"{API_BASE}/documents", timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.session_state.known_files = set(data.get("documents", []))
        except Exception:
            # If backend is down, we just start with empty set (fail soft)
            pass
    
    # Uploader key to reset widget after successful upload
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

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

        MAX_FILE_SIZE_MB = 200

        uploaded_files = st.file_uploader(
            "Upload local documents (PDF / TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key=f"file_uploader_{st.session_state.uploader_key}"
        )

        # Logic: Filter files before sending
        upload_queue = []
        duplicate_files = []
        oversized_files = []

        if uploaded_files:
            for file in uploaded_files:
                # Client Check 1: File Size
                file_size_mb = file.size / (1024 * 1024)
                if file_size_mb > MAX_FILE_SIZE_MB:
                    oversized_files.append((file.name, file_size_mb))
                    continue
                
                # Client Check 2: Name Duplication (Fast feedback)
                if file.name in st.session_state.known_files:
                    duplicate_files.append(file.name)
                    continue
                
                upload_queue.append(file)

        # Show warnings for any problematic files when user selects them
        # (Uploader reset after upload means this won't trigger false warnings)
        for name, size in oversized_files:
            st.error(f"❌ {name}: Exceeds {MAX_FILE_SIZE_MB}MB limit.")
        for name in duplicate_files:
            st.warning(f"⚠️ {name}: Already indexed.")

        # Button to trigger upload for valid files
        if upload_queue:
            if st.button(f"Upload {len(upload_queue)} New File(s)"):
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(upload_queue):
                    status_text.text(f"Uploading {file.name}...")
                    
                    # Prepare multipart form data
                    files_payload = [
                        ("files", (file.name, file.getvalue(), file.type))
                    ]
                    
                    try:
                        # Send to backend
                        resp = post_files("/upload-docs", files_payload)
                        
                        # --- RESPONSE HANDLING (Per your valid concern) ---
                        if resp.status_code == 200:
                            data = resp.json()
                            
                            if "new_files" in data:
                                # TRUE SUCCESS
                                st.success(f"✅ {file.name}: Indexed.")
                                st.session_state.known_files.add(file.name)
                            else:
                                # DUPLICATE CONTENT (Server-side hash check)
                                st.warning(f"⚠️ {file.name}: Content already exists (Skipped).")
                                st.session_state.known_files.add(file.name)
                        
                        elif resp.status_code == 400:
                            # Validation Error (e.g., .docx masquerading as .txt)
                            err_msg = resp.json().get("detail", "Unsupported format")
                            st.error(f"❌ {file.name}: {err_msg}")
                            
                        else:
                            # Server Error (500)
                            st.error(f"❌ {file.name}: Server error.")
                            
                    except Exception as e:
                        st.error(f"❌ {file.name}: Connection failed.")
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(upload_queue))
                
                status_text.text("Processing complete.")
                # Reset file uploader by incrementing key
                st.session_state.uploader_key += 1
                # Rerun to update the sidebar list visually
                st.rerun()

        st.divider()
        st.markdown("### 📄 Indexed Documents")
        
        if not st.session_state.known_files:
            st.info("No documents indexed.")
        else:
            for name in sorted(list(st.session_state.known_files)):
                st.write(f"📄 {name}")

        # Clear button (Global clear, as selective delete is not supported)
        if st.button("🗑️ Clear All Documents"):
            try:
                resp = post_json("/clear", {})
                if resp.ok:
                    st.session_state.chat_history = []
                    st.session_state.known_files = set()
                    st.session_state.uploader_key += 1  # Reset file uploader
                    st.success("System reset.")
                    st.rerun()
                else:
                    st.error("Failed to clear system.")
            except Exception:
                st.error("Backend connection failed.")

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
            try:
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
                elif resp.status_code == 404:
                            st.error("No documents found. Please upload documents first.")
                elif resp.status_code == 500:
                            st.error("Server error while generating answer.")
                else:
                            st.error("Something went wrong while processing your question.")

            except Exception:
                st.error("Cannot connect to backend. Is server running?")