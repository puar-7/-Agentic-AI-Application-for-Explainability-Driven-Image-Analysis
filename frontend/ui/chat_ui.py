import streamlit as st
from ui.shared import post_json, post_files
from ui.theme import get_greeting_color
import requests
from datetime import datetime
import os


def get_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning, how can I help?"
    elif hour < 17:
        return "Good afternoon, how can I help?"
    else:
        return "Good evening, how can I help?"


def _group_sources(sources: list) -> dict:
    """
    Splits sources into two groups by source_type, then:
      - Groups local docs  by filename with sorted unique page numbers
      - Groups web results by URL (deduplicated), keeping title + url

    Returns:
        {
            "documents": {filename: [page, ...] or None},
            "web":       [{"title": ..., "url": ...}, ...]
        }
    """
    doc_groups = {}
    web_seen   = {}

    for src in sources:
        metadata    = src.get("metadata", {})
        source_type = metadata.get("source_type", "document")

        if source_type == "document":
            source_path = metadata.get("source", "")
            filename    = os.path.basename(source_path) if source_path else "Unknown source"
            page        = metadata.get("page")

            if filename not in doc_groups:
                doc_groups[filename] = set() if page is not None else None

            if page is not None and doc_groups[filename] is not None:
                doc_groups[filename].add(int(page) + 1)

        elif source_type == "web":
            url   = metadata.get("url", "")
            title = metadata.get("title", "Untitled")
            if url and url not in web_seen:
                web_seen[url] = {"title": title, "url": url}

    return {
        "documents": {
            filename: sorted(pages) if pages is not None else None
            for filename, pages in doc_groups.items()
        },
        "web": list(web_seen.values()),
    }


def _render_sources(sources: list) -> None:
    """
    Renders source attribution lines below a chat message.
    Unchanged from original.
    """
    if not sources:
        return

    grouped    = _group_sources(sources)
    doc_groups = grouped["documents"]
    web_groups = grouped["web"]

    source_lines = []

    for filename, pages in doc_groups.items():
        if pages is None:
            source_lines.append(f"📄 {filename}")
        elif len(pages) == 1:
            source_lines.append(f"📄 {filename}  •  Page {pages[0]}")
        else:
            pages_str = ", ".join(str(p) for p in pages)
            source_lines.append(f"📄 {filename}  •  Pages {pages_str}")

    for web in web_groups:
        title = web["title"]
        url   = web["url"]
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
        except Exception:
            domain = url
        source_lines.append(
            f"🌐 [{title}]({url}) &nbsp;`{domain}`"
        )

    if source_lines:
        sources_md = "  \n".join(source_lines)
        st.markdown(
            f"<div style='font-size:0.8rem; color:#6B8FA3; "
            f"margin-top:4px;'>{sources_md}</div>",
            unsafe_allow_html=True
        )


# ------------------------------------------------------------------
# CHANGE 1 — ZIP upload feedback renderer
#
# New helper that renders the four-category response from the backend
# for ZIP uploads. Called from the upload button handler when the
# backend returns the new response schema.
#
# For direct file uploads the existing per-file success/warning/error
# messages are kept — they're simpler and sufficient for single files.
# ------------------------------------------------------------------

def _render_upload_result(result: dict) -> None:
    """
    Renders a structured upload result from the backend.
    Used for both direct and ZIP uploads now that the backend
    returns the four-category schema for both.

    Shows:
        - st.success summary line
        - st.warning for duplicates and unsupported files (collapsible)
        - st.error for failures (collapsible, with reasons)
    """
    indexed     = result.get("indexed", [])
    duplicates  = result.get("skipped_duplicates", [])
    unsupported = result.get("skipped_unsupported", [])
    failed      = result.get("failed", [])
    source      = result.get("source", "direct")

    label = "ZIP" if source == "zip" else "upload"

    if indexed:
        st.success(f"✅ {len(indexed)} file(s) indexed from {label}.")

    if not indexed and not duplicates and not unsupported and not failed:
        st.info("No files were processed.")
        return

    if duplicates:
        with st.expander(f"⏭️ {len(duplicates)} already indexed (skipped)"):
            for name in duplicates:
                st.caption(f"• {name}")

    if unsupported:
        with st.expander(f"⚠️ {len(unsupported)} unsupported file type(s) skipped"):
            for name in unsupported:
                st.caption(f"• {name}")

    if failed:
        with st.expander(f"❌ {len(failed)} file(s) failed to load"):
            for item in failed:
                if isinstance(item, dict):
                    st.caption(f"• **{item.get('file', '?')}** — {item.get('reason', '')}")
                else:
                    st.caption(f"• {item}")


def render_chat_ui():
    # ------------------------------------------------------------------
    # Session state initialisation — unchanged
    # ------------------------------------------------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "known_files" not in st.session_state:
        st.session_state.known_files = set()
        try:
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

    # Stores upload results across the rerun boundary so they survive
    # the st.rerun() call that resets the file uploader widget.
    # Without this, _render_upload_result output is wiped before the
    # user can read it.
    if "last_upload_results" not in st.session_state:
        st.session_state.last_upload_results = []

    greeting_placeholder = st.empty()

    def _handle_query(query, placeholder):
        with st.spinner("Generating answer..."):
            history_payload = []
            for msg in st.session_state.chat_history:
                if isinstance(msg, tuple):
                    h_role, h_content = msg[0], msg[1]
                else:
                    h_role, h_content = msg["role"], msg["content"]
                history_payload.append({"role": h_role, "content": h_content})

            try:
                resp = post_json("/chat", {
                    "query": query,
                    "history": history_payload
                })

                if resp.ok:
                    data    = resp.json()
                    answer  = data.get("answer", "")
                    sources = data.get("sources", [])

                    st.chat_message("assistant").write(answer)

                    st.session_state.chat_history.append({
                        "role":    "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                    _render_sources(sources)

                elif resp.status_code == 404:
                    st.error("No documents found. Please upload documents first.")
                elif resp.status_code == 500:
                    st.error("Server error while generating answer.")
                else:
                    st.write("**Status Code:**", resp.status_code)
                    st.write("**Content-Type:**", resp.headers.get("Content-Type", "unknown"))
                    st.code(resp.text[:2000], language="html")

            except Exception as e:
                st.error("Cannot connect to backend. Is server running?")
                st.code(str(e))
            finally:
                placeholder.empty()

    if st.session_state.pending_query:
        user_query = st.session_state.pending_query
        st.session_state.pending_query = None
        st.chat_message("user").write(user_query)
        st.session_state.chat_history.append({
            "role":    "user",
            "content": user_query,
            "sources": None,
        })
        _handle_query(user_query, greeting_placeholder)

    if not st.session_state.chat_history:
        no_docs_hint = (
            '<p style="font-size:0.88rem; color:#9AACBB; margin:0; text-align:center;">'
            'Upload documents from the sidebar to get started.'
            '</p>'
            if not st.session_state.known_files else ""
        )
        greeting_color = get_greeting_color()
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
                color: {greeting_color};
                margin: 0;
                letter-spacing: -0.4px;
                text-align: center;
            ">{get_greeting()}</p>
            {no_docs_hint}
        </div>
        """
        greeting_placeholder.markdown(greeting_html, unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if isinstance(msg, tuple):
            role, content, sources = msg[0], msg[1], None
        else:
            role    = msg["role"]
            content = msg["content"]
            sources = msg.get("sources")

        st.chat_message(role).write(content)

        if sources:
            _render_sources(sources)

    # ------------------------------------------------------------------
    # Sidebar — document management
    # ------------------------------------------------------------------
    with st.sidebar:
        # ------------------------------------------------------------------
        # Render upload results from the previous run.
        # These were stored in session state before st.rerun() was called,
        # so they survive the rerun and are visible to the user here.
        # Cleared immediately after rendering so they don't persist forever.
        # ------------------------------------------------------------------
        if st.session_state.last_upload_results:
            for _result in st.session_state.last_upload_results:
                _render_upload_result(_result)
            st.session_state.last_upload_results = []

        st.markdown("#### Document Management")

        # CHANGE 2 — Add "zip" to accepted types.
        # Also expanded to include all newly supported direct-upload types.
        # The size warning is removed for ZIPs specifically because a ZIP
        # of 30 PDFs may legitimately exceed 200MB; size enforcement for
        # ZIPs happens server-side via Streamlit's upload limit (configurable
        # in .streamlit/config.toml).
        uploaded_files = st.file_uploader(
            "Upload documents or a folder ZIP  (PDF / TXT / DOCX / XLSX / PPTX / ZIP)",
            type=["pdf", "txt", "docx", "xlsx", "pptx", "zip"],
            accept_multiple_files=True,
            key=f"file_uploader_{st.session_state.uploader_key}"
        )

        MAX_FILE_SIZE_MB = 200

        upload_queue    = []
        duplicate_files = []
        oversized_files = []

        if uploaded_files:
            for file in uploaded_files:
                is_zip = file.name.lower().endswith(".zip")

                # Skip client-side size check for ZIPs — server handles it
                if not is_zip:
                    file_size_mb = file.size / (1024 * 1024)
                    if file_size_mb > MAX_FILE_SIZE_MB:
                        oversized_files.append((file.name, file_size_mb))
                        continue

                # ZIPs are never in known_files by their ZIP name —
                # their contents are. So skip the duplicate check for ZIPs.
                if not is_zip and file.name in st.session_state.known_files:
                    duplicate_files.append(file.name)
                    continue

                upload_queue.append(file)

        for name, size in oversized_files:
            st.error(f"{name}: Exceeds {MAX_FILE_SIZE_MB}MB limit.")
        for name in duplicate_files:
            st.warning(f"{name}: Already indexed.")

        if upload_queue:
            if st.button(f"Upload {len(upload_queue)} File(s)"):
                progress_bar = st.progress(0)
                status_text  = st.empty()

                for i, file in enumerate(upload_queue):
                    status_text.text(f"Uploading {file.name}...")
                    files_payload = [("files", (file.name, file.getvalue(), file.type))]

                    try:
                        resp = post_files("/upload-docs", files_payload)

                        if resp.status_code == 200:
                            result = resp.json()

                            # Store result in session state — NOT rendered here.
                            # st.rerun() at the end of this block wipes anything
                            # rendered inline. Storing here means the results
                            # survive the rerun and are rendered at the top of
                            # the sidebar on the next script execution.
                            st.session_state.last_upload_results.append(result)

                            # Update known_files immediately — this also survives
                            # the rerun since it's in session state.
                            for indexed_name in result.get("indexed", []):
                                st.session_state.known_files.add(indexed_name)

                        elif resp.status_code == 400:
                            err_msg = resp.json().get("detail", "Bad request")
                            st.error(f"{file.name}: {err_msg}")
                        else:
                            st.write(f"**{file.name} — Status Code:**", resp.status_code)
                            st.write("**Content-Type:**", resp.headers.get("Content-Type", "unknown"))
                            st.code(resp.text[:2000], language="html")

                    except Exception as e:
                        st.error(f"{file.name}: Failed to reach backend.")
                        st.code(str(e))

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
                    st.session_state.chat_history  = []
                    st.session_state.known_files   = set()
                    st.session_state.uploader_key += 1
                    st.success("System reset.")
                    st.rerun()
                else:
                    st.error("Failed to clear system.")
            except Exception:
                st.error("Backend connection failed.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Chat input — unchanged
    # ------------------------------------------------------------------
    user_query = st.chat_input("Ask a question based on uploaded documents")

    if user_query:
        st.chat_message("user").write(user_query)
        st.session_state.chat_history.append({
            "role":    "user",
            "content": user_query,
            "sources": None,
        })
        _handle_query(user_query, greeting_placeholder)