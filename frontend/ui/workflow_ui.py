import streamlit as st
from ui.shared import post_json, API_BASE


# Port 8000 serves static files — image URLs from the adapter are
# relative paths that need this prefix to be renderable in the browser
def _abs_url(relative_url: str) -> str:
    return f"{API_BASE}{relative_url}"


def render_workflow_ui():
    # ------------------------------------------------------------------
    # Persist the last successful result in session state so it survives
    # any rerun that isn't triggered by the Run button — most importantly,
    # the theme toggle rerun.  Only a new successful run overwrites it.
    # ------------------------------------------------------------------
    if "last_workflow_result" not in st.session_state:
        st.session_state.last_workflow_result = None

    st.subheader("⚙️ Workflow Mode")

    st.markdown(
        """
        Enter the workflow configuration **in plain text**, following this format:

        ```
        dataset name - CELEBA
        model name - RESNET
        execution mode - white | black | both
        ```

        Optional fields (can be omitted — defaults shown):
        ```
        target variable - gender
        spurious attribute - age
        similarity - COSINE | EUCLIDEAN       (default: COSINE)
        explainer - LIME | SHAP | RISE        (default: LIME)
        ```

        Supported datasets: `CELEBA`, `VGGFACE2`, `DIGIFACE`  
        Supported models: `RESNET`, `FACENET`

        > **Note:** Black-box build mode runs a full pipeline
        > (embedding extraction → FAISS index → evaluation → LIME explanations).
        > This may take several minutes depending on dataset size.
        """
    )

    workflow_input_text = st.text_area(
        "Workflow Configuration",
        height=200,
        placeholder=(
            "dataset name - CELEBA\n"
            "model name - RESNET\n"
            "execution mode - black"
        ),
    )

    run_clicked = st.button("▶ Run Workflow")

    # ------------------------------------------------------------------
    # Handle button click
    #
    # Errors (bad input, network failure, pipeline error) are shown
    # transiently via st.error / st.warning and do NOT overwrite the
    # stored result, so a previous successful result stays visible.
    #
    # Only a new successful response overwrites last_workflow_result.
    # ------------------------------------------------------------------
    if run_clicked:
        if not workflow_input_text.strip():
            st.warning("Please provide workflow configuration before running.")
        else:
            with st.spinner(
                "Running workflow… Black-box build may take several minutes. "
                "Please keep this tab open."
            ):
                try:
                    resp = post_json("/workflow", {"config_text": workflow_input_text})
                except Exception as e:
                    st.error("Failed to reach backend.")
                    st.code(str(e))
                    resp = None

            if resp is not None:
                if not resp.ok:
                    st.write("**Status Code:**", resp.status_code)
                    st.write("**Content-Type:**", resp.headers.get("Content-Type", "unknown"))
                    st.code(resp.text[:2000], language="html")
                else:
                    result = resp.json()
                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        # Store on success — this is the only place we write
                        st.session_state.last_workflow_result = result

    # ------------------------------------------------------------------
    # Render last successful result
    #
    # This block runs on EVERY script execution — including reruns from
    # theme toggle — so results are always visible as long as session
    # state holds them.
    # ------------------------------------------------------------------
    if st.session_state.last_workflow_result is not None:
        _render_workflow_result(st.session_state.last_workflow_result)


# ------------------------------------------------------------------
# Result renderer — separated so the render path is clearly distinct
# from the fetch path above.
# ------------------------------------------------------------------

def _render_workflow_result(result: dict) -> None:
    """
    Pure display function.  Renders a completed workflow result dict.
    Called from render_workflow_ui() reading session state, so it works
    identically on a fresh run and on any subsequent rerun.
    """
    st.success("Workflow completed successfully.")

    # ── White-box result ──────────────────────────────────────────────
    if result.get("white_box_result"):
        with st.expander("🔍 White-box Analysis", expanded=True):
            st.json(result["white_box_result"])

    # ── Black-box result ──────────────────────────────────────────────
    if result.get("black_box_result"):
        bb        = result["black_box_result"]
        bb_raw    = bb.get("raw_output") or {}
        bb_status = bb.get("status")

        with st.expander("📦 Black-box Analysis", expanded=True):
            if bb_status == "failure":
                st.error(f"Black-box pipeline failed: {bb.get('summary', '')}")
                if bb_raw.get("error"):
                    st.code(bb_raw["error"])
            else:
                _render_bb_success(bb_raw)

    # ── LLM Report ───────────────────────────────────────────────────
    if result.get("report"):
        with st.expander("📝 Workflow Report", expanded=True):
            report_content = result["report"]
            if "human_readable" in report_content:
                st.markdown(report_content["human_readable"])
            else:
                st.json(report_content)


# ------------------------------------------------------------------
# Black-box success renderer (unchanged from original)
# ------------------------------------------------------------------

def _render_bb_success(bb_raw: dict) -> None:
    """
    Renders:
        1. Build configuration summary
        2. Retrieval metrics dashboard
        3. LIME explanation image pairs (overlay + heatmap)
    """

    # ---- Configuration summary ----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset",    bb_raw.get("dataset_name", "—"))
    col2.metric("Model",      bb_raw.get("model_name",   "—"))
    col3.metric("Similarity", bb_raw.get("similarity",   "—"))
    col4.metric("Explainer",  bb_raw.get("explainer",    "—"))

    st.divider()

    # ---- Retrieval metrics ----
    st.markdown("#### Retrieval Metrics")

    metrics_block = bb_raw.get("metrics", {})

    # The metrics JSON has a nested "metrics" key from save_results()
    # Handle both flat and nested formats defensively
    if "metrics" in metrics_block:
        metric_values = metrics_block["metrics"]
    else:
        metric_values = metrics_block

    if metric_values:
        _render_metrics(metric_values)
    else:
        st.caption("No metrics available — evaluation may not have run.")

    st.divider()

    # ---- LIME explanation images ----
    images = bb_raw.get("images", [])

    st.markdown("#### LIME Explanation Samples")

    if not images:
        st.caption(
            "No explanation images found. "
            "This can happen if LIME did not complete before the build finished."
        )
        return

    st.caption(
        f"Showing {len(images)} image(s). "
        "Left: LIME overlay (highlighted regions). "
        "Right: Importance heatmap (jet colormap — warm = high influence)."
    )

    for img in images:
        image_id    = img.get("image_id", "unknown")
        overlay_url = _abs_url(img["overlay_url"])
        heatmap_url = _abs_url(img["heatmap_url"])

        st.markdown(f"**Image: `{image_id}`**")
        left, right = st.columns(2)

        with left:
            st.image(overlay_url, caption="LIME Overlay", width=300)
        with right:
            st.image(heatmap_url, caption="Importance Heatmap", width=300)

        st.divider()


# ------------------------------------------------------------------
# Metrics dashboard helper (unchanged from original)
# ------------------------------------------------------------------

def _render_metrics(metric_values: dict) -> None:
    """
    Renders retrieval metrics as st.metric tiles.

    Known metric keys and their display labels:
        top1_accuracy        → Top-1 Accuracy
        top5_accuracy        → Top-5 Accuracy
        mrr                  → Mean Reciprocal Rank
        self_top1_accuracy   → Self Top-1 Accuracy (flat datasets)
        avg_top2_score       → Avg Top-2 Score
        avg_similarity_gap   → Avg Similarity Gap

    Any unknown keys are rendered as a fallback table below the tiles.
    """

    DISPLAY_LABELS = {
        "top1_accuracy":      ("Top-1 Accuracy",       True),
        "top5_accuracy":      ("Top-5 Accuracy",       True),
        "mrr":                ("Mean Reciprocal Rank", True),
        "self_top1_accuracy": ("Self Top-1 Accuracy",  True),
        "avg_top2_score":     ("Avg Top-2 Score",      False),
        "avg_similarity_gap": ("Avg Similarity Gap",   False),
    }

    known   = {k: v for k, v in metric_values.items() if k in DISPLAY_LABELS}
    unknown = {k: v for k, v in metric_values.items() if k not in DISPLAY_LABELS}

    if known:
        keys   = list(known.keys())
        chunks = [keys[i:i+3] for i in range(0, len(keys), 3)]

        for chunk in chunks:
            cols = st.columns(len(chunk))
            for col, key in zip(cols, chunk):
                label, is_accuracy = DISPLAY_LABELS[key]
                value = known[key]

                display_val = f"{value * 100:.2f}%" if is_accuracy else f"{value:.4f}"
                col.metric(label=label, value=display_val)

    if unknown:
        st.markdown("**Additional Metrics**")
        rows = [
            {"Metric": k, "Value": f"{v:.4f}" if isinstance(v, float) else str(v)}
            for k, v in unknown.items()
        ]
        st.table(rows)