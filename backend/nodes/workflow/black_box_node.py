import asyncio
from typing import Dict

import httpx

from backend.graph.state import GraphState
from backend.schemas.execution_result import ExecutionResult

# Port 8001 microservice
BLACK_BOX_API_URL = "http://localhost:8001"

# Polling configuration
POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS  = 1800   # 30 minutes — covers large datasets( change as required)


class BlackBoxNode:
    """
    Async LangGraph node that drives the black-box build pipeline.

    Flow:
        1. Extract dataset_name, model_name, similarity, explainer
           from state.workflow_input  (no more hardcoded paths or hacks)
        2. POST /build  → receives job_id immediately
        3. Poll GET /status/{job_id} every POLL_INTERVAL_SECONDS
        4. On "completed" → package metrics + image URLs into ExecutionResult
        5. On "failed"    → package error into ExecutionResult (no exception raise,
                            so ReportGenerationNode still runs and MongoDB still saves)
    """

    async def __call__(self, state: GraphState) -> Dict:

        # ----------------------------------------------------------
        # Guard: workflow_input must exist
        # ----------------------------------------------------------
        if not state.workflow_input:
            return {
                "black_box_result": ExecutionResult(
                    method="black_box",
                    status="failure",
                    summary="Workflow input missing for black-box execution.",
                    raw_output={"error": "state.workflow_input is None"},
                )
            }

        wi = state.workflow_input

        # ----------------------------------------------------------
        # Step 1 — Build the request payload from workflow_input.
        # dataset_name, model_name, similarity, explainer are all
        # validated upstream by WorkflowInputParserNode, so no
        # extra validation needed here.
        # ----------------------------------------------------------
        build_payload = {
            "dataset_name": wi.dataset_name,
            "model_name":   wi.model_name,
            "similarity":   wi.similarity,
            "explainer":    wi.explainer,
        }

        print(
            f"[BlackBoxNode] Firing build — "
            f"dataset={wi.dataset_name} model={wi.model_name} "
            f"similarity={wi.similarity} explainer={wi.explainer}"
        )

        # ----------------------------------------------------------
        # Step 2 — POST /build, get job_id back immediately
        # ----------------------------------------------------------
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(
                    f"{BLACK_BOX_API_URL}/build",
                    json=build_payload,
                )
                resp.raise_for_status()
                build_data = resp.json()

            except httpx.ConnectError:
                return {
                    "black_box_result": ExecutionResult(
                        method="black_box",
                        status="failure",
                        summary=(
                            "Could not connect to the Black-Box microservice. "
                            "Is port 8001 running?"
                        ),
                        raw_output={"error": "ConnectionError on POST /build"},
                    )
                }

            except httpx.HTTPStatusError as e:
                return {
                    "black_box_result": ExecutionResult(
                        method="black_box",
                        status="failure",
                        summary=f"Black-Box microservice returned an error on /build: {e}",
                        raw_output={"error": str(e)},
                    )
                }

        job_id = build_data.get("job_id")
        if not job_id:
            return {
                "black_box_result": ExecutionResult(
                    method="black_box",
                    status="failure",
                    summary="Black-Box microservice did not return a job_id.",
                    raw_output={"response": build_data},
                )
            }

        print(f"[BlackBoxNode] Build accepted — job_id={job_id}")

        # ----------------------------------------------------------
        # Step 3 — Poll GET /status/{job_id} until done or timeout
        # ----------------------------------------------------------
        elapsed = 0

        while elapsed < POLL_TIMEOUT_SECONDS:

            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            elapsed += POLL_INTERVAL_SECONDS

            async with httpx.AsyncClient(timeout=10.0) as client:
                try:
                    status_resp = await client.get(
                        f"{BLACK_BOX_API_URL}/status/{job_id}"
                    )
                    status_resp.raise_for_status()
                    job_data = status_resp.json()

                except httpx.ConnectError:
                    # Transient network hiccup — keep polling
                    print(
                        f"[BlackBoxNode] Poll failed (ConnectError) — "
                        f"job_id={job_id} elapsed={elapsed}s. Retrying..."
                    )
                    continue

                except httpx.HTTPStatusError as e:
                    # 404 right after firing can be a transient race —
                    # the background task may not have written the job file yet
                    if e.response.status_code == 404 and elapsed <= POLL_INTERVAL_SECONDS * 2:
                        print(
                            f"[BlackBoxNode] Job file not yet written (404) — "
                            f"job_id={job_id}. Retrying..."
                        )
                        continue

                    return {
                        "black_box_result": ExecutionResult(
                            method="black_box",
                            status="failure",
                            summary=f"Unexpected error while polling status: {e}",
                            raw_output={"error": str(e), "job_id": job_id},
                        )
                    }

            job_status = job_data.get("status")
            print(
                f"[BlackBoxNode] Poll — job_id={job_id} "
                f"status={job_status} elapsed={elapsed}s"
            )

            # --------------------------------------------------
            # Step 4a — Build completed successfully
            # --------------------------------------------------
            if job_status == "completed":
                metrics = job_data.get("metrics", {})
                images  = job_data.get("images",  [])

                # Build a human-readable summary from real metrics
                # so ReportGenerationNode has structured data to work with
                summary = _build_summary(
                    wi.dataset_name, wi.model_name,
                    wi.similarity, metrics, images,
                )

                return {
                    "black_box_result": ExecutionResult(
                        method="black_box",
                        status="success",
                        summary=summary,
                        raw_output={
                            "job_id":      job_id,
                            "dataset_name": wi.dataset_name,
                            "model_name":   wi.model_name,
                            "similarity":   wi.similarity,
                            "explainer":    wi.explainer,
                            "metrics":      metrics,
                            "images":       images,        # list of {image_id, overlay_url, heatmap_url}
                        },
                    )
                }

            # --------------------------------------------------
            # Step 4b — Build failed inside the pipeline
            # --------------------------------------------------
            if job_status == "failed":
                error_msg = job_data.get("error", "Unknown pipeline error.")
                return {
                    "black_box_result": ExecutionResult(
                        method="black_box",
                        status="failure",
                        summary=f"Black-box pipeline failed: {error_msg}",
                        raw_output={
                            "job_id": job_id,
                            "error":  error_msg,
                        },
                    )
                }

            # --------------------------------------------------
            # Still processing — loop continues
            # --------------------------------------------------

        # ----------------------------------------------------------
        # Step 5 — Timeout exceeded
        # ----------------------------------------------------------
        return {
            "black_box_result": ExecutionResult(
                method="black_box",
                status="failure",
                summary=(
                    f"Black-box build timed out after "
                    f"{POLL_TIMEOUT_SECONDS // 60} minutes. "
                    "The pipeline may still be running on port 8001."
                ),
                raw_output={
                    "job_id":          job_id,
                    "elapsed_seconds": elapsed,
                },
            )
        }


# ----------------------------------------------------------
# Helper — build a structured summary string from real metrics
# ----------------------------------------------------------
def _build_summary(
    dataset_name: str,
    model_name: str,
    similarity: str,
    metrics: dict,
    images: list,
) -> str:
    """
    Produces a structured plain-text summary from the real pipeline outputs.
    This is what ReportGenerationNode receives as context — no LLM invention.
    """
    lines = [
        f"Black-box build completed successfully.",
        f"  Dataset   : {dataset_name}",
        f"  Model     : {model_name}",
        f"  Similarity: {similarity}",
        "",
        "Retrieval Metrics:",
    ]

    if metrics:
        metric_data = metrics.get("metrics", metrics)   # handle nested or flat
        for key, value in metric_data.items():
            # Format floats to 4 decimal places, pass others through
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
    else:
        lines.append("  No metrics available.")

    lines.append("")
    lines.append(f"Explainability images generated: {len(images)}")

    return "\n".join(lines)