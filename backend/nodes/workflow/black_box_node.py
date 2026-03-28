# backend/nodes/workflow/black_box_node.py
#
# Drives the black-box build pipeline via MCP.
#
# Change from original:
#   - httpx removed entirely
#   - mcp.client.sse + ClientSession replace raw HTTP calls
#   - BLACK_BOX_API_URL replaced with BLACK_BOX_MCP_URL (env-var driven)
#   - Polling loop structure, timeout logic, ExecutionResult packaging
#     are all identical to the original

import asyncio
import json
import os
from typing import Dict

from mcp.client.sse import sse_client
from mcp import ClientSession

from backend.graph.state import GraphState
from backend.schemas.execution_result import ExecutionResult

# ----------------------------------------------------------
# MCP server URL
# Override via environment variable for Docker / remote deployment.
# Port 8001 mounts the MCP server at /mcp  →  SSE at /mcp/sse
# ----------------------------------------------------------
BLACK_BOX_MCP_URL = os.getenv(
    "BLACK_BOX_MCP_URL",
    "http://localhost:8001/mcp/sse"
)

# Polling configuration — unchanged from original
POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS  = 1800   # 30 minutes


class BlackBoxNode:
    """
    Async LangGraph node that drives the black-box build pipeline via MCP.

    Flow:
        1. Call run_build MCP tool  → receives job_id immediately
        2. Poll get_job_status MCP tool every POLL_INTERVAL_SECONDS
        3. On "completed" → package metrics + image URLs into ExecutionResult
        4. On "failed"    → package error into ExecutionResult (no exception
                            raise, so ReportGenerationNode still runs)

    Each MCP call opens its own SSE session. This is intentional — it avoids
    SSE connection timeouts during long builds (which can take 30+ minutes)
    without requiring any keepalive logic.
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

        build_payload = {
            "dataset_name": wi.dataset_name,
            "model_name":   wi.model_name,
            "similarity":   wi.similarity,
            "explainer":    wi.explainer,
        }

        print(
            f"[BlackBoxNode] Firing build via MCP — "
            f"dataset={wi.dataset_name} model={wi.model_name} "
            f"similarity={wi.similarity} explainer={wi.explainer}"
        )

        # ----------------------------------------------------------
        # Step 1 — Call run_build tool, get job_id back immediately
        # ----------------------------------------------------------
        try:
            async with sse_client(BLACK_BOX_MCP_URL) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    build_result = await session.call_tool(
                        "run_build",
                        arguments=build_payload,
                    )

        except Exception as e:
            return {
                "black_box_result": ExecutionResult(
                    method="black_box",
                    status="failure",
                    summary=(
                        f"Could not connect to the Black-Box MCP server at "
                        f"{BLACK_BOX_MCP_URL}. Is port 8001 running? "
                        f"Error: {e}"
                    ),
                    raw_output={"error": str(e)},
                )
            }

        # MCP tool itself returned an error
        if build_result.isError:
            error_text = (
                build_result.content[0].text
                if build_result.content
                else "Unknown MCP tool error"
            )
            return {
                "black_box_result": ExecutionResult(
                    method="black_box",
                    status="failure",
                    summary=f"run_build MCP tool returned an error: {error_text}",
                    raw_output={"error": error_text},
                )
            }

        build_data = json.loads(build_result.content[0].text)
        job_id = build_data.get("job_id")

        if not job_id:
            return {
                "black_box_result": ExecutionResult(
                    method="black_box",
                    status="failure",
                    summary="run_build tool did not return a job_id.",
                    raw_output={"response": build_data},
                )
            }

        print(f"[BlackBoxNode] Build accepted — job_id={job_id}")

        # ----------------------------------------------------------
        # Step 2 — Poll get_job_status tool until done or timeout
        #
        # Each poll opens a fresh MCP session. This avoids SSE timeout
        # issues during long builds without any keepalive complexity.
        # ----------------------------------------------------------
        elapsed = 0

        while elapsed < POLL_TIMEOUT_SECONDS:

            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            elapsed += POLL_INTERVAL_SECONDS

            # ---- Poll ----
            try:
                async with sse_client(BLACK_BOX_MCP_URL) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        status_result = await session.call_tool(
                            "get_job_status",
                            arguments={"job_id": job_id},
                        )

            except Exception as e:
                # Transient connection failure — keep polling, same as
                # original ConnectError handling on httpx
                print(
                    f"[BlackBoxNode] Poll connection failed — "
                    f"job_id={job_id} elapsed={elapsed}s: {e}. Retrying..."
                )
                continue

            # MCP tool error on status check — transient, keep polling
            if status_result.isError:
                print(
                    f"[BlackBoxNode] get_job_status tool error — "
                    f"job_id={job_id} elapsed={elapsed}s. Retrying..."
                )
                continue

            job_data   = json.loads(status_result.content[0].text)
            job_status = job_data.get("status")

            print(
                f"[BlackBoxNode] Poll — job_id={job_id} "
                f"status={job_status} elapsed={elapsed}s"
            )

            # Job file not yet written — background thread hasn't started
            # writing yet. Same race as original 404 handling on httpx.
            if job_status == "not_found" and elapsed <= POLL_INTERVAL_SECONDS * 2:
                print(
                    f"[BlackBoxNode] Job file not yet written — "
                    f"job_id={job_id}. Retrying..."
                )
                continue

            # --------------------------------------------------
            # Build completed successfully
            # --------------------------------------------------
            if job_status == "completed":
                metrics = job_data.get("metrics", {})
                images  = job_data.get("images",  [])

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
                            "job_id":       job_id,
                            "dataset_name": wi.dataset_name,
                            "model_name":   wi.model_name,
                            "similarity":   wi.similarity,
                            "explainer":    wi.explainer,
                            "metrics":      metrics,
                            "images":       images,
                        },
                    )
                }

            # --------------------------------------------------
            # Build failed inside the pipeline
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

            # Still processing — loop continues

        # ----------------------------------------------------------
        # Timeout exceeded
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
# Helper — unchanged from original
# ----------------------------------------------------------
def _build_summary(
    dataset_name: str,
    model_name: str,
    similarity: str,
    metrics: dict,
    images: list,
) -> str:
    lines = [
        "Black-box build completed successfully.",
        f"  Dataset   : {dataset_name}",
        f"  Model     : {model_name}",
        f"  Similarity: {similarity}",
        "",
        "Retrieval Metrics:",
    ]

    if metrics:
        metric_data = metrics.get("metrics", metrics)
        for key, value in metric_data.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
    else:
        lines.append("  No metrics available.")

    lines.append("")
    lines.append(f"Explainability images generated: {len(images)}")

    return "\n".join(lines)