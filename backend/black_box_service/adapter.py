import os
import sys
import json
import time
from contextlib import contextmanager

# ---------------------------------------------------------
# PATH HACK (Fixes internal "core" and "explainability" imports)
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.abspath(os.path.join(current_dir, "..", "black_box_core"))

if core_dir not in sys.path:
    sys.path.insert(0, core_dir)

# ---------------------------------------------------------
# IMPORT BLACK BOX CODE
# ---------------------------------------------------------
import runner
from runner import Runner
from inference import Inference
from generate_metadata import MetadataGenerator

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
# All job status files live here (relative to workspace root)
JOBS_DIR = "storage/jobs"


# ---------------------------------------------------------
# 1. CONTEXT MANAGER — Safe Working Directory
# ---------------------------------------------------------
@contextmanager
def safe_workspace(workspace_path):
    """
    Temporarily changes CWD so all relative paths inside the
    black_box_core code resolve correctly inside our workspace.
    """
    original_cwd = os.getcwd()
    os.makedirs(workspace_path, exist_ok=True)
    os.chdir(workspace_path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


# ---------------------------------------------------------
# 2. CLASS OVERRIDE — Fix Hardcoded Output Path
# ---------------------------------------------------------
class SafeMetadataGenerator(MetadataGenerator):
    """
    Overrides the hardcoded 'C:\\Users\\...' metadata output path
    with a path relative to the current safe workspace.
    """
    def __init__(self, dataset_path, dataset_name):
        super().__init__(dataset_path, dataset_name)
        self.output_dir = os.path.abspath("./metadata")
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_path = os.path.join(
            self.output_dir, f"{self.dataset_name}_metadata.pkl"
        )

# Inject our safe class so runner.py uses it transparently
runner.MetadataGenerator = SafeMetadataGenerator


# ---------------------------------------------------------
# 3. JOB FILE HELPERS
# ---------------------------------------------------------

def _jobs_dir(workspace_dir: str) -> str:
    """Returns the absolute path to the jobs directory."""
    return os.path.join(workspace_dir, JOBS_DIR)


def _job_path(workspace_dir: str, job_id: str) -> str:
    return os.path.join(_jobs_dir(workspace_dir), f"{job_id}.json")


def write_job(workspace_dir: str, job_id: str, payload: dict) -> None:
    """Atomically write a job status file."""
    os.makedirs(_jobs_dir(workspace_dir), exist_ok=True)
    path = _job_path(workspace_dir, job_id)
    # Write to a temp file then rename for atomicity
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def read_job(workspace_dir: str, job_id: str) -> dict:
    """Read a job status file. Returns None if it doesn't exist yet."""
    path = _job_path(workspace_dir, job_id)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------
# 4. IMAGE HARVESTING
# ---------------------------------------------------------

def harvest_results(workspace_dir: str, n: int = 5) -> list:
    """
    Scans the outputs/heatmaps directory and returns up to n image pairs
    where BOTH the overlay AND the heatmap exist.

    Returns a list of dicts with URL paths (not OS paths) so the
    frontend can render them via the static file server on port 8000.
    """
    overlay_dir = os.path.join(
        workspace_dir, "outputs", "heatmaps", "overlays"
    )
    heatmap_dir = os.path.join(
        workspace_dir, "outputs", "heatmaps", "heatmap_images"
    )

    if not os.path.exists(overlay_dir):
        return []

    pairs = []

    for fname in sorted(os.listdir(overlay_dir)):
        if not fname.endswith("_overlay.jpg"):
            continue

        stem = fname.replace("_overlay.jpg", "")
        heatmap_fname = f"{stem}_heatmap.jpg"
        heatmap_full = os.path.join(heatmap_dir, heatmap_fname)

        # Only include this image if BOTH files exist
        if not os.path.exists(heatmap_full):
            continue

        pairs.append({
            "image_id": stem,
            # URL paths served by port 8000's static mount at /outputs
            "overlay_url": f"/outputs/heatmaps/overlays/{fname}",
            "heatmap_url": f"/outputs/heatmaps/heatmap_images/{heatmap_fname}",
        })

        if len(pairs) == n:
            break

    return pairs


# ---------------------------------------------------------
# 5. METRICS LOADER
# ---------------------------------------------------------

def load_metrics(workspace_dir: str, model_name: str, dataset_name: str,
                 similarity: str) -> dict:
    """
    Reads the metrics JSON produced by the evaluation step.
    Returns an empty dict if the file doesn't exist.
    """
    metrics_file = os.path.join(
        workspace_dir,
        "outputs", "reports", "metric_reports",
        f"{model_name.lower()}_{dataset_name.lower()}_{similarity.lower()}_metrics.json"
    )
    if not os.path.exists(metrics_file):
        return {}
    with open(metrics_file, "r") as f:
        return json.load(f)


# ---------------------------------------------------------
# 6. BUILD — The Heavy Pipeline
# ---------------------------------------------------------

def run_blackbox_build(
    workspace_dir: str,
    job_id: str,
    dataset_name: str = "CELEBA",
    model_name: str = "RESNET",
    similarity: str = "COSINE",
    explainer: str = "LIME",
):
    """
    Runs the full black-box build pipeline inside the safe workspace.

    Writes job status files at three points:
        - On start     → status: processing
        - On success   → status: completed  (includes metrics + harvested images)
        - On failure   → status: failed     (includes error message)

    This function is designed to be run in a FastAPI BackgroundTask.
    """

    # --- Mark job as started ---
    write_job(workspace_dir, job_id, {
        "job_id": job_id,
        "status": "processing",
        "started_at": time.time(),
        "dataset_name": dataset_name,
        "model_name": model_name,
        "similarity": similarity,
        "explainer": explainer,
    })

    print(f"[Adapter] Build started — job_id={job_id}")

    try:
        with safe_workspace(workspace_dir):
            pipeline_runner = Runner()

            # Bypass CLI prompts by injecting values directly
            pipeline_runner.dataset_name = dataset_name
            pipeline_runner.model_name = model_name
            pipeline_runner.similarity_measure = similarity
            pipeline_runner.explain_model = explainer

            # THE HEAVY WORK — wrapped in try/except
            pipeline_runner.run()

        # --- Harvest outputs (paths resolved from workspace root) ---
        metrics = load_metrics(workspace_dir, model_name, dataset_name, similarity)
        images = harvest_results(workspace_dir, n=5)

        # --- Mark job as completed ---
        write_job(workspace_dir, job_id, {
            "job_id": job_id,
            "status": "completed",
            "started_at": time.time(),
            "dataset_name": dataset_name,
            "model_name": model_name,
            "similarity": similarity,
            "explainer": explainer,
            "metrics": metrics,
            "images": images,
        })

        print(f"[Adapter] Build completed — job_id={job_id} | "
              f"images_harvested={len(images)}")

    except MemoryError:
        msg = "Build failed: ran out of memory. Try a smaller dataset or batch size."
        print(f"[Adapter] MemoryError — job_id={job_id}")
        write_job(workspace_dir, job_id, {
            "job_id": job_id,
            "status": "failed",
            "error": msg,
        })

    except FileNotFoundError as e:
        msg = f"Build failed: required file not found — {e}"
        print(f"[Adapter] FileNotFoundError — job_id={job_id}: {e}")
        write_job(workspace_dir, job_id, {
            "job_id": job_id,
            "status": "failed",
            "error": msg,
        })

    except Exception as e:
        msg = f"Build failed: unexpected error — {type(e).__name__}: {e}"
        print(f"[Adapter] Unexpected error — job_id={job_id}: {e}")
        write_job(workspace_dir, job_id, {
            "job_id": job_id,
            "status": "failed",
            "error": msg,
        })


# ---------------------------------------------------------
# 7. INFERENCE (Unchanged from original)
# ---------------------------------------------------------

def run_blackbox_inference(
    workspace_dir: str,
    query_image_path: str,
    dataset_name: str = "CELEBA",
    model_name: str = "RESNET",
    similarity: str = "COSINE",
):
    """
    Runs inference using the pre-built index.
    Kept intact — infer mode is not being modified.
    """
    print("[Adapter] Starting Black-Box Inference")

    with safe_workspace(workspace_dir):
        infer = Inference(
            dataset_name=dataset_name,
            model_name=model_name,
            similarity=similarity,
            top_k=5,
        )
        infer.load_system()
        results = infer.predict(query_image_path)

    return {
        "status": "success",
        "matches": results,
    }