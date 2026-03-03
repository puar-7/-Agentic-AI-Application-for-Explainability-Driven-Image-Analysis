import os
import sys
import json
from contextlib import contextmanager

# ---------------------------------------------------------
# PATH HACK (Fixes her internal "core" and "explainability" imports)
# ---------------------------------------------------------
# Get the absolute path to the backend/black_box_core directory
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.abspath(os.path.join(current_dir, "..", "black_box_core"))


#sys.path --> list of directories Python searches for imports. We need to add core_dir to this list so her imports work.
# Inject it into Python's path so her code thinks it is at the root
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)

# ---------------------------------------------------------
# IMPORT HER CODE
# ---------------------------------------------------------
import runner
from runner import Runner
from inference import Inference
from generate_metadata import MetadataGenerator
# ---------------------------------------------------------
# 1. THE CONTEXT MANAGER (Corral Relative Paths)
# ---------------------------------------------------------
@contextmanager
def safe_workspace(workspace_path):
    """
    Temporarily changes the Current Working Directory (CWD).
    When her code runs `os.makedirs("embeddings")`, it will safely 
    happen inside our designated workspace instead of the root directory.
    """
    original_cwd = os.getcwd() #get current dir
    os.makedirs(workspace_path, exist_ok=True) #make if not exist
    os.chdir(workspace_path)#change dir
    try:
        yield # pauses exection here and runs the block inside the "with" statement
    finally:
        os.chdir(original_cwd) #always run so restore directory

# ---------------------------------------------------------
# 2. THE CLASS OVERRIDE (Fixing the C:\ Drive Hardcode)
# ---------------------------------------------------------
class SafeMetadataGenerator(MetadataGenerator):
    """
    Inherits from her exact class, but intercepts and safely rewrites
    the hardcoded 'C:\\Users\\abdul\\...' paths immediately after instantiation.
    """
    def __init__(self, dataset_path, dataset_name):
        # Call her original __init__ to set up everything she needs
        super().__init__(dataset_path, dataset_name)
        
        # INSTANTLY override the hardcoded attributes with our safe paths
        self.output_dir = os.path.abspath("./metadata") # Relative to the safe workspace
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_path = os.path.join(self.output_dir, f"{self.dataset_name}_metadata.pkl")

# Inject our Safe class into her runner module so her code uses it without knowing!
runner.MetadataGenerator = SafeMetadataGenerator


# ---------------------------------------------------------
# 3. ADAPTER FUNCTIONS (Bypassing CLI & Printing)
# ---------------------------------------------------------

def run_blackbox_build(
    workspace_dir: str, 
    dataset_name: str = "CELEBA", 
    model_name: str = "RESNET", 
    similarity: str = "COSINE", 
    explainer: str = "LIME"
):
    """
    Wraps her `Runner.run()` build mode. Bypasses the CLI inputs.
    """
    print(f"--- Adapter: Starting Black-Box Build in {workspace_dir} ---")
    
    with safe_workspace(workspace_dir):
        # Instantiate her runner
        pipeline_runner = Runner()
        
        # BYPASS CLI: Manually inject the properties she normally gets via input()
        pipeline_runner.dataset_name = dataset_name
        pipeline_runner.model_name = model_name
        pipeline_runner.similarity_measure = similarity
        pipeline_runner.explain_model = explainer
        
        # Execute her heavy pipeline safely
        pipeline_runner.run()
        
        # HARVEST OUTPUTS: Read her saved JSON metrics to return them as data
        metrics_file = os.path.join(
            "outputs", "reports", "metric_reports", 
            f"{model_name.lower()}_{dataset_name.lower()}_{similarity.lower()}_metrics.json"
        )
        
        results = {"status": "success", "metrics": {}}
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                results["metrics"] = json.load(f)
                
        return results

def run_blackbox_inference(
    workspace_dir: str, 
    query_image_path: str,
    dataset_name: str = "CELEBA", 
    model_name: str = "RESNET", 
    similarity: str = "COSINE"
):
    """
    Wraps her `Inference` class. 
    This is just a utility function for testing. In Phase 2, we will pull 
    the `load_system()` part out into the FastAPI server startup to save RAM.
    """
    print(f"--- Adapter: Starting Black-Box Inference ---")
    
    with safe_workspace(workspace_dir):
        infer = Inference(
            dataset_name=dataset_name,
            model_name=model_name,
            similarity=similarity,
            top_k=5
        )
        
        # Note: This is slow! We will fix this in Phase 2.
        infer.load_system()
        
        # Get the actual returned array instead of just reading her prints
        results = infer.predict(query_image_path)
        
        return {
            "status": "success",
            "matches": results
        }