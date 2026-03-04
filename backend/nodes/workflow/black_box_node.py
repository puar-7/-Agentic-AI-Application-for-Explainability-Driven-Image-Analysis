import os
import requests
from typing import Dict
from backend.graph.state import GraphState
from backend.schemas.execution_result import ExecutionResult

# The URL of our new FastAPI microservice
BLACK_BOX_API_URL = "http://localhost:8001"

class BlackBoxNode:
    def __call__(self, state: GraphState) -> Dict:
        if not state.workflow_input:
            return {
                "error": "Workflow input missing for black-box execution."
            }

        print("--- LangGraph: Triggering Black-Box Node ---")

        # ---------------------------------------------------------
        # THE PLACEHOLDER HACK
        # Mapping tabular inputs to her image-retrieval needs.
        # ---------------------------------------------------------
        
        # Since the UI doesn't provide an image yet, we hardcode a fallback 
        # path pointing to the first image in the CelebA dataset.
        # 
        fallback_image_path = os.path.abspath(
            "./backend/storage/shared_workspace/data/celeba/000001.jpg"
        )
        
        payload = {
            "query_image_path": fallback_image_path
        }

        try:
            # 1. Make the API Call to our new Microservice
            print(f"📡 Sending request to {BLACK_BOX_API_URL}/infer...")
            response = requests.post(
                f"{BLACK_BOX_API_URL}/infer", 
                json=payload, 
                timeout=15  # 15 second timeout just in case
            )
            response.raise_for_status() # Raises an error if the status is 4xx or 5xx
            
            api_data = response.json()
            matches = api_data.get("matches", [])
            
            # 2. Package the API response into our strict ExecutionResult schema
            result = ExecutionResult(
                method="black_box",
                status="success",
                summary=f"Black-Box API connected successfully! Found {len(matches)} matches.",
                raw_output={
                    "note": "Used placeholder image to bypass tabular input mismatch.",
                    "original_tabular_inputs": {
                        "dataset_path": state.workflow_input.dataset_path,
                        "target_variable": state.workflow_input.target_variable
                    },
                    "api_matches": matches
                }
            )

        except requests.exceptions.ConnectionError:
            # If forgot to start the FastAPI server!
            result = ExecutionResult(
                method="black_box",
                status="failure",
                summary="Failed to connect to the Black-Box Microservice.",
                raw_output={"error": "Is the FastAPI server running on port 8001?"}
            )
        except requests.exceptions.RequestException as e:
            # If the API crashed or returned a 500 error
            result = ExecutionResult(
                method="black_box",
                status="failure",
                summary="Black-Box API returned an error.",
                raw_output={"error": str(e)}
            )

        return {
            "black_box_result": result
        }