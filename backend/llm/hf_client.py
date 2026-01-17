import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace


# Load environment variables from .env
load_dotenv()


def get_chat_llm(
    repo_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    temperature: float = 0.2,
    max_new_tokens: int = 512,
):
    """
    Creates a Chat LLM using Hugging Face Hosted Inference API.
    """

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HUGGINGFACEHUB_API_TOKEN not found. "
            "Make sure it is set in the .env file."
        )

    endpoint_llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        huggingfacehub_api_token=hf_token,
    )

    return ChatHuggingFace(llm=endpoint_llm)
