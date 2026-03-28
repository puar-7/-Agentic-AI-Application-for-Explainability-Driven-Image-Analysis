import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace

load_dotenv()

# ------------------------------------------------------------------
# Model registry
#
# Each entry maps a provider name to the config needed to serve it.
# "source" tells the factory which backend to use:
#   "hf"     → HuggingFace Serverless Inference API
#   "sarvam" → Sarvam's own OpenAI-compatible API (api.sarvam.ai)
# ------------------------------------------------------------------
_MODEL_REGISTRY = {
    "llama": {
        "source":  "hf",
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "display": "Llama 3 8B Instruct (HuggingFace)",
    },
    "sarvam": {
        "source":  "sarvam",
        "repo_id": "sarvam-m",
        "display": "Sarvam-M (api.sarvam.ai)",
    },
}

_VALID_PROVIDERS = set(_MODEL_REGISTRY.keys())


# ------------------------------------------------------------------
# Backend: HuggingFace Serverless Inference
# ------------------------------------------------------------------

def _create_hf_chat(repo_id: str, temperature: float, max_new_tokens: int):
    """
    Instantiates a model via HuggingFace Serverless Inference API.

    No provider= override — the default endpoint (api-inference.huggingface.co)
    is what Llama 3 is available on for standard HF accounts. The provider=
    parameter routes through a different router layer and causes 404s for
    models not registered on that newer infrastructure.
    """
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HUGGINGFACEHUB_API_TOKEN not found in .env."
        )

    endpoint_llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        huggingfacehub_api_token=hf_token,
    )

    return ChatHuggingFace(llm=endpoint_llm)


# ------------------------------------------------------------------
# Backend: Sarvam API  (OpenAI-compatible)
# ------------------------------------------------------------------

def _create_sarvam_chat(model_id: str, temperature: float, max_new_tokens: int):
    """
    Instantiates Sarvam-M via Sarvam's own OpenAI-compatible API.

    sarvamai/sarvam-m is NOT hosted on HuggingFace Inference — the model
    card exists there but the serving endpoint is api.sarvam.ai.
    Sarvam's API follows the OpenAI chat completions spec, so ChatOpenAI
    with a custom base_url is the correct integration path.

    The returned ChatOpenAI object implements the same LangChain
    BaseChatModel interface as ChatHuggingFace — all downstream nodes
    call .invoke(messages) identically and never know which backend is
    underneath. No node changes required.

    Requires:
        SARVAM_API_KEY in .env   (get from https://dashboard.sarvam.ai)
        pip install langchain-openai
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai is required for Sarvam API. "
            "Run: pip install langchain-openai"
        )

    sarvam_key = os.getenv("SARVAM_API_KEY")
    if not sarvam_key:
        raise EnvironmentError(
            "SARVAM_API_KEY not found in .env. "
            "Get your key from https://dashboard.sarvam.ai and add:\n"
            "  SARVAM_API_KEY=your_key_here"
        )

    return ChatOpenAI(
        model=model_id,
        temperature=temperature,
        max_tokens=max_new_tokens,
        api_key=sarvam_key,
        base_url="https://api.sarvam.ai/v1",
    )


# ------------------------------------------------------------------
# Internal dispatcher — single place that routes provider → backend
# ------------------------------------------------------------------

def _create_llm(provider: str, temperature: float, max_new_tokens: int):
    config = _MODEL_REGISTRY[provider]
    source = config["source"]

    if source == "hf":
        return _create_hf_chat(config["repo_id"], temperature, max_new_tokens)

    if source == "sarvam":
        return _create_sarvam_chat(config["repo_id"], temperature, max_new_tokens)

    raise ValueError(
        f"Unknown source '{source}' in registry for provider '{provider}'."
    )


# ------------------------------------------------------------------
# Named model constructors — stable public API
# Kept so any code that imported these directly still works.
# ------------------------------------------------------------------

def get_chat_llm(temperature: float = 0.2, max_new_tokens: int = 512):
    """Llama 3 8B Instruct via HuggingFace Inference API."""
    return _create_llm("llama", temperature, max_new_tokens)


def get_sarvam_llm(temperature: float = 0.2, max_new_tokens: int = 512):
    """Sarvam-M via Sarvam's own API at api.sarvam.ai."""
    return _create_llm("sarvam", temperature, max_new_tokens)


# ------------------------------------------------------------------
# Unified selectors — the only functions nodes should call
# ------------------------------------------------------------------

def get_llm(temperature: float = 0.2, max_new_tokens: int = 512):
    """
    Selector for chat and report generation tasks.
    Reads LLM_PROVIDER from .env — defaults to 'llama'.

    Supported values (case-insensitive):
        llama   → Llama 3 8B via HuggingFace Inference API
        sarvam  → Sarvam-M via api.sarvam.ai

    Falls back to Llama with a warning if the configured provider fails.
    If Llama itself fails, the exception propagates — there is no further
    fallback from the default.
    """
    provider = os.getenv("LLM_PROVIDER", "llama").lower().strip()

    if provider not in _VALID_PROVIDERS:
        raise ValueError(
            f"[LLM Factory] Unknown LLM_PROVIDER='{provider}'. "
            f"Valid options: {sorted(_VALID_PROVIDERS)}"
        )

    try:
        print(f"[LLM Factory] Loading {_MODEL_REGISTRY[provider]['display']} for generation.")
        return _create_llm(provider, temperature, max_new_tokens)
    except Exception as e:
        if provider == "llama":
            raise   # no further fallback from the default
        print(
            f"\n[LLM Factory] WARNING: '{provider}' unavailable — falling back to Llama 3.\n"
            f"  Reason: {type(e).__name__}: {e}\n"
        )
        return _create_llm("llama", temperature, max_new_tokens)


def get_grader_llm(temperature: float = 0.0, max_new_tokens: int = 15):
    """
    Selector for the strict single-word grading task.
    Reads GRADER_LLM_PROVIDER from .env — defaults to 'llama'.

    Kept separate from get_llm() because the grader does binary
    classification with a very tight token budget. temperature and
    max_new_tokens are task-driven constraints, not model defaults.
    Llama 3 Instruct follows one-word format instructions more reliably
    than a generative multilingual model — hence the conservative default.
    """
    provider = os.getenv("GRADER_LLM_PROVIDER", "llama").lower().strip()

    if provider not in _VALID_PROVIDERS:
        raise ValueError(
            f"[LLM Factory] Unknown GRADER_LLM_PROVIDER='{provider}'. "
            f"Valid options: {sorted(_VALID_PROVIDERS)}"
        )

    try:
        print(f"[LLM Factory] Loading {_MODEL_REGISTRY[provider]['display']} for grading task.")
        return _create_llm(provider, temperature, max_new_tokens)
    except Exception as e:
        if provider == "llama":
            raise
        print(
            f"\n[LLM Factory] WARNING: '{provider}' grader unavailable — falling back to Llama 3.\n"
            f"  Reason: {type(e).__name__}: {e}\n"
        )
        return _create_llm("llama", temperature, max_new_tokens)


# ------------------------------------------------------------------
# Startup probe — call from server.py startup_event()
# ------------------------------------------------------------------

def validate_llm_providers() -> None:
    """
    Validates configured providers at server startup by attempting to
    construct the model client objects. Does NOT make an inference call —
    just verifies API keys are present and clients initialise cleanly.

    Surfaces missing keys, bad provider names, or import errors at boot
    time rather than on the first user request.
    """
    llm_provider    = os.getenv("LLM_PROVIDER",        "llama").lower().strip()
    grader_provider = os.getenv("GRADER_LLM_PROVIDER", "llama").lower().strip()

    print(
        f"\n[LLM Config] ──────────────────────────────────\n"
        f"  LLM_PROVIDER         = {llm_provider}"
        f"  →  {_MODEL_REGISTRY.get(llm_provider, {}).get('display', 'UNKNOWN')}\n"
        f"  GRADER_LLM_PROVIDER  = {grader_provider}"
        f"  →  {_MODEL_REGISTRY.get(grader_provider, {}).get('display', 'UNKNOWN')}\n"
        f"[LLM Config] ──────────────────────────────────\n"
    )

    for label, name in [("generation", llm_provider), ("grader", grader_provider)]:
        if name not in _VALID_PROVIDERS:
            raise ValueError(
                f"[LLM Config] Invalid provider '{name}' for {label}. "
                f"Valid options: {sorted(_VALID_PROVIDERS)}"
            )

    get_llm()
    get_grader_llm()

    print("[LLM Config] Provider validation complete.\n")