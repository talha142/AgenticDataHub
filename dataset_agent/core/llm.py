"""
core/llm.py
───────────
Single factory that returns a LangChain ChatOpenAI instance pointed at
your local vLLM server.  Every agent imports `get_llm()` — swap the
env vars and the whole system moves to a different model/server.
"""

import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


@lru_cache(maxsize=1)
def get_llm(
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> ChatOpenAI:
    """
    Returns a ChatOpenAI client wired to a vLLM OpenAI-compatible endpoint.

    vLLM exposes POST /v1/chat/completions with the same schema as OpenAI,
    so langchain_openai.ChatOpenAI works without any changes — just point
    base_url at your server and set the model name to whatever you loaded.

    Environment variables (see .env.example):
        VLLM_BASE_URL   – e.g. http://localhost:8000/v1
        VLLM_API_KEY    – any string (vLLM ignores it unless --api-key is set)
        VLLM_MODEL      – exact model name vLLM was started with
    """
    port = int(os.getenv("VLLM_PORT", "8000"))
    base_url = os.getenv("VLLM_BASE_URL", f"http://localhost:{port}/v1")
    api_key  = os.getenv("VLLM_API_KEY",  "not-needed")
    model    = os.getenv("VLLM_MODEL",    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")

    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        # vLLM supports streaming; keep it on for long scrape summaries
        streaming=True,
    )
