"""
Web page browsing + summarization helpers.

This module provides:
- `jina_readpage(url, ...)`: fetch page content via a read-proxy
- `get_browse_summary(prompt, ...)`: summarize using an LLM provider

Provider identifiers are intentionally omitted from file/class/function names
for clean open-source distribution.
"""

import os
import random
import time
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import requests
except ImportError:
    requests = None


def _get_jina_proxies() -> Optional[dict]:
    """Build request proxies for the read-proxy."""
    # Open-source build: do not use any proxy settings.
    return None


def jina_readpage(url: str, max_retry: int = 10) -> str:
    """Fetch page content via the read-proxy."""
    if not requests:
        return "[browse] requests library not available."

    jina_key = os.environ.get("JINA_API_KEYS", "")
    if not jina_key:
        return "[browse] JINA_API_KEYS environment variable is not set."

    proxies = _get_jina_proxies()

    for attempt in range(max_retry):
        try:
            headers = {"Authorization": f"Bearer {jina_key}"}
            response = requests.get(
                f"https://r.jina.ai/{url}",
                headers=headers,
                timeout=50,
                proxies=proxies,
            )
            if response.status_code == 200:
                return response.text
            if response.status_code == 429:
                wait_time = 4 + random.uniform(2, 4)
                print(
                    f"[Browse] jina_readpage 429 Too Many Requests, retrying in {wait_time:.2f}s url={url!r}",
                    flush=True,
                )
                time.sleep(wait_time)
                continue
            raise ValueError(f"jina readpage error: {response.text}")
        except Exception as e:
            print(f"[Browse] jina_readpage attempt={attempt} url={url!r} error: {e}", flush=True)
            if attempt == max_retry - 1:
                return "[browse] Failed to read page."
            time.sleep(0.5)

    return "[browse] Failed to read page."


def get_browse_summary(
    prompt: str,
    generate_engine: str = "deepseekchat",
    max_retry: int = 10,
) -> Optional[str]:
    if not OpenAI:
        return None

    if generate_engine == "deepseekchat":
        # Keep the engine option, but do not depend on DEEPSEEK_* env vars.
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("BROWSE_SUMMARY_API_KEY") or os.environ.get("API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("API_BASE") or "https://api.deepseek.com/v1"
        model = os.environ.get("BROWSE_SUMMARY_MODEL", "deepseek-chat") or "deepseek-chat"
    elif generate_engine == "vllm":
        api_base = os.environ.get("BROWSE_SUMMARY_BASE_URL", "").strip()
        model = os.environ.get("BROWSE_SUMMARY_MODEL", "Qwen3").strip() or "Qwen3"
        api_key = os.environ.get("BROWSE_SUMMARY_API_KEY", "EMPTY")
        if not api_base:
            return None
    elif generate_engine == "geminiflash":
        api_key = os.environ.get("GEMINI_API_KEY")
        api_base = os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
        model = "gemini-2.0-flash-exp"
    elif generate_engine == "openai":
        api_key = os.environ.get("API_KEY")
        api_base = os.environ.get("API_BASE")
        model = os.environ.get("SUMMARY_MODEL_NAME", "gpt-3.5-turbo-0125")
    else:
        return None

    if not api_base:
        return None
    if generate_engine != "vllm" and not api_key:
        return None

    client = OpenAI(api_key=api_key, base_url=api_base)
    for attempt in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8192,
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                return content
        except Exception as e:
            print(
                f"[Browse] get_browse_summary attempt={attempt} engine={generate_engine} error: {e}",
                flush=True,
            )
            if attempt == max_retry - 1:
                return None
            time.sleep(random.uniform(1, 4))

    return None

