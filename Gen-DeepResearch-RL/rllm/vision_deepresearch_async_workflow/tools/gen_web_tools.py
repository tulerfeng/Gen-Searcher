"""
Open-source friendly web tools for Vision-DeepResearch.

Provides three DeepResearchTool instances:
- search: text search
- image_search: image search + local download
- visit: web page fetch + summarization

This layer only defines tool interfaces and delegates heavy logic to
`gen_universal_image_search_impl` and `gen_jina_browse_impl`.
"""

from __future__ import annotations

import os
import random
import re
import time
from typing import List, Optional, Union

try:
    import requests
except ImportError:
    requests = None  # type: ignore

from vision_deepresearch_async_workflow.tools.shared import DeepResearchTool


def _clean_html_b(text: str) -> str:
    return re.sub(r"</?b>", "", text or "")


def _text_search_sync(queries: List[str], topk: int = 10, max_retry: int = 100) -> str:
    """Blocking web search: POST to TEXT_SEARCH_API_BASE_URL (e.g. Serper /search), markdown formatted."""
    if requests is None:
        return "[Search] requests is not installed."

    api_key = (os.environ.get("SERPER_KEY_ID") or "").strip()
    if not api_key:
        return "[Search] SERPER_KEY_ID is not set."

    url = (os.environ.get("TEXT_SEARCH_API_BASE_URL") or "").strip()
    if not url:
        return "[Search] TEXT_SEARCH_API_BASE_URL is not set."

    topk = min(10, max(1, topk))
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    results: List[str] = []
    for query in queries:
        q_clean = (query or "").replace('"', "").replace("'", "")
        payload = {"q": q_clean, "num": topk}
        appended = False

        for retry in range(max_retry):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=30, proxies=None)

                if resp.status_code == 429:
                    wait = 3 + random.uniform(1, 4)
                    print(
                        f"[Search] 429 Too Many Requests, retrying in {wait:.2f}s query={q_clean!r}",
                        flush=True,
                    )
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    print(
                        f"[Search] {resp.status_code} server error, retry={retry} query={q_clean!r}",
                        flush=True,
                    )
                    if retry >= 15:
                        results.append(
                            f"Search failed for '{q_clean}': HTTP {resp.status_code} Server Error"
                        )
                        appended = True
                        break
                    time.sleep(3 + random.uniform(1, 4))
                    continue

                resp.raise_for_status()
                data = resp.json()
                # Some error responses are HTTP 200 with error payload (see serper.dev docs).
                _err = None
                if isinstance(data, dict):
                    _err = data.get("message") or data.get("error")
                if isinstance(data, dict) and _err and not data.get("organic"):
                    raise RuntimeError(f"Serper search error: {_err}")

                # Serper returns organic[]; use `or []` so JSON null becomes []
                organic = data.get("organic") or []
                if not isinstance(organic, list):
                    print(f"[Search] invalid 'organic' in response for query={q_clean!r}", flush=True)
                    results.append(f"No results for '{q_clean}'.")
                    appended = True
                    break

                snippets: List[str] = []
                for page in organic[:topk]:
                    title = _clean_html_b(page.get("title", ""))
                    link = page.get("link", "")
                    snippet = _clean_html_b(page.get("snippet", ""))
                    snippets.append(f"[{title}]({link}) {snippet}")

                results.append("\n\n".join(snippets) if snippets else f"No results for '{q_clean}'.")
                appended = True
                break

            except Exception as e:
                print(f"[Search] _text_search_sync retry={retry} query={q_clean!r} error: {e}", flush=True)
                if retry == max_retry - 1:
                    results.append(f"Search failed for '{q_clean}': {e}")
                    appended = True
                else:
                    time.sleep(3 + random.uniform(1, 4))

        if not appended:
            results.append(
                f"Search failed for '{q_clean}': exhausted {max_retry} retries "
                "(rate limit, server errors, or no response)."
            )

    return "\n\n".join(
        f"--- search result for [{q}] ---\n{r}\n--- end of search result ---"
        for q, r in zip(queries, results)
    )


def _image_search_sync(
    query: str,
    top_k: int = 10,
    save_dir: str = "saved_img",
    sample_id: Optional[Union[int, str]] = None,
    max_retry: int = 50,
) -> str:
    """Run image search and return formatted markdown."""
    try:
        from vision_deepresearch_async_workflow.tools.gen_universal_image_search_impl import search_universal_image
    except ImportError as e:
        print(f"[ImageSearch] gen_universal_image_search_impl not available: {e}", flush=True)
        return "[ImageSearch] image search backend is not available."

    query = (query or "").strip()
    if not query:
        return "[ImageSearch] 'query' must be a non-empty string."

    try:
        print(f"[ImageSearch] Calling backend query={query!r} top_k={top_k}", flush=True)
        results = search_universal_image(
            query=query,
            topk=min(max(1, top_k), 20),
            max_retry=max_retry,
            save_dir=save_dir,
            sample_id=sample_id,
        )
        urls = [r.get("url", "") for r in results if r.get("url")]
        print(f"[ImageSearch] Returned {len(results)} images, urls: {urls}", flush=True)
    except Exception as e:
        print(f"[ImageSearch] Error: {e}", flush=True)
        return f"[ImageSearch] Error: {str(e)}"

    if not results:
        print(f"[ImageSearch] No image results for query={query!r}", flush=True)
        return (
            f"[ImageSearch] No image results found for '{query}'. "
            "Your search query is too long or too specific. "
            "You MUST retry image_search using <= 3 words that describe only the main visual subject "
            '(e.g. "person name", "city skyline", "logo").'
        )

    lines: List[str] = [f"--- image search result for [{query}] ---"]
    for i, r in enumerate(results, start=1):
        title = r.get("title", "image")
        url = r.get("url", "")
        local_path = r.get("local_path", "")
        page_url = r.get("page_url", "")
        lines.append(f"{i}. title: {title}")
        lines.append(f"   url: {url}")
        lines.append(f"   local_path: {local_path}")
        if page_url:
            lines.append(f"   page_url: {page_url}")
    lines.append("--- end of image search result ---")
    return "\n".join(lines)


def _browse_sync(
    url: str,
    query: str,
    read_engine: str = "jina",
    generate_engine: str = "deepseekchat",
    max_retry: int = 10,
) -> str:
    """Fetch page via read-proxy and summarize with an LLM."""
    # Optional random delay to spread traffic.
    if os.environ.get("BROWSE_RANDOM_SLEEP", "").strip().lower() in ("1", "true", "yes"):
        time.sleep(random.uniform(0, 16))

    if read_engine != "jina":
        return "[Browse] Only jina read engine is supported in the open-source version."

    try:
        from vision_deepresearch_async_workflow.tools.gen_jina_browse_impl import jina_readpage
    except ImportError:
        jina_readpage = None

    if jina_readpage is None:
        return "[Browse] browse backend is not available."

    try:
        source_text = jina_readpage(url, max_retry=max_retry)
    except Exception as e:
        print(f"[Browse] jina_readpage error url={url!r}: {e}", flush=True)
        return "Browse error. Please try again."

    if not source_text.strip() or source_text.startswith("[browse] Failed"):
        print(f"[Browse] Empty or failed read for url={url!r}", flush=True)
        return "Browse error. Please try again."

    browse_query = query or "Detailed summary of the page."

    try:
        from vision_deepresearch_async_workflow.tools.gen_jina_browse_impl import get_browse_summary
    except ImportError:
        get_browse_summary = None

    def _single_summary(prompt: str) -> Optional[str]:
        if get_browse_summary is None:
            return None
        return get_browse_summary(prompt, generate_engine=generate_engine)

    output: Optional[str] = None
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        tokenized = encoding.encode(source_text)
        if len(tokenized) > 95000 and get_browse_summary is not None:
            output = (
                "Since the content is too long, the result is split and answered separately. "
                "Please combine the results to get the complete answer.\n"
            )
            num_split = max(2, len(tokenized) // 95000 + 1)
            chunk_len = len(tokenized) // num_split
            outputs: List[str] = []
            for i in range(num_split):
                start_idx = i * chunk_len
                end_idx = min(start_idx + chunk_len + 1024, len(tokenized))
                source_text_i = encoding.decode(tokenized[start_idx:end_idx])
                prompt_i = (
                    "Please read the source content and answer a following question:\n"
                    f"--- begin of source content ---\n{source_text_i}\n--- end of source content ---\n\n"
                    "If there is no relevant information, please clearly refuse to answer. "
                    f"Now answer the question based on the above content:\n{browse_query}"
                )
                out_i = _single_summary(prompt_i)
                outputs.append(out_i or "")
            for i in range(num_split):
                output += (
                    f"--- begin of result part {i + 1} ---\n{outputs[i]}\n"
                    f"--- end of result part {i + 1} ---\n\n"
                )
        else:
            prompt = (
                "Please read the source content and answer a following question:\n"
                f"---begin of source content---\n{source_text}\n---end of source content---\n\n"
                "If there is no relevant information, please clearly refuse to answer. "
                f"Now answer the question based on the above content:\n{browse_query}"
            )
            output = _single_summary(prompt)
    except ImportError:
        prompt = (
            "Please read the source content and answer a following question:\n"
            f"---begin of source content---\n{source_text}\n---end of source content---\n\n"
            "If there is no relevant information, please clearly refuse to answer. "
            f"Now answer the question based on the above content:\n{browse_query}"
        )
        output = _single_summary(prompt)

    if output is None:
        output = (source_text[:120000] + "...") if len(source_text) > 120000 else source_text
    if not (output and str(output).strip()):
        print(f"[Browse] get_browse_summary returned empty for url={url!r}", flush=True)
        return "Browse error. Please try again."
    return str(output).strip()


class WebTextSearchTool(DeepResearchTool):
    """Text search tool (Serper Google web search API)."""

    def __init__(self):
        super().__init__(
            name="search",
            description="Perform web (text) searches. Supply an array 'query' of search strings; returns top results for each.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of search queries.",
                    },
                    "top_k": {"type": "integer", "description": "Max results per query (default 10)."},
                },
                "required": ["query"],
            },
        )

    async def call(self, query: Union[str, List[str]] = None, top_k: int = 10, **kwargs) -> str:
        raw = kwargs.get("queries") if "queries" in kwargs else query
        queries = [raw] if isinstance(raw, str) else (raw or [])
        if not queries:
            return "[Search] 'query' or 'queries' is required (string or array of strings)."
        return await self._run_blocking(lambda: _text_search_sync(queries, topk=top_k))


class UniversalImageSearchTool(DeepResearchTool):
    """Image search tool (text-to-image search for reference images)."""

    def __init__(self):
        super().__init__(
            name="image_search",
            description="Text-to-image search. Given a text query, returns image results (title, url, local_path).",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Descriptive text query for image search."},
                    "top_k": {"type": "integer", "description": "Number of images to return (default 10)."},
                    "save_dir": {"type": "string", "description": "Directory to save images (default: saved_img)."},
                    "sample_id": {"type": "string", "description": "Optional sample id prefix for filenames."},
                },
                "required": ["query"],
            },
        )
        self._save_dir = (os.environ.get("IMAGE_SEARCH_SAVE_DIR") or "").strip() or "saved_img"
        self._max_retry = 100

    async def call(
        self,
        query: str = "",
        top_k: int = 10,
        save_dir: Optional[str] = None,
        sample_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        return await self._run_blocking(
            lambda: _image_search_sync(
                query=query,
                top_k=top_k,
                save_dir=save_dir or self._save_dir,
                sample_id=sample_id,
                max_retry=self._max_retry,
            )
        )


class JinaBrowseTool(DeepResearchTool):
    """Web browsing tool (read-proxy + LLM summarization)."""

    def __init__(self):
        super().__init__(
            name="visit",
            description="Visit webpage(s) and return content/summary. Provide url and goal (query for the page).",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "array", "items": {"type": "string"}, "description": "URL(s) to visit."},
                    "goal": {"type": "string", "description": "What information to extract from the page(s)."},
                },
                "required": ["url", "goal"],
            },
        )
        self._read_engine = "jina"
        self._generate_engine = os.environ.get("BROWSE_GENERATE_ENGINE", "deepseekchat")
        self._max_retry = 10

    async def call(
        self,
        url: Union[str, List[str]] = None,
        goal: str = "",
        query: str = "",
        **kwargs,
    ) -> str:
        urls = [url] if isinstance(url, str) else (url or [])
        if not urls:
            return "[Visit] 'url' is required."
        # Gen-image agent passes "query"; other callers may use "goal"
        effective_goal = (goal or query or kwargs.get("query") or "").strip()
        goal = effective_goal or "Detailed summary of the page."
        results: List[str] = []
        for u in urls[:5]:
            r = await self._run_blocking(
                lambda uu=u: _browse_sync(
                    url=uu,
                    query=goal,
                    read_engine=self._read_engine,
                    generate_engine=self._generate_engine,
                    max_retry=self._max_retry,
                )
            )
            results.append(r)
        return "\n\n=======\n\n".join(results)


__all__ = [
    "WebTextSearchTool",
    "UniversalImageSearchTool",
    "JinaBrowseTool",
    "DeepResearchTool",
]

