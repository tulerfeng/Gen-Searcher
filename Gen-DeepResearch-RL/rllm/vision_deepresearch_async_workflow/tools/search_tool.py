import asyncio
import json
import os

from vision_deepresearch_async_workflow.tools.shared import (
    DeepResearchTool,
    get_cache_async,
    get_cache_key,
    log_search,
    log_tool_event,
    run_with_retries_async,
    set_cache_async,
)


class SearchTool(DeepResearchTool):
    """Web search tool using Zhipu or Serp API."""

    MAX_URLS = 10

    def __init__(self):
        super().__init__(
            name="search",
            description="Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of query strings. Include multiple complementary search queries in a single call.",
                    },
                },
                "required": ["query"],
            },
        )
        self.zhipu_api_key = os.getenv("ZHIPU_API_KEY")
        self.serp_api_key = os.getenv("SERP_API_KEY")
        self.zhipu_search_url = os.getenv(
            "TEXT_SEARCH_URL", "https://search-svip.bigmodel.cn/api/paas/v4/search"
        )
        self.serp_search_url = os.getenv(
            "TEXT_SEARCH_URL", "https://google.serper.dev/search"
        )

    def contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    async def _zhipu_search(self, query: str | list) -> str:
        """Use Zhipu web_search API when key is available."""
        try:
            import requests
        except ImportError:
            return """[Search - Dependencies Required]

Please install requests: pip install requests"""

        queries = [query] if isinstance(query, str) else query
        proxies = self._get_requests_proxies()

        async def search_single_query(q: str) -> str:
            # Check cache for individual query
            cache_key = get_cache_key(q)
            cached_result = await get_cache_async(
                "text_search", cache_key, executor=self.executor
            )
            if cached_result:
                return cached_result

            # Build request
            headers = {
                # Zhipu PaaS expects raw token in Authorization; keep value as-is
                "Authorization": self.zhipu_api_key,
                "Content-Type": "application/json",
            }
            location = "us"
            body = {
                "q": q,
                "search_engine": "search_prime",
                "location": location,
                "query_rewrite": False,
                "content_size": "high",
            }

            def send_request():
                return requests.post(
                    self.zhipu_search_url,
                    headers=headers,
                    data=json.dumps(body, ensure_ascii=False),
                    timeout=300,
                    proxies=proxies,
                )

            try:
                resp = await run_with_retries_async(
                    send_request, executor=self.executor
                )
            except Exception as exc:  # noqa: BLE001
                error_message = f"Search request failed for '{q}': {exc}"
                log_search("Zhipu", "Exception", q, error=error_message)
                return error_message

            text = resp.text
            try:
                data_obj = resp.json()
            except Exception:
                data_obj = None

            if resp.status_code != 200:
                error_message = f"HTTP {resp.status_code}: {text}"
                log_search("Zhipu", "HTTPError", q, error=error_message)
                return f"Search returned HTTP {resp.status_code} for '{q}'\n{text}"

            items = []
            if isinstance(data_obj, dict):
                items = data_obj.get("search_result") or data_obj.get("data") or []

            web_snippets: list[str] = []
            for idx, item in enumerate(items[: self.MAX_URLS], 1):
                title = (
                    item.get("title", "Untitled")
                    if isinstance(item, dict)
                    else "Untitled"
                )
                url = item.get("url", "") if isinstance(item, dict) else ""
                snippet = item.get("description", "") if isinstance(item, dict) else ""
                date = item.get("date") if isinstance(item, dict) else None

                snippet = (snippet or "").strip()

                entry = f"{idx}. [{title}]({url})"
                if date:
                    entry += f"\n   Date published: {date}"
                if snippet:
                    entry += f"\n   {snippet}"
                web_snippets.append(entry)

            content = (
                f"Search for '{q}' returned {len(web_snippets)} results:\n\n"
                + "\n\n".join(web_snippets)
                if web_snippets
                else f"No search results found for '{q}'"
            )
            # Store individual query result in cache (we've already passed error checks above)
            if not web_snippets:
                await set_cache_async(
                    "text_search", cache_key, q, content, executor=self.executor
                )

            return content

        tasks = [search_single_query(q) for q in queries]
        all_results: list[str] = await asyncio.gather(*tasks) if tasks else []

        final_result = (
            "\n=======\n".join(all_results)
            if len(all_results) > 1
            else (all_results[0] if all_results else "")
        )

        return final_result

    async def _serp_search(self, query: str | list) -> str:
        """Use Serp web search API when key is available."""
        try:
            import requests
        except ImportError:
            return """[Search - Dependencies Required]

Please install requests: pip install requests"""

        queries = [query] if isinstance(query, str) else query
        proxies = self._get_requests_proxies()

        async def search_single_query(q: str) -> str:
            cache_key = get_cache_key(q)
            cached_result = await get_cache_async(
                "text_search", cache_key, executor=self.executor
            )
            if cached_result:
                return cached_result

            payload = {
                "q": q,
                "hl": "en",
                "gl": "us",
            }

            headers = {
                "X-API-KEY": self.serp_api_key,
                "Content-Type": "application/json",
            }

            def send_request():
                return requests.post(
                    self.serp_search_url,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False),
                    timeout=300,
                    proxies=proxies,
                )

            try:
                resp = await run_with_retries_async(
                    send_request, executor=self.executor
                )
            except Exception as exc:  # noqa: BLE001
                error_message = f"Search request failed for '{q}': {exc}"
                log_search("Serp", "Exception", q, error=error_message)
                return error_message

            text = resp.text
            try:
                data_obj = resp.json()
            except Exception:
                data_obj = None

            if resp.status_code != 200:
                error_message = f"HTTP {resp.status_code}: {text}"
                log_search("Serp", "HTTPError", q, error=error_message)
                return f"Search returned HTTP {resp.status_code} for '{q}'\n{text}"

            items = []
            if isinstance(data_obj, dict):
                items = data_obj.get("organic") or []

            web_snippets: list[str] = []
            for idx, item in enumerate(items[: self.MAX_URLS], 1):
                title = (
                    item.get("title", "Untitled")
                    if isinstance(item, dict)
                    else "Untitled"
                )
                url = item.get("link", "") if isinstance(item, dict) else ""
                snippet = item.get("snippet", "") if isinstance(item, dict) else ""
                date = item.get("date") if isinstance(item, dict) else None

                snippet = (snippet or "").strip()

                entry = f"{idx}. [{title}]({url})"
                if date:
                    entry += f"\n   Date published: {date}"
                if snippet:
                    entry += f"\n   {snippet}"
                web_snippets.append(entry)

            content = (
                f"Search for '{q}' returned {len(web_snippets)} results:\n\n"
                + "\n\n".join(web_snippets)
                if web_snippets
                else f"No search results found for '{q}'"
            )
            if not web_snippets:
                await set_cache_async(
                    "text_search", cache_key, q, content, executor=self.executor
                )

            return content

        tasks = [search_single_query(q) for q in queries]
        all_results: list[str] = await asyncio.gather(*tasks) if tasks else []

        final_result = (
            "\n=======\n".join(all_results)
            if len(all_results) > 1
            else (all_results[0] if all_results else "")
        )

        return final_result

    async def call(self, query: str | list, **kwargs) -> str:
        """
        Search the web using Zhipu or Serp API.

        Args:
            query: Search query string or list of queries

        Returns:
            Formatted search results
        """
        # Prefer Zhipu if key available
        if self.zhipu_api_key:
            return await self._zhipu_search(query)

        if not self.serp_api_key:
            message = f"""[Search - API Key Required]

To enable real web search, configure one of these options:

Option 1 - Zhipu:
1. Add to .env: ZHIPU_API_KEY=your_key_here

Option 2 - Serp:
1. Get an API key from https://serper.dev
2. Add to .env: SERP_API_KEY=your_key_here

Placeholder results for '{query}'..."""

            log_tool_event("Search", "APIKeyMissing", f"query={query}", level="ERROR")
            log_search("Serp", "Config", str(query), error=message)
            return message

        return await self._serp_search(query)
