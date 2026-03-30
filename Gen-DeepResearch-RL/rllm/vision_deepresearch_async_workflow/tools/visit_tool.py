import asyncio
import json
import os
import random
import re
from typing import Any

from vision_deepresearch_async_workflow.tools.shared import (
    DeepResearchTool,
    get_cache_async,
    get_cache_key,
    log_tool_event,
    run_with_retries_async,
    set_cache_async,
    shorten_for_log,
)


class VisitTool(DeepResearchTool):
    """Web page visiting with content extraction."""

    MAX_URLS = 5
    MAX_CONTENT_CHARS = 120000

    EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Requirements**
- Return a valid JSON object only (no code fences, Markdown, comments, or additional text).
- The JSON must contain exactly the keys "rational", "evidence", and "summary".
- Each key must map to a string value. Use an empty string if no content is available.
- Do not include any extra keys or explanatory sentences outside the JSON object.

Example:
{{"rational": "Explain why the information is relevant to the goal.", "evidence": "Quote or paraphrase the key supporting content from the webpage.", "summary": "Provide a concise summary that connects the evidence back to the goal."}}
"""

    def __init__(self):
        super().__init__(
            name="visit",
            description="Visit webpage(s) and return the summary of the content.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": ["string", "array"],
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs.",
                    },
                    "goal": {
                        "type": "string",
                        "description": "The goal of the visit for webpage(s).",
                    },
                },
                "required": ["url", "goal"],
            },
        )
        self.zhipu_api_key = os.getenv("ZHIPU_API_KEY")
        self.jina_api_key = os.getenv("JINA_API_KEY")
        self.zhipu_reader_url = os.getenv(
            "READER_URL", "https://search-svip.bigmodel.cn/api/paas/v4/reader"
        )
        self.jina_reader_url = os.getenv("READER_URL", "https://r.jina.ai")
        self.extract_model = os.getenv("EXTRACT_MODEL", "Qwen3-VL-30B-A3B-Instruct")
        self.extract_max_tokens = 16384
        raw_extract_urls = os.getenv("EXTRACT_URL", "")
        self.extract_urls = [
            item.strip() for item in raw_extract_urls.split(",") if item.strip()
        ]

    async def call(self, url: str | list, goal: str = "", **kwargs) -> str:
        """Visit webpages via Reader API and optionally summarize with a local model."""

        urls = [url] if isinstance(url, str) else url
        if not urls:
            return "[Visit] No valid URL provided"

        tasks = [
            self._handle_single_url(target_url, goal)
            for target_url in urls[: self.MAX_URLS]
        ]
        results = await asyncio.gather(*tasks) if tasks else []

        return "\n\n=======\n\n".join(results)

    async def _handle_single_url(self, url: str, goal: str) -> str:
        normalized_url = self._normalize_url(url)

        try:
            reader_payload = await self._fetch_reader_content(normalized_url)
        except Exception as exc:  # noqa: BLE001
            log_tool_event(
                source="Visit/Reader",
                status="Exception",
                message=f"url={normalized_url}",
                error=str(exc),
                level="ERROR",
            )
            return self._build_failure_message(
                normalized_url, goal, f"Unable to fetch webpage content: {exc}"
            )

        if reader_payload is None:
            return self._build_failure_message(
                normalized_url, goal, "Reader API returned empty payload"
            )

        content = reader_payload.get("content") or ""
        description = reader_payload.get("description") or ""

        if not content:
            fallback = description or "Webpage content is empty"
            return self._build_failure_message(normalized_url, goal, fallback)

        content = self._truncate_content(content)

        summary_result = await self._summarize_with_extract(
            content, goal, reader_payload
        )

        if summary_result is None:
            log_tool_event(
                "Visit", "ExtractSummaryFailed", f"url={normalized_url}", level="ERROR"
            )
            evidence_text = content
            summary_text = (
                description or "Summary service unavailable. Returning raw content."
            )
        else:
            evidence_text = summary_result.get("evidence") or content
            summary_text = summary_result.get("summary") or description or ""

        return self._format_success(normalized_url, goal, evidence_text, summary_text)

    def _normalize_url(self, url: str) -> str:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if not parsed.scheme:
            return f"https://{url}"
        return url

    def _select_extract_url(self) -> str | None:
        if not self.extract_urls:
            return None
        selected = random.choice(self.extract_urls)
        if not re.search(r"/v1/chat/completions/?$", selected):
            selected = f"{selected.rstrip('/')}/v1/chat/completions"
        return selected

    async def _fetch_reader_content(self, url: str) -> dict[str, Any] | None:
        # Check cache first
        cache_key = get_cache_key(url)
        cached_result = await get_cache_async(
            "text_visit", cache_key, executor=self.executor
        )
        if cached_result:
            try:
                return json.loads(cached_result)
            except json.JSONDecodeError:
                pass  # Continue with API call if cache is corrupted

        try:
            import requests
        except ImportError as exc:  # noqa: PERF203
            raise RuntimeError("Visit tool requires 'requests' package") from exc

        proxies = self._get_requests_proxies()

        if self.zhipu_api_key:
            headers = {
                "Content-Type": "application/json",
            }
            if self.zhipu_api_key:
                headers["Authorization"] = self.zhipu_api_key

            # Support optional headers consistent with the demo scripts
            optional_headers = {
                "X-Return-Format": "markdown",
                "X-No-Cache": "false",
                "X-Timeout": "60",
                "X-Retain-Images": "false",
                "X-With-Images-Summary": "false",
                "X-With-Links-Summary": "false",
            }
            headers.update({k: v for k, v in optional_headers.items() if v is not None})

            body = {
                "url": url,
            }

            def send_request():
                return requests.post(
                    self.zhipu_reader_url,
                    headers=headers,
                    data=json.dumps(body, ensure_ascii=False),
                    timeout=60,
                    proxies=proxies,
                )

            response = await run_with_retries_async(
                send_request, executor=self.executor
            )

            if response.status_code != 200:
                raise RuntimeError(f"Reader API returned HTTP {response.status_code}")

            try:
                payload = response.json()
            except json.JSONDecodeError as exc:  # noqa: PERF203
                raise RuntimeError("Reader API returned non-JSON payload") from exc

            if not isinstance(payload, dict):
                raise RuntimeError("Reader API payload structure is invalid")

            if payload.get("code") != 200:
                raise RuntimeError(
                    f"Reader API returned error code: {payload.get('code')}"
                )

            data = payload.get("data")
            if not isinstance(data, dict):
                raise RuntimeError("Reader API data field missing or malformed")

            result = {
                "content": data.get("content") or "",
                "description": data.get("description") or "",
                "meta": data,
            }
        else:
            headers = {
                "Authorization": self.jina_api_key,
            }
            body = {
                "url": url,
            }

            def send_request():
                return requests.post(
                    self.jina_reader_url,
                    headers=headers,
                    data=body,
                    timeout=60,
                    proxies=proxies,
                )

            response = await run_with_retries_async(
                send_request, executor=self.executor
            )

            if response.status_code != 200:
                raise RuntimeError(f"Reader API returned HTTP {response.status_code}")

            result = {
                "content": response.text or "",
                "description": "",
                "meta": {
                    "provider": "jina",
                    "url": url,
                    "reader_url": self.jina_reader_url,
                },
            }

        # Store result in cache only if we have valid content
        if result["content"].strip():
            await set_cache_async(
                "text_visit",
                cache_key,
                url,
                json.dumps(result, ensure_ascii=False),
                executor=self.executor,
            )

        return result

    def _truncate_content(self, content: str) -> str:
        if len(content) <= self.MAX_CONTENT_CHARS:
            return content
        return content[: self.MAX_CONTENT_CHARS] + "\n[Content truncated...]"

    async def _summarize_with_extract(
        self, content: str, goal: str, reader_payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        extract_url = self._select_extract_url()
        if not extract_url:
            log_tool_event(
                source="Visit/Extract",
                status="Config",
                message="EXTRACT_URL is not set, skip extract service",
            )
            return None

        try:
            import requests
        except ImportError:
            log_tool_event(
                source="Visit/Extract",
                status="DependencyMissing",
                message="'requests' package not installed, cannot call extract service",
                level="WARNING",
            )
            return None

        prompt = self.EXTRACTOR_PROMPT.format(
            webpage_content=content, goal=goal or "N/A"
        )

        extract_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        if self.extract_model:
            payload = {
                "model": self.extract_model,
                "messages": extract_messages,
                "max_tokens": self.extract_max_tokens,
            }

        headers = {"Content-Type": "application/json"}
        proxies = self._get_requests_proxies()

        try:
            response = await run_with_retries_async(
                lambda: requests.post(
                    url=extract_url,
                    headers=headers,
                    json=payload,
                    timeout=60,
                    proxies=proxies,
                ),
                executor=self.executor,
            )
        except Exception as exc:  # noqa: BLE001
            log_tool_event(
                source="Visit/Extract",
                status="RequestError",
                message=f"url={extract_url}",
                error=str(exc),
                level="ERROR",
            )
            return None

        if response.status_code != 200:
            log_tool_event(
                source="Visit/Extract",
                status="HTTPError",
                message=f"url={extract_url} status={response.status_code}",
                level="WARNING",
            )
            return None

        try:
            result = response.json()
        except json.JSONDecodeError:
            log_tool_event(
                source="Visit/Extract",
                status="ParseError",
                message="Extract service returned non-JSON response, unable to parse",
                level="WARNING",
            )
            return None

        raw_payload: str | dict | None = None
        content_source: str | None = None

        if isinstance(result, dict):
            choices = result.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0] or {}
                if isinstance(first_choice, dict):
                    message_dict = first_choice.get("message")
                    if isinstance(message_dict, dict):
                        message_content = message_dict.get("content")
                        if isinstance(message_content, str) and message_content.strip():
                            raw_payload = message_content
                            content_source = "choices[0].message.content"
                    if raw_payload is None:
                        text_candidate = first_choice.get("text")
                        if isinstance(text_candidate, str) and text_candidate.strip():
                            raw_payload = text_candidate
                            content_source = "choices[0].text"
            if raw_payload is None:
                fallback_payload = result.get("content") or result.get("data")
                if isinstance(fallback_payload, (str, dict)):
                    raw_payload = fallback_payload
                    content_source = "response.content/data"

        if raw_payload is None:
            log_tool_event(
                source="Visit/Extract",
                status="InvalidContent",
                message="Extract service response missing usable content",
                level="WARNING",
            )
            return None

        fallback_used = False
        content_dict: dict | None = None

        if isinstance(raw_payload, dict):
            content_dict = raw_payload
        elif isinstance(raw_payload, str):
            candidate = raw_payload.strip()
            if candidate.startswith("`"):
                candidate = candidate.strip("`")
            try:
                content_dict = json.loads(candidate)
                content_source = f"{content_source or 'string_payload'} -> json.loads"
            except json.JSONDecodeError:
                fallback_used = True
                summary_text = candidate
                content_dict = {
                    "rational": "",
                    "evidence": summary_text,
                    "summary": summary_text,
                }
        if not isinstance(content_dict, dict):
            log_tool_event(
                source="Visit/Extract",
                status="InvalidContent",
                message="Extract service response does not contain JSON summary content",
                level="WARNING",
            )
            return None

        return content_dict

    def _build_failure_message(self, url: str, goal: str, reason: str) -> str:
        useful_information = f"The useful information in {url} for user goal {goal or 'N/A'} as follows: \n\n"
        useful_information += "Evidence in page: \n" + reason + "\n\n"
        useful_information += (
            "Summary: \n"
            + "Unable to retrieve webpage content. Please check the link or try again later."
            + "\n\n"
        )

        reason_preview = shorten_for_log(reason)
        result_preview = shorten_for_log(useful_information)
        log_tool_event(
            source="Visit",
            status="Failure",
            message=(
                f"url={url} "
                f"reason_len={len(reason)} "
                f"result_len={len(useful_information)} "
                f"reason_preview={json.dumps(reason_preview, ensure_ascii=False)} "
                f"result_preview={json.dumps(result_preview, ensure_ascii=False)}"
            ),
            level="WARNING",
        )

        return useful_information

    def _format_success(self, url: str, goal: str, evidence: str, summary: str) -> str:
        useful_information = f"The useful information in {url} for user goal {goal or 'N/A'} as follows: \n\n"
        useful_information += "Evidence in page: \n" + evidence + "\n\n"
        useful_information += (
            "Summary: \n" + (summary or "No summary generated") + "\n\n"
        )

        evidence_text = evidence or ""
        summary_text = summary or ""
        evidence_preview = shorten_for_log(evidence_text)
        summary_preview = shorten_for_log(summary_text)
        return useful_information
