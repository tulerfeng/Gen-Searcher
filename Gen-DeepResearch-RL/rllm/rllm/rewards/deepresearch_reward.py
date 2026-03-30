import asyncio
import json
import os
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx

from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardOutput


def _shorten_for_log(text: Any, limit: int = 200) -> str:
    """Create a concise preview string for debug prints."""

    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = json.dumps(text, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            text = str(text)

    text = text.replace("\n", "\\n")
    if len(text) <= limit * 2:
        return text
    return f"{text[:limit]} ... {text[-limit:]}"


def _print_debug(tag: str, **fields: Any) -> None:
    """Standardized stdout debugging for reward function."""

    parts = [f"[Reward][DeepResearch][{tag}]"]
    for key, value in fields.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    print(" ".join(parts))


T = TypeVar("T")


def _get_requests_proxies() -> dict[str, str | None] | None:
    """Build requests-compatible proxy mapping from TOOL_HTTPS_PROXY."""

    proxy_value = os.getenv("TOOL_HTTPS_PROXY")
    if proxy_value is None:
        return None

    proxy_value = proxy_value.strip()
    if not proxy_value or proxy_value.lower() == "none":
        return {"http": None, "https": None}

    return {"http": proxy_value, "https": proxy_value}


def _run_with_retries_sync(
    func: Callable[[], T], attempts: int = 3, delay: float = 1.0
) -> T:
    """Execute a callable with retry support (synchronous)."""

    last_error: Exception | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= attempts:
                break
            if delay > 0:
                time.sleep(delay)

    if last_error is not None:
        raise last_error

    raise RuntimeError("run_with_retries_sync executed without performing any attempts")


async def _async_run_with_retries(
    func: Callable[[], T], attempts: int = 3, delay: float = 1.0
) -> T:
    """Execute a callable with retry support using asyncio.to_thread."""

    last_error: Exception | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            return await asyncio.to_thread(func)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= attempts:
                break
            if delay > 0:
                await asyncio.sleep(delay)

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "async_run_with_retries executed without performing any attempts"
    )


def _run_async_or_sync(
    coro: Awaitable[T], fallback: Callable[[], T], *, fallback_tag: str
) -> tuple[T, str]:
    """
    Execute coroutine when no running event loop exists, otherwise use fallback.

    Returns a tuple of (result, execution_mode).
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro), "async"

    _print_debug(fallback_tag, reason="event_loop_running", action="use_sync_fallback")
    return fallback(), "sync_fallback"


class RewardDeepResearchFn:
    """
    A simple 1/0 reward for DeepResearch-style answers.

    - Extract final answer from the model response (prefers <answer>...</answer> if present)
    - Normalize and exact-match against ground truth (string or list of strings)
    - Return 1.0 if correct, else 0.0
    """

    def __init__(self, config: RewardConfig):
        self.config = config
        # judge configuration from env
        self.use_judge: bool = True  # default enable LLM judge
        self.base_url: str | None = os.getenv("JUDGE_BASE_URL")
        # vLLM local judge support
        self.vllm_url: str | None = os.getenv("JUDGE_URL")
        self.is_vllm: bool = bool(self.vllm_url)
        # prefer explicit JUDGE_API_KEY; fallback to OPENAI / TOGETHER
        self.api_key: str | None = (
            os.getenv("JUDGE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("TOGETHER_AI_API_KEY")
        )
        self.model: str = os.getenv(
            "JUDGE_MODEL", os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")
        )

        # Infer base_url if not provided
        if not self.is_vllm and self.base_url is None:
            if os.getenv("TOGETHER_AI_API_KEY") and self.api_key == os.getenv(
                "TOGETHER_AI_API_KEY"
            ):
                self.base_url = "https://api.together.xyz/v1"
                # provide a reasonable default together model id if not overridden
                if "gpt-" in self.model.lower():
                    self.model = os.getenv(
                        "JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct-Turbo"
                    )
            else:
                self.base_url = "https://api.openai.com/v1"

        # If using vLLM locally, prefer a Qwen instruct model when default is GPT-like
        if self.is_vllm and "gpt-" in self.model.lower():
            self.model = os.getenv("JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct-Turbo")

        # http client (sync)
        self._client: httpx.Client | None = None
        self._judge_proxies = _get_requests_proxies()
        self.judge_retry_attempts = int(os.getenv("JUDGE_RETRY_ATTEMPTS", "3"))
        self.judge_retry_delay = float(os.getenv("JUDGE_RETRY_DELAY", "1.0"))
        self.judge_timeout = float(os.getenv("JUDGE_TIMEOUT", "300"))
        if self.use_judge and (self.api_key or self.is_vllm):
            try:
                self._client = httpx.Client(timeout=60.0)
            except Exception:
                self._client = None

    def _strip_think(self, s: str) -> str:
        # Remove <think>...</think> blocks and collapse whitespace
        s = re.sub(r"<think>[\s\S]*?</think>", " ", s, flags=re.IGNORECASE)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def extract_final_answer(self, response: str) -> str:
        response = (response or "").strip()
        if not response:
            return ""

        # Remove <think> ... </think>
        response = self._strip_think(response)

        # Prefer <answer>...</answer>
        m = re.search(r"<answer>([\s\S]*?)</answer>", response, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # Fallback: try boxed LaTeX like \boxed{...}
        def unbox(s: str) -> str | None:
            try:
                i = s.find("boxed{")
                if i == -1:
                    return None
                i += 6
                depth = 1
                j = i
                while depth and j < len(s):
                    depth += (s[j] == "{") - (s[j] == "}")
                    j += 1
                if depth:
                    return None
                return s[i : j - 1]
            except Exception:
                return None

        boxed = unbox(response)
        if boxed is not None:
            return boxed.strip()

        return response

    def normalize(self, s: str) -> str:
        s = s or ""
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        # remove simple surrounding punctuation
        s = re.sub(r"^[^\w\-]+|[^\w\-]+$", "", s)
        return s

    async def async_call(self, input: RewardInput) -> RewardOutput:
        """Async wrapper to avoid blocking event loop."""
        return await asyncio.to_thread(self.__call__, input)

    def __call__(self, input: RewardInput) -> RewardOutput:
        model_response = input.action
        task_info: dict[str, Any] = input.task_info or {}
        answer = task_info.get("answer")
        question = str(task_info.get("question", ""))

        _print_debug(
            "Input",
            question_len=len(question),
            question_preview=_shorten_for_log(question),
            answer_type=type(answer).__name__,
            answer_len=len(str(answer)) if answer is not None else 0,
            answer_preview=_shorten_for_log(answer),
            response_len=len(model_response or ""),
            response_preview=_shorten_for_log(model_response),
        )

        if answer is None:
            _print_debug("InputError", reason="No answer provided")
            return RewardOutput(
                reward=self.config.unk_error_reward,
                is_correct=False,
                metadata={"error": "No answer provided"},
            )

        extracted = self.extract_final_answer(model_response)
        # _print_debug(
        #     "Extracted",
        #     extracted_len=len(extracted),
        #     extracted_preview=_shorten_for_log(extracted),
        # )

        # Support list of references
        if isinstance(answer, (list, tuple)):
            refs = [self.normalize(str(g)) for g in answer]
        else:
            refs = [self.normalize(str(answer))]

        pred = self.normalize(extracted)
        exact_match = pred in refs
        # _print_debug(
        #     "Normalized",
        #     prediction=pred,
        #     refs_count=len(refs),
        #     refs_preview=_shorten_for_log(refs),
        #     exact_match=exact_match,
        # )

        # Try LLM judge first; if unable to obtain explicit yes/no after retries, default to no
        is_correct = False
        judgment_text: str | None = None
        judge_decided: bool = False
        judge_attempts: int = 0
        fallback_reason: str | None = None
        judge_execution_mode: str | None = None

        # Build judging prompt once for both vLLM and OpenAI/Together
        judging_prompt = (
            "You are an impartial judge evaluating whether a deep research report contains the correct answer.\n\n"
            "[Question]\n{question}\n\n"
            "[Correct Answer]\n{reference_answer}\n\n"
            "[Deep Research Report]\n{assistant_answer}\n\n"
            "Task: Determine if the deep research report contains the correct answer anywhere in its content.\n\n"
            "Instructions:\n"
            "1. Read through the entire research report carefully\n"
            "2. Look for the correct answer anywhere in the report (it may be embedded in paragraphs, tables, or sections)\n"
            "3. Check if the information in the report is consistent with the correct answer\n"
            '4. The answer does NOT need to be in a specific format or labeled as "final answer"\n'
            "5. Provide your reasoning\n"
            '6. Answer with "yes" if the report contains the correct answer, "no" if it doesn\'t or contradicts it\n\n'
            "Output format:\n"
            "correct: [yes/no]\n"
            "reasoning: [your explanation]"
        ).format(
            question=str(task_info.get("question", "")),
            reference_answer=str(answer),
            assistant_answer=str(extracted),
        )

        # _print_debug(
        #     "JudgeConfig",
        #     use_judge=self.use_judge,
        #     is_vllm=self.is_vllm,
        #     has_client=bool(self._client),
        #     model=self.model,
        #     base_url=self.base_url or "N/A",
        #     vllm_url=self.vllm_url or "N/A",
        #     prompt_len=len(judging_prompt),
        # )

        if self.use_judge and self._client is not None:
            if self.is_vllm and self.vllm_url:
                try:
                    import requests
                except ImportError as exc:  # noqa: PERF203
                    judgment_text = f"judge_error: requests_missing ({exc})"
                    _print_debug(
                        "Judge/vLLM", status="DependencyMissing", error=str(exc)
                    )
                else:
                    url = self.vllm_url
                    # accept either full endpoint or base URL
                    if not re.search(r"/v1/chat/completions/?$", url):
                        url = f"{url.rstrip('/')}/v1/chat/completions"

                    judging_messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": judging_prompt},
                    ]
                    payload = {
                        "model": self.model,
                        "messages": judging_messages,
                        # "temperature": 0.1,
                        "max_tokens": 800,
                    }
                    headers = {"Content-Type": "application/json"}
                    attempt_counter: dict[str, int] = {"value": 0}
                    proxies_state = "enabled" if self._judge_proxies else "disabled"

                    def send_request() -> requests.Response:
                        attempt_counter["value"] += 1
                        attempt_idx = attempt_counter["value"]
                        _print_debug(
                            "Judge/vLLM/Request",
                            attempt=attempt_idx,
                            url=url,
                            payload_preview=_shorten_for_log(payload),
                            proxies=proxies_state,
                        )
                        try:
                            return requests.post(
                                url=url,
                                headers=headers,
                                json=payload,
                                timeout=self.judge_timeout,
                                proxies=self._judge_proxies,
                            )
                        except Exception as exc:  # noqa: BLE001
                            _print_debug(
                                "Judge/vLLM/Error",
                                attempt=attempt_idx,
                                error=str(exc),
                            )
                            raise

                    try:
                        async_coro = _async_run_with_retries(
                            send_request,
                            attempts=self.judge_retry_attempts,
                            delay=self.judge_retry_delay,
                        )
                        resp, judge_execution_mode = _run_async_or_sync(
                            async_coro,
                            lambda: _run_with_retries_sync(
                                send_request,
                                attempts=self.judge_retry_attempts,
                                delay=self.judge_retry_delay,
                            ),
                            fallback_tag="Judge/vLLM/AsyncFallback",
                        )
                    except Exception as exc:  # noqa: BLE001
                        judgment_text = f"judge_error: {exc}"
                        judge_attempts = attempt_counter["value"]
                        _print_debug(
                            "Judge/vLLM/Failure",
                            attempts=attempt_counter["value"],
                            error=str(exc),
                        )
                    else:
                        judge_attempts = attempt_counter["value"]
                        response_text = resp.text
                        # _print_debug(
                        #     "Judge/vLLM/Response",
                        #     attempt=judge_attempts,
                        #     status=resp.status_code,
                        #     mode=judge_execution_mode,
                        #     proxies=proxies_state,
                        #     body_preview=_shorten_for_log(response_text),
                        # )
                        try:
                            resp.raise_for_status()
                            data = resp.json()
                        except Exception as exc:  # noqa: BLE001
                            judgment_text = f"judge_error: {exc}"
                            _print_debug(
                                "Judge/vLLM/Error",
                                attempt=judge_attempts,
                                error=str(exc),
                            )
                        else:
                            choices = data.get("choices") or [{}]
                            first_choice = choices[0] or {}
                            content_text = ""
                            content_source = None
                            finish_reason = None
                            if isinstance(first_choice, dict):
                                message_dict = first_choice.get("message")
                                if isinstance(message_dict, dict):
                                    message_content = message_dict.get("content", "")
                                    if (
                                        isinstance(message_content, str)
                                        and message_content.strip()
                                    ):
                                        content_text = message_content
                                        content_source = "choices[0].message.content"
                                if not content_text:
                                    text_candidate = first_choice.get("text", "")
                                    if (
                                        isinstance(text_candidate, str)
                                        and text_candidate.strip()
                                    ):
                                        content_text = text_candidate
                                        content_source = "choices[0].text"
                                finish_reason = (
                                    first_choice.get("finish_reason")
                                    if finish_reason is None
                                    else finish_reason
                                )
                            if not content_text and isinstance(data, dict):
                                fallback_content = data.get("content")
                                if (
                                    isinstance(fallback_content, str)
                                    and fallback_content.strip()
                                ):
                                    content_text = fallback_content
                                    content_source = "response.content"
                            usage_info = data.get("usage")
                            judgment_text = str(content_text or "")
                            _print_debug(
                                "Judge/vLLM/Parsed",
                                attempt=judge_attempts,
                                finish_reason=finish_reason or "N/A",
                                usage=_shorten_for_log(usage_info),
                                content_source=content_source or "N/A",
                                content_preview=_shorten_for_log(judgment_text),
                            )
                            if not judgment_text.strip():
                                _print_debug(
                                    "Judge/vLLM/EmptyContent",
                                    attempt=judge_attempts,
                                    finish_reason=finish_reason or "N/A",
                                    usage=_shorten_for_log(usage_info),
                                    content_source=content_source or "N/A",
                                    body_preview=_shorten_for_log(response_text),
                                )
                            lt = judgment_text.lower()

                            parsed: bool | None = None
                            if "correct:" in lt:
                                try:
                                    line = [
                                        line
                                        for line in lt.split("\n")
                                        if "correct:" in line
                                    ][0]
                                    parsed = "yes" in line
                                except Exception:
                                    parsed = None
                            else:
                                stripped = lt.strip()
                                if stripped.startswith("correct: yes"):
                                    parsed = True
                                elif stripped.startswith("correct: no"):
                                    parsed = False

                            if parsed is not None:
                                is_correct = bool(parsed)
                                judge_decided = True
                                _print_debug(
                                    "Judge/vLLM/Decision",
                                    attempt=judge_attempts,
                                    parsed=is_correct,
                                )
            elif self.api_key:
                try:
                    for attempt in range(3):
                        judge_attempts = attempt + 1
                        payload = {
                            "model": self.model,
                            "messages": [{"role": "user", "content": judging_prompt}],
                            "temperature": 0.1,
                            "max_tokens": 800,
                        }
                        _print_debug(
                            "Judge/OpenAI/Request",
                            attempt=judge_attempts,
                            url=f"{self.base_url.rstrip('/')}/chat/completions",
                            payload_preview=_shorten_for_log(payload),
                        )
                        resp = self._client.post(
                            url=f"{self.base_url.rstrip('/')}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json",
                            },
                            json=payload,
                        )
                        response_text = resp.text
                        resp.raise_for_status()
                        data = resp.json()
                        choices = data.get("choices") or [{}]
                        first_choice = choices[0] or {}
                        message_dict = (
                            first_choice.get("message", {})
                            if isinstance(first_choice, dict)
                            else {}
                        )
                        content = ""
                        finish_reason = None
                        if isinstance(message_dict, dict):
                            content = message_dict.get("content", "") or ""
                        if isinstance(first_choice, dict):
                            finish_reason = first_choice.get("finish_reason")
                        usage_info = data.get("usage")
                        judgment_text = str(content or "")
                        _print_debug(
                            "Judge/OpenAI/Response",
                            attempt=judge_attempts,
                            status=resp.status_code,
                            finish_reason=finish_reason or "N/A",
                            usage=_shorten_for_log(usage_info),
                            content_preview=_shorten_for_log(judgment_text),
                            body_preview=_shorten_for_log(response_text),
                        )
                        if not judgment_text.strip():
                            _print_debug(
                                "Judge/OpenAI/EmptyContent",
                                attempt=judge_attempts,
                                finish_reason=finish_reason or "N/A",
                                usage=_shorten_for_log(usage_info),
                                body_preview=_shorten_for_log(response_text),
                            )
                        lt = judgment_text.lower()

                        parsed: bool | None = None
                        if "correct:" in lt:
                            try:
                                line = [
                                    line
                                    for line in lt.split("\n")
                                    if "correct:" in line
                                ][0]
                                parsed = "yes" in line
                            except Exception:
                                parsed = None
                        else:
                            stripped = lt.strip()
                            if stripped.startswith("correct: yes"):
                                parsed = True
                            elif stripped.startswith("correct: no"):
                                parsed = False

                        if parsed is not None:
                            is_correct = bool(parsed)
                            judge_decided = True
                            _print_debug(
                                "Judge/OpenAI/Decision",
                                attempt=judge_attempts,
                                parsed=is_correct,
                            )
                            break
                except Exception as e:
                    judgment_text = f"judge_error: {e}"
                    _print_debug(
                        "Judge/OpenAI/Error", attempt=judge_attempts, error=str(e)
                    )
                    # fall back below
        else:
            _print_debug(
                "JudgeSkipped",
                use_judge=self.use_judge,
                has_client=bool(self._client),
            )

        if not judge_decided:
            fallback_reason = "judge_undecided_use_exact_match"
            is_correct = bool(exact_match)
            _print_debug(
                "JudgeFallback",
                reason=fallback_reason,
                exact_match=exact_match,
            )
        elif not is_correct:
            # judge decided "no" explicitly; keep as is
            pass
        else:
            # judge decided "yes"
            pass

        reward = (
            self.config.correct_reward if is_correct else self.config.incorrect_reward
        )
        _print_debug(
            "RewardResult",
            reward=reward,
            is_correct=is_correct,
            judge_decided=judge_decided,
            judge_attempts=judge_attempts,
            judge_mode=judge_execution_mode or "N/A",
            proxy_state="enabled" if self._judge_proxies else "disabled",
            exact_match=exact_match,
            fallback=fallback_reason or "None",
            judgment_preview=_shorten_for_log(judgment_text),
        )
        metadata = {
            "extracted_answer": extracted,
            "normalized_prediction": pred,
            "normalized_refs": refs,
            "judge_used": bool(judgment_text is not None),
            "judge_decided": judge_decided,
            "judge_attempts": judge_attempts,
            "judgment": judgment_text,
            "judge_proxy_enabled": bool(self._judge_proxies),
        }
        metadata["exact_match"] = exact_match
        if judge_execution_mode is not None:
            metadata["judge_execution_mode"] = judge_execution_mode
        if fallback_reason is not None:
            metadata["fallback_reason"] = fallback_reason
        return RewardOutput(reward=reward, is_correct=is_correct, metadata=metadata)
