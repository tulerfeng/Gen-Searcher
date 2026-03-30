"""
Gen Image Agent: multi-round ReAct agent for image generation tasks.

Core features:
1) Manage IMG_### ids independently per trajectory
2) Use the system prompt from prompt_gen.py
3) Support image_search and automatically assign IMG_###
4) Output <answer> containing gen_prompt and reference_images
"""
from __future__ import annotations

import asyncio
import json
import json5
import os
import re
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from rllm.engine.rollout import RolloutEngine
from rllm.workflows.workflow import TerminationEvent

# Constants
OBS_START = "<tool_response>"
OBS_END = "\n</tool_response>"


def _get_max_llm_call_per_run() -> int:
    raw = os.environ.get("MAX_LLM_CALL_PER_RUN", "9")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 9


MAX_LLM_CALL_PER_RUN = _get_max_llm_call_per_run()
# Per round, from image_search results, how many images at most are attached as multimodal inputs to the model.
# (The tool may return more images, but extra ones are only described in text.)
MAX_IMAGES_PER_SEARCH_FOR_MODEL = 5

FINAL_MESSAGE = """
=== FINAL STEP: OUTPUT ANSWER NOW ===
This is your FINAL step. You have NO more chances.

CRITICAL RULES:
1. Tool calls are ABSOLUTELY FORBIDDEN. Any <tool_call> will be IGNORED.
2. You MUST output <answer>...</answer> immediately.
3. Even if information is incomplete or uncertain, you MUST generate an answer with what you have.
4. Use available IMG_### from previous searches. If none, describe the prompt without references.
5. Do NOT write <think>. Do NOT write <tool_call>. Do NOT explain or apologize.

FORMAT (output this EXACTLY):
<answer>
{
  "gen_prompt": "your detailed generation prompt here",
  "reference_images": [{"img_id": "IMG_###", "note": "..."}]
}
</answer>

Output your answer JSON NOW:
""".strip()

# System prompt from prompt_gen.py
SYSTEM_PROMPT_GEN_IMAGE = """You are a helpful assistant for grounding prompts for image generation.

Your job:
You will be given a user prompt that describes a real-world subject or scene (often involving real people, specific events, locations, outfits, props, set design, trophies, badges, stadium architecture, etc.).
Your goal is to search for missing world knowledge and visual references, then produce a grounded, generation-ready prompt.

Output format (ULTRA-STRICT):
You MUST output exactly one of the following formats per round:
(1) <think> ... </think>
    <tool_call> ... </tool_call>
OR
(2) <think> ... </think>
    <answer> ... </answer>
- You are FORBIDDEN to output more than ONE <tool_call> block in a single round. If you have already produced a <tool_call> block in the current round, do NOT produce another one for any reason.

Critical rule:
In EVERY round, you MUST write <think> ... </think> first, and then choose EXACTLY ONE of:
- a single <tool_call> ... </tool_call> (continue searching/verifying), OR
- <answer> ... </answer> (terminate the task; final output).
You MUST NOT output <tool_call> without a preceding <think>.
You MUST NOT output both <tool_call> and <answer> in the same round.

EXCEPTION - Final Step Override:
If you receive a message containing "FINAL STEP" or "Final Step Reached":
- IGNORE all other rules about <think> and <tool_call>
- Tool calls are ABSOLUTELY FORBIDDEN at this point
- You MUST immediately output ONLY <answer>...</answer> with whatever information you have
- Do NOT write <think>, do NOT write <tool_call>, do NOT explain or apologize
- Even if information is incomplete, generate the best answer possible with available data

EXCEPTION - Response Too Long:
If you receive a message containing "RESPONSE TOO LONG" or "TRUNCATED":
- Your previous response exceeded the length limit and was cut off
- Do NOT write <think>. Skip all reasoning text.
- Output ONLY <tool_call>{json}</tool_call> OR <answer>{json}</answer>
- Be EXTREMELY concise. No explanations, no extra text.
- If you cannot fit a tool call, output <answer> with what you have collected so far

Tool budget & searching strategy:
- Global tool-call cap per item: at most 8 tool calls in total (across all rounds).
- Use as few tool calls as possible. Do NOT "use up" the budget.
- You must call "image_search" tool at least once.
- Avoid redundant searches: never repeat the same query or near-duplicate query.
- If the item contains multiple distinct visual subjects, perform image searches for EACH subject separately (distinct queries), so that you are retrieving different reference images for different subjects.

Which tools to use:
- Prefer "image_search" when the prompt involves real people, specific scenes, exact outfits/props/venues, or anything visually grounded.
- Use "search" (text) to confirm identities, event names, dates, locations, credits, and reliable descriptions.
- Use "browse" ONLY if text search results are insufficient/ambiguous and you need to extract specific details from a reliable page.
  - Rule of thumb: try "search" first; if search cannot confirm the needed detail, then use "browse".


Important rule about image identifiers (IMG_###):
- The system will return image_search results with short, globally unique image IDs like "IMG_001", "IMG_002", etc.
- The image IDs may not start from 001.
- In your reasoning, you may refer to images ONLY by these IMG_### IDs.
- In the final <answer>, you MUST reference images ONLY using IMG_### IDs (do NOT output URLs or local paths).
- Never copy long image URLs/paths into your final answer. The caller will map IMG_### back to {url, local_path, title} automatically.


Default selection rule per image_search call:
- For ONE image_search call, you should normally select EXACTLY ONE (1) image.
- Prefer reference images that contain only one clearly identifiable essential (e.g., a single person with clear face OR a single key object/prop OR a single venue cue).
- Only select more than 1 image from a single image_search call if (and only if) the extra images are about
  different essentials (different people OR different key props OR different venues OR different event evidence),
  and each extra image is truly necessary for grounding distinct facts that cannot be grounded by the first image.
- Otherwise, choose just one best image. This is the expected behavior in most cases.

STRICT de-duplication rule:
- Images are considered duplicates if they share ANY ONE of the following:
  (A) same main person (same identity), OR
  (B) same main object/prop (same essential item), OR
  (C) same essential scene/event moment (same occasion), OR
  (D) same essential setting/venue (same place),
  EVEN IF the angle/crop/background differs.
- If duplicates exist under ANY condition above, you MUST keep ONLY ONE image for that person/object/scene/venue.
  Pick the single clearest, highest-resolution, most informative one.

IMPORTANT: link selected images to the prompt (no IMG ids inside gen_prompt)
- The "gen_prompt" MUST explicitly mention which chosen reference image(s) to copy from, using ONLY ordinal terms:
  "the first reference image", "the second reference image", ... (based on the order in "reference_images").
- Do NOT write "IMG_###" inside gen_prompt.
- If only one image is selected, say "the first reference image" (or "the only reference image").
- When you mention a detail (hair/outfit/prop text/background), tie it to an ordinal reference image so the training target can align text to images.

In <think>:
- Write a practical plan and progress notes.
- Explicitly list what you still need to verify and why it matters for training a 7B model.
- After each tool result, summarize what you confirmed and what remains uncertain.
- Keep it concise; do not include unnecessary hidden reasoning.

In <answer>:
Return a single JSON object (not a JSON array) with these keys:
- "gen_prompt": a single grounded prompt for an image generation model (natural language, specific composition, camera, lighting, wardrobe, props, background, time/context). This prompt MUST NOT contain any URLs.
  - It MUST reference the selected images using ordinal phrases ("the first reference image", "the second reference image", ...).
  - It MUST NOT include IMG_### IDs.
- "reference_images": a list (1–5 items). Each item must be an object:
  {"img_id": "IMG_###", "note": "..."}
  describing what the image shows and what to copy.
  - "img_id" MUST be one of the IMG_### identifiers returned by image_search.
  - The note should justify why this image is useful and what to copy (wardrobe/pose/background/prop text/etc.).
  - Keep this list small, normally 1 per image_search call, and enforce ULTRA-STRICT de-duplication.
  - Reference image count must <= 5.

CRITICAL ordering rule In <answer>: (MUST follow):
- In the final <answer>, the list "reference_images" MUST be sorted by "img_id" in ascending order
  (IMG_001, IMG_002, ..., IMG_010, ...). Do NOT output them in any other order.
- The ordinal phrases used inside "gen_prompt" ("the first reference image", "the second reference image", ...)
  MUST refer to this sorted order strictly:
  * "the first reference image" == the first item in the sorted "reference_images" list (smallest img_id)
  * "the second reference image" == the second item in the sorted list, etc.
- Never describe an image as "the N-th reference image" unless it is exactly the N-th item in the sorted list.

Rules:
- Do not fabricate facts or URLs.
- Do not paste the entire user prompt verbatim into search. Search key entities/attributes and refine.
- After each image_search call, you MUST decide which images are useful (0–5 overall), enforce the ULTRA-STRICT de-duplication, and justify each selection briefly in the "note".
- Keep the final output grounded, precise, and suitable for training.

# Tools
You may call one function per round.

Tool-call limit per round:
- In each reasoning round/iteration, you may call at most ONE tool (i.e., only one <tool_call> block is allowed per round).

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Web text search tool that performs batched searches: supply an array 'queries'; the tool retrieves search results for each query.", "parameters": {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}, "description": "Array of query strings. You will get brief results with (title, url, snippet) for each query."}, "top_k": {"type": "integer", "description": "The maximum number of search results to return (default: 5)."}}, "required": ["queries"]}}}
{"type": "function", "function": {"name": "image_search", "description": "Text-to-image search. Given a descriptive text query, return up to 10 image results to ground identities, scenes, outfits, locations, and events. Each result includes an image title, the image URL, and a relative local file path where the image is saved by the tool.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "A descriptive text query for image search. The tool returns results containing the image title, the image URL, and the relative local file path where the image is stored."}, "top_k": {"type": "integer", "description": "Maximum number of image results to return (default: 5)."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "browse", "description": "Browse a webpage and extract relevant information based on a specific query.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The URL of the webpage to browse."}, "query": {"type": "string", "description": "The specific query to extract relevant information from the webpage."}}, "required": ["url", "query"]}}}
</tools>

For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Proceed step by step. Use as few tools as needed. Never repeat the same search.
In <think>, keep output concise; avoid long-winded reasoning.
 """


def today_date():
    """Get today's date in YYYY-MM-DD format."""
    return datetime.now().date().strftime("%Y-%m-%d")


def _shorten_text(text: str, max_len: int = 100) -> str:
    """Truncate text for log preview."""
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


class ImageIdManager:
    """Per-trajectory IMG_### id manager."""
    
    def __init__(self):
        self.counter = 0
        self.img_map: Dict[str, Dict[str, str]] = {}  # img_id -> {url, local_path, title, page_url}
        self.key_to_img_id: Dict[str, str] = {}  # (local_path or url) -> img_id
    
    def allocate_img_id(self) -> str:
        """Allocate the next IMG_### id."""
        self.counter += 1
        return f"IMG_{self.counter:03d}"
    
    def register_images(self, refs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Register image search results and return a list with img_id assigned."""
        out: List[Dict[str, str]] = []
        for r in refs:
            key = r.get("local_path") or r.get("url") or ""
            if not key:
                continue
            
            if key in self.key_to_img_id:
                img_id = self.key_to_img_id[key]
            else:
                img_id = self.allocate_img_id()
                self.key_to_img_id[key] = img_id
                self.img_map[img_id] = {
                    "img_id": img_id,
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "local_path": r.get("local_path", ""),
                    "page_url": r.get("page_url", ""),
                }
            
            rr = dict(r)
            rr["img_id"] = img_id
            out.append(rr)
        
        return out
    
    def format_image_search_response(self, query: str, refs: List[Dict[str, str]]) -> str:
        """Format image_search tool output (with IMG_###)."""
        lines = [f"--- image search result for [{query}] ---"]
        for r in refs:
            img_id = r.get("img_id", "")
            title = r.get("title", "image")
            url = r.get("url", "")
            local_path = r.get("local_path", "")
            page_url = r.get("page_url", "")
            lines.append(f"{img_id}: title: {title}")
            lines.append(f"  url: {url}")
            lines.append(f"  local_path: {local_path}")
            if page_url:
                lines.append(f"  page_url: {page_url}")
        lines.append("--- end of image search result ---")
        return "\n".join(lines)


def parse_image_search_results(tool_text: str, max_items: int = 10) -> List[Dict[str, str]]:
    """Parse image_search tool output text and extract image info."""
    lines = tool_text.splitlines()
    results: List[Dict[str, str]] = []
    cur: Dict[str, str] = {}
    
    def flush():
        nonlocal cur
        if cur.get("local_path") or cur.get("url") or cur.get("title"):
            results.append(cur)
        cur = {}
    
    for ln in lines:
        s = ln.strip()
        m_idx = re.match(r"^(\d+)\.\s*title:\s*(.*)$", s)
        if m_idx:
            flush()
            cur["title"] = m_idx.group(2).strip()
            continue
        if s.startswith("url:"):
            cur["url"] = s[len("url:"):].strip()
            continue
        if s.startswith("local_path:"):
            cur["local_path"] = s[len("local_path:"):].strip()
            continue
        if s.startswith("page_url:"):
            cur["page_url"] = s[len("page_url:"):].strip()
            continue
    
    flush()
    
    # Deduplicate
    seen = set()
    uniq: List[Dict[str, str]] = []
    for r in results:
        key = r.get("local_path") or r.get("url") or ""
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(r)
        if len(uniq) >= max_items:
            break
    return uniq


def extract_answer_json_from_text(text: str) -> Dict:
    """Extract JSON inside <answer>...</answer> from model output."""
    m = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Missing <answer>...</answer> block.")
    raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except Exception as e:
        raise ValueError(f"Invalid JSON inside <answer>: {e}")


class GenImageDeepResearchAgent:
    """
    Multi-round ReAct agent for image generation tasks.
    Manages IMG_### ids independently per trajectory.
    """
    
    def __init__(
        self,
        rollout_engine: RolloutEngine,
        tools: dict = None,
        system_prompt: str | None = None,
        **kwargs,
    ):
        self.rollout_engine = rollout_engine
        self.tools = tools or {}
        self.system_prompt = system_prompt or SYSTEM_PROMPT_GEN_IMAGE
        self.max_llm_calls = MAX_LLM_CALL_PER_RUN
        
        # Create a new ImageIdManager per run
        self.img_manager: Optional[ImageIdManager] = None
        
        # Token tracking (tracked per round before termination; includes vision/text split)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.response_token_lengths: List[int] = []
        self.tool_response_token_lengths: List[int] = []
        self.model_call_durations: List[float] = []
        self.tool_call_durations: List[float] = []
        self.total_model_time = 0.0
        self.total_tool_time = 0.0
        self.prompt_lengths: List[int] = []
        self.text_prompt_lengths: List[int] = []
        self.vision_prompt_lengths: List[int] = []
        self.per_image_vision_token_counts_by_round: List[List[int]] = []

        # Get context limits
        self.max_prompt_tokens, self.max_response_tokens, self.max_context_tokens = (
            self._get_model_context_limits(rollout_engine)
        )
        # Per-turn generation token limit (only caps rollout max_new_tokens per turn; training uses data.max_response_length)
        self.max_new_tokens_per_turn = int(os.environ.get("GEN_MAX_NEW_TOKENS_PER_TURN", "4096"))
        
        print("[GenImageAgent] Initialization completed")
        print(f"  max_prompt_tokens: {self.max_prompt_tokens}")
        print(f"  max_response_tokens: {self.max_response_tokens}")
        print(f"  max_new_tokens_per_turn: {self.max_new_tokens_per_turn}")
    
    def _get_model_context_limits(self, rollout_engine) -> tuple[int, int, int]:
        """Get token limits from the rollout engine."""
        default_prompt = 2048
        default_response = 2048
        
        max_prompt = default_prompt
        max_response = default_response
        
        config = rollout_engine.config if hasattr(rollout_engine, "config") else None
        data_cfg = config.data if config is not None and hasattr(config, "data") else None
        
        if data_cfg is not None:
            if hasattr(data_cfg, "max_prompt_length") and data_cfg.max_prompt_length:
                max_prompt = int(data_cfg.max_prompt_length)
            if hasattr(data_cfg, "max_response_length") and data_cfg.max_response_length:
                max_response = int(data_cfg.max_response_length)
        
        return max(max_prompt, 1), max(max_response, 1), max_prompt + max_response
    
    def record_token_usage(self, response) -> None:
        """Record token usage (tracked on normal returns; termination round stats may be carried by exceptions)."""
        prompt_tokens = getattr(response, "prompt_length", None)
        completion_tokens = getattr(response, "completion_length", None)
        text_pl = getattr(response, "text_prompt_length", None)
        vision_pl = getattr(response, "vision_prompt_length", None)

        if prompt_tokens is not None:
            try:
                p = int(prompt_tokens)
                self.total_prompt_tokens += p
                self.prompt_lengths.append(p)
                v = int(vision_pl) if vision_pl is not None else 0
                t = int(text_pl) if text_pl is not None else (p - v)
                self.text_prompt_lengths.append(t)
                self.vision_prompt_lengths.append(v)
                per_img = getattr(response, "per_image_vision_token_counts", None)
                if isinstance(per_img, list):
                    self.per_image_vision_token_counts_by_round.append(per_img.copy())
                    if per_img:
                        print(f"[GenImageAgent] Per-image vision token counts (this round): {per_img}")
                else:
                    self.per_image_vision_token_counts_by_round.append([])
            except (TypeError, ValueError):
                p = prompt_tokens if isinstance(prompt_tokens, int) else 0
                self.prompt_lengths.append(p)
                self.text_prompt_lengths.append(p)
                self.vision_prompt_lengths.append(0)
                self.per_image_vision_token_counts_by_round.append([])

        # Robust fallback chain for completion length:
        # 1) response.completion_length
        # 2) len(response.completion_ids)
        # 3) local tokenize(response.text)
        completion_tokens_int: Optional[int] = None
        if completion_tokens is not None:
            try:
                completion_tokens_int = int(completion_tokens)
            except (TypeError, ValueError):
                completion_tokens_int = None

        if completion_tokens_int is None:
            completion_ids = getattr(response, "completion_ids", None)
            if isinstance(completion_ids, list):
                completion_tokens_int = len(completion_ids)
            elif hasattr(completion_ids, "__len__"):
                try:
                    completion_tokens_int = len(completion_ids)
                except Exception:
                    completion_tokens_int = None

        if completion_tokens_int is None:
            text = ""
            if hasattr(response, "text") and response.text:
                text = response.text
            completion_tokens_int = self._estimate_text_tokens(text)

        if completion_tokens_int is not None and completion_tokens_int >= 0:
            self.total_completion_tokens += completion_tokens_int
            self.response_token_lengths.append(completion_tokens_int)

    def _estimate_text_tokens(self, text: str) -> int:
        """Estimate token length for plain text content."""
        if not text:
            return 0
        try:
            tokenizer = getattr(self.rollout_engine, "tokenizer", None)
            if tokenizer is not None:
                return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
        return 0
    
    async def call_server(self, messages: list[dict], **kwargs):
        """Call rollout engine to get model response. Per-turn generation is capped by GEN_MAX_NEW_TOKENS_PER_TURN."""
        try:
            # Filter out kwargs that should not be passed into the LLM call
            filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'global_step'}
            filtered_kwargs.setdefault("max_tokens", self.max_new_tokens_per_turn)
            print(f"[GenImageAgent] Calling model, messages count: {len(messages)}")
            response = await self.rollout_engine.get_model_response(messages=messages, **filtered_kwargs)
            return response
        except Exception as exc:
            print(f"[GenImageAgent] call_server failed: {exc}")
            raise
    
    async def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs) -> str:
        """Execute a tool call."""
        print(f"[GenImageAgent] Execute tool: {tool_name}")
        print(f"[GenImageAgent] Tool args: {json.dumps(tool_args, ensure_ascii=False)[:200]}")
        
        if tool_name not in self.tools:
            available = list(self.tools.keys())
            return f"Tool {tool_name} not found. Available tools: {available}"
        
        tool_call_start = time.perf_counter()
        try:
            tool = self.tools[tool_name]
            if hasattr(tool, "call"):
                if asyncio.iscoroutinefunction(tool.call):
                    result = await tool.call(**tool_args)
                else:
                    result = tool.call(**tool_args)
            elif callable(tool):
                result = tool(**tool_args)
            else:
                result = f"Tool {tool_name} is not callable"
            
            tool_call_elapsed = time.perf_counter() - tool_call_start
            self.tool_call_durations.append(tool_call_elapsed)
            self.total_tool_time += tool_call_elapsed
            result_str = str(result)
            print(f"[GenImageAgent] Tool result preview: {_shorten_text(result_str, 200)}")
            return result_str
            
        except Exception as e:
            tool_call_elapsed = time.perf_counter() - tool_call_start
            self.tool_call_durations.append(tool_call_elapsed)
            self.total_tool_time += tool_call_elapsed
            print(f"[GenImageAgent] Tool call failed: {e}")
            return f"Error calling tool {tool_name}: {e}"
    
    def _build_result(
        self,
        *,
        question: str,
        messages: list[dict],
        prediction: dict,
        termination: str,
        rounds: int,
        start_time: float,
        response_truncated_once: bool = False,
        format_error_once: bool = False,
    ) -> dict:
        """Build result dict. response_truncated_once/format_error_once are for logging only (no longer affect reward)."""
        token_usage = {
            "max_response_tokens": max(self.response_token_lengths, default=0),
            "response_token_lengths": self.response_token_lengths.copy(),
            "tool_response_token_lengths": self.tool_response_token_lengths.copy(),
            "prompt_lengths": self.prompt_lengths.copy(),
            "text_prompt_lengths": self.text_prompt_lengths.copy(),
            "vision_prompt_lengths": self.vision_prompt_lengths.copy(),
            "per_image_vision_token_counts": [r.copy() for r in self.per_image_vision_token_counts_by_round],
            "max_prompt_length": self.max_prompt_tokens,
            "max_response_length": self.max_response_tokens,
        }
        trajectory_total_time = time.time() - start_time
        timing = {
            "model_call_durations": [round(float(x), 1) for x in self.model_call_durations],
            "total_model_time": round(float(self.total_model_time), 1),
            "tool_call_durations": [round(float(x), 1) for x in self.tool_call_durations],
            "total_tool_time": round(float(self.total_tool_time), 1),
            "trajectory_total_time": round(float(trajectory_total_time), 1),
        }
        
        result = {
            "question": question,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "rounds": rounds,
            "time_taken": trajectory_total_time,
            "token_usage": token_usage,
            "timing": timing,
            "response_truncated_once": response_truncated_once,
            "format_error_once": format_error_once,
        }
        return result
    
    async def run(self, question: str, **kwargs) -> dict:
        """
        Main reasoning loop.
        
        Args:
            question: user image generation prompt
        
        Returns:
            Result dict containing messages, prediction, etc.
        """
        # Filter kwargs that should not be passed into the LLM call (workflow-only kwargs)
        llm_kwargs = {k: v for k, v in kwargs.items() if k != 'global_step'}
        
        start_time = time.time()
        
        # Create a new ImageIdManager per run
        self.img_manager = ImageIdManager()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.response_token_lengths = []
        self.tool_response_token_lengths = []
        self.model_call_durations = []
        self.tool_call_durations = []
        self.total_model_time = 0.0
        self.total_tool_time = 0.0
        self.prompt_lengths = []
        self.text_prompt_lengths = []
        self.vision_prompt_lengths = []
        self.per_image_vision_token_counts_by_round = []

        print(f"\n{'='*80}")
        print("[GenImageAgent] Start a new trajectory")
        print(f"[GenImageAgent] Question: {question}")
        print(f"{'='*80}\n")
        
        system_prompt = self.system_prompt + today_date()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        
        num_llm_calls_available = self.max_llm_calls
        round_idx = 0
        termination = "unknown"
        prediction = {}
        tool_calls_made = 0
        response_truncated_once = False  # At least one per-turn truncation happened
        format_error_once = False       # At least one format-error branch happened
        last_round_was_too_long = False  # Previous round was truncated and already warned; two consecutive -> repeated_response

        while num_llm_calls_available > 0:
            round_idx += 1
            num_llm_calls_available -= 1
            
            print(f"\n[GenImageAgent] ===== Round {round_idx} Start =====")
            print(f"[GenImageAgent] Remaining LLM calls: {num_llm_calls_available}")
            print(f"[GenImageAgent] Tool calls made: {tool_calls_made}")
            
            # Call the model
            llm_call_start = time.perf_counter()
            try:
                response = await self.call_server(messages, **llm_kwargs)
                llm_call_elapsed = time.perf_counter() - llm_call_start
                self.model_call_durations.append(llm_call_elapsed)
                self.total_model_time += llm_call_elapsed
            except Exception as exc:
                llm_call_elapsed = time.perf_counter() - llm_call_start
                self.model_call_durations.append(llm_call_elapsed)
                self.total_model_time += llm_call_elapsed
                print(f"[GenImageAgent] Model call failed: {exc}")
                prediction = {"error": f"call_server failed: {exc}"}
                exc_str = str(exc)
                if "TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED" in exc_str or "TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED" in exc_str:
                    termination = "max_response_length_exceeded"
                    prediction["error_type"] = "max_response_length_exceeded"
                    if isinstance(exc, TerminationEvent) and getattr(exc, "prompt_length", None) is not None:
                        self.prompt_lengths.append(exc.prompt_length)
                        self.text_prompt_lengths.append(
                            exc.text_prompt_length if getattr(exc, "text_prompt_length", None) is not None else exc.prompt_length
                        )
                        self.vision_prompt_lengths.append(
                            getattr(exc, "vision_prompt_length", None) or 0
                        )
                        per_img = getattr(exc, "per_image_vision_token_counts", None)
                        self.per_image_vision_token_counts_by_round.append(
                            list(per_img) if isinstance(per_img, list) else []
                        )
                        if per_img:
                            print(f"[GenImageAgent] Per-image vision token counts (before length termination): {per_img}")
                    print("[GenImageAgent] Hit MAX_RESPONSE_LENGTH_EXCEEDED (trajectory too long); terminate this trajectory and return")
                    result = self._build_result(
                        question=question,
                        messages=messages,
                        prediction=prediction,
                        termination=termination,
                        rounds=round_idx,
                        start_time=start_time,
                    )
                    token_usage = result.get("token_usage", {})
                    print(
                        "[GenImageAgent] MAX_RESPONSE_LENGTH_EXCEEDED token stats: "
                        f"prompt_lengths={token_usage.get('prompt_lengths', [])}, "
                        f"text_prompt_lengths={token_usage.get('text_prompt_lengths', [])}, "
                        f"vision_prompt_lengths={token_usage.get('vision_prompt_lengths', [])}, "
                        f"max_prompt_length={token_usage.get('max_prompt_length')}, "
                        f"response_token_lengths={token_usage.get('response_token_lengths', [])}, "
                        f"tool_response_token_lengths={token_usage.get('tool_response_token_lengths', [])}"
                    )
                    return result
                else:
                    termination = "error"
                return self._build_result(
                    question=question,
                    messages=messages,
                    prediction=prediction,
                    termination=termination,
                    rounds=round_idx,
                    start_time=start_time,
                )
            
            self.record_token_usage(response)
            finish_reason = getattr(response, "finish_reason", None)  # "length" means per-turn max_tokens truncation

            content = response.text if hasattr(response, "text") and response.text else ""
            n_tok = self.response_token_lengths[-1] if self.response_token_lengths else 0
            print(f"[GenImageAgent] Model output length (tokens): {n_tok}")
            print(f"[GenImageAgent] Model output preview: {_shorten_text(content, 500)}")
            
            # Strip tool_response tail from content
            if "<tool_response>" in content:
                pos = content.find("<tool_response>")
                content = content[:pos]
            
            # Detect repeated response: if identical to the previous assistant content, terminate immediately
            last_assistant_content = None
            for m in reversed(messages):
                if m.get("role") == "assistant":
                    last_assistant_content = (m.get("content") or "").strip()
                    break
            if last_assistant_content is not None and content.strip() == last_assistant_content:
                print("[GenImageAgent] Detected repeated response (identical to previous round); terminate immediately")
                termination = "repeated_response"
                prediction = {"error": "repeated_response", "error_type": "repeated_response"}
                messages.append({"role": "assistant", "content": content.strip()})
                result = self._build_result(
                    question=question,
                    messages=messages,
                    prediction=prediction,
                    termination=termination,
                    rounds=round_idx,
                    start_time=start_time,
                )
                return result
            
            # Check for tool_call
            if "<tool_call>" in content and "</tool_call>" in content:
                last_round_was_too_long = False  # Tool call produced; clear previous "too long" marker
                tool_calls_made += 1
                print(f"[GenImageAgent] Detected tool_call (#{tool_calls_made})")
                
                assistant_message = {"role": "assistant", "content": content.strip()}
                messages.append(assistant_message)
                
                tool_call_text = content.split("<tool_call>")[1].split("</tool_call>")[0]
                
                try:
                    tool_call = json5.loads(tool_call_text)
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("arguments", {})
                    
                    print(f"[GenImageAgent] Tool name: {tool_name}")
                    
                    # Execute tool
                    tool_result = await self.custom_call_tool(tool_name, tool_args)
                    
                    # For image_search, assign IMG_### ids
                    if tool_name == "image_search":
                        print("[GenImageAgent] Processing image_search results...")
                        refs_all = parse_image_search_results(tool_result, max_items=10)
                        print(f"[GenImageAgent] Parsed {len(refs_all)} images")
                        
                        # Register images and assign IMG_### ids
                        refs_with_ids = self.img_manager.register_images(refs_all)
                        print(f"[GenImageAgent] IMG ids assigned: {[r.get('img_id') for r in refs_with_ids]}")
                        
                        # Format content returned to the model
                        query = tool_args.get("query", "image_search")
                        tool_result_formatted = self.img_manager.format_image_search_response(query, refs_with_ids)
                        
                        tool_response = f"<tool_response>\n{tool_result_formatted}\n</tool_response>"
                        # Only take the first N images as multimodal inputs per round
                        # (the parser accumulates all user messages with images across rounds)
                        images_for_model: List[Dict[str, str]] = []
                        for r in refs_with_ids[:MAX_IMAGES_PER_SEARCH_FOR_MODEL]:
                            path = (r.get("local_path") or "").strip()
                            if path and os.path.exists(path):
                                images_for_model.append({"image": path})
                        user_msg: Dict[str, Any] = {"role": "user", "content": tool_response}
                        if images_for_model:
                            user_msg["images"] = images_for_model
                        print(
                            f"[GenImageAgent] Images passed to model this round: {len(images_for_model)} "
                            f"(max {MAX_IMAGES_PER_SEARCH_FOR_MODEL})"
                        )
                        print(f"[GenImageAgent] Tool response preview: {_shorten_text(tool_response, 500)}")
                        t_tok = self._estimate_text_tokens(tool_response)
                        self.tool_response_token_lengths.append(t_tok)
                        print(f"[GenImageAgent] Tool output length (tokens): {t_tok}")
                        messages.append(user_msg)
                    else:
                        tool_response = f"<tool_response>\n{tool_result}\n</tool_response>"
                        print(f"[GenImageAgent] Tool response preview: {_shorten_text(tool_response, 500)}")
                        t_tok = self._estimate_text_tokens(tool_response)
                        self.tool_response_token_lengths.append(t_tok)
                        print(f"[GenImageAgent] Tool output length (tokens): {t_tok}")
                        messages.append({"role": "user", "content": tool_response})
                
                except Exception as e:
                    print(f"[GenImageAgent] Tool call parse/execute failed: {e}")
                    error_msg = f"[Tool Error]: {e}"
                    tool_response = f"<tool_response>\n{error_msg}\n</tool_response>"
                    t_tok = self._estimate_text_tokens(tool_response)
                    self.tool_response_token_lengths.append(t_tok)
                    print(f"[GenImageAgent] Tool output length (tokens): {t_tok}")
                    messages.append({"role": "user", "content": tool_response})
                
                continue
            
            # Check for final answer
            elif "<answer>" in content and "</answer>" in content:
                last_round_was_too_long = False  # Answer produced; clear previous "too long" marker
                print("[GenImageAgent] Detected <answer>; extracting final answer")
                
                messages.append({"role": "assistant", "content": content.strip()})
                
                try:
                    ans = extract_answer_json_from_text(content)
                    print("[GenImageAgent] Parsed answer JSON successfully")
                    print(f"[GenImageAgent] gen_prompt: {_shorten_text(ans.get('gen_prompt', ''), 500)}")
                    print(f"[GenImageAgent] reference_images count: {len(ans.get('reference_images', []))}")
                    
                    # Validate fields (require gen_prompt and reference_images)
                    if not isinstance(ans, dict):
                        raise ValueError("Answer JSON is not an object.")
                    for k in ["gen_prompt", "reference_images"]:
                        if k not in ans:
                            raise ValueError(f"Answer JSON missing key: {k}")
                    
                    raw_refs = ans.get("reference_images", [])
                    if not isinstance(raw_refs, list):
                        raise ValueError("reference_images must be a list.")
                    
                    # Enrich reference_images (fetch full info from img_map)
                    enriched_refs: List[Dict[str, str]] = []
                    for r in raw_refs[:5]:
                        if not isinstance(r, dict):
                            continue
                        img_id = (r.get("img_id") or "").strip()
                        note = r.get("note", "")
                        if not img_id:
                            raise ValueError("reference_images item missing img_id.")
                        if img_id not in self.img_manager.img_map:
                            raise ValueError(f"Unknown img_id in answer: {img_id}")
                        
                        src = self.img_manager.img_map[img_id]
                        enriched_refs.append({
                            "img_id": img_id,
                            "url": src.get("url", ""),
                            "local_path": src.get("local_path", ""),
                            "title": src.get("title", ""),
                            "page_url": src.get("page_url", ""),
                            "note": note,
                        })
                    
                    print(f"[GenImageAgent] Enriched reference_images: {[r.get('img_id') for r in enriched_refs]}")
                    
                    final_json = {
                        "gen_prompt": ans.get("gen_prompt", ""),
                        "reference_images": enriched_refs,
                    }
                    
                    prediction = final_json
                    termination = "answer"
                    
                except Exception as e:
                    print(f"[GenImageAgent] Failed to parse answer: {e}")
                    prediction = {"error": f"answer_parse_failed: {e}"}
                    termination = "answer_parse_failed"
                
                break
            
            # Format error (missing </tool_call> or </answer>).
            # If caused by per-turn truncation, use the dedicated hint and mark response_truncated_once; otherwise mark format_error_once.
            else:
                print("[GenImageAgent] Format error: missing <tool_call> or <answer> tags")
                messages.append({"role": "assistant", "content": content.strip()})
                if finish_reason == "length":
                    # Two consecutive overlong rounds -> repeated_response and terminate immediately
                    if last_round_was_too_long:
                        print(
                            "[GenImageAgent] Two consecutive rounds exceeded GEN_MAX_NEW_TOKENS_PER_TURN; "
                            "mark as repeated_response and terminate"
                        )
                        termination = "repeated_response"
                        prediction = {"error": "repeated_response", "error_type": "repeated_response"}
                        result = self._build_result(
                            question=question,
                            messages=messages,
                            prediction=prediction,
                            termination=termination,
                            rounds=round_idx,
                            start_time=start_time,
                            response_truncated_once=response_truncated_once,
                            format_error_once=format_error_once,
                        )
                        return result
                    last_round_was_too_long = True
                    response_truncated_once = True
                    observation = (
                        "=== RESPONSE TOO LONG ===\n"
                        "Your previous response was TRUNCATED because it exceeded the length limit.\n"
                        "You MUST be MUCH MORE CONCISE.\n\n"
                        "CRITICAL RULES:\n"
                        "- Do NOT write <think>. Output ONLY <tool_call> or <answer>.\n"
                        "- Output format: <tool_call>{json}</tool_call> OR <answer>{json}</answer>\n"
                        "- Keep it SHORT. No explanations, no reasoning text.\n"
                        "- If you cannot fit the response, output <answer> with what you have.\n\n"
                        "Respond NOW with ONE of: <tool_call>...</tool_call> OR <answer>...</answer>"
                    )
                else:
                    last_round_was_too_long = False
                    observation = "Error: Invalid format. You MUST output exactly one of: <tool_call>...</tool_call> OR <answer>...</answer>. Do not output both. Try again."
                    format_error_once = True
                messages.append({"role": "user", "content": observation})
                continue
        
        # Reached max turns without an answer: inject FINAL_MESSAGE and give one last chance
        if termination != "answer" and round_idx >= self.max_llm_calls:
            print(
                f"[GenImageAgent] Reached max turns ({self.max_llm_calls}); inject FINAL_MESSAGE and request final answer"
            )
            messages.append({"role": "user", "content": FINAL_MESSAGE})
            try:
                llm_call_start = time.perf_counter()
                response = await self.call_server(messages, **llm_kwargs)
                llm_call_elapsed = time.perf_counter() - llm_call_start
                self.model_call_durations.append(llm_call_elapsed)
                self.total_model_time += llm_call_elapsed
                self.record_token_usage(response)
                content = response.text if hasattr(response, "text") and response.text else ""
                n_tok = self.response_token_lengths[-1] if self.response_token_lengths else 0
                print(f"[GenImageAgent] Final round model output length (tokens): {n_tok}")
                print(f"[GenImageAgent] Final round model output preview: {_shorten_text(content, 500)}")
                
                # In the final round, if the model still outputs <tool_call>, forcibly ignore it
                if "<tool_call>" in content:
                    print("[GenImageAgent] WARNING: Final round still produced <tool_call>; it will be ignored")
                    # Truncate content before <tool_call>
                    tool_call_pos = content.find("<tool_call>")
                    content = content[:tool_call_pos]
                
                if "<tool_response>" in content:
                    pos = content.find("<tool_response>")
                    content = content[:pos]
                # Record the final-round assistant response regardless of whether <answer> is present
                messages.append({"role": "assistant", "content": content.strip()})
                if "<answer>" in content and "</answer>" in content:
                    try:
                        ans = extract_answer_json_from_text(content)
                        if isinstance(ans, dict) and "gen_prompt" in ans and "reference_images" in ans:
                            raw_refs = ans.get("reference_images", [])
                            if isinstance(raw_refs, list):
                                enriched_refs = []
                                for r in raw_refs[:5]:
                                    if not isinstance(r, dict):
                                        continue
                                    img_id = (r.get("img_id") or "").strip()
                                    note = r.get("note", "")
                                    if not img_id or img_id not in self.img_manager.img_map:
                                        continue
                                    src = self.img_manager.img_map[img_id]
                                    enriched_refs.append({
                                        "img_id": img_id,
                                        "url": src.get("url", ""),
                                        "local_path": src.get("local_path", ""),
                                        "title": src.get("title", ""),
                                        "page_url": src.get("page_url", ""),
                                        "note": note,
                                    })
                                prediction = {
                                    "gen_prompt": ans.get("gen_prompt", ""),
                                    "reference_images": enriched_refs,
                                }
                                termination = "answer"
                    except Exception as e:
                        print(f"[GenImageAgent] Final round failed to parse answer: {e}")
                        print("[GenImageAgent] Final round did not produce a valid answer; mark as no_answer")
            except Exception as exc:
                llm_call_elapsed = time.perf_counter() - llm_call_start
                self.model_call_durations.append(llm_call_elapsed)
                self.total_model_time += llm_call_elapsed
                print(f"[GenImageAgent] Final round model call failed: {exc}")
        
        # Build result
        if not prediction or (isinstance(prediction, dict) and "error" not in prediction):
            if termination != "answer":
                print("[GenImageAgent] No valid answer found")
                prediction = {"error": "No valid answer found"}
                termination = "no_answer"
        
        result = self._build_result(
            question=question,
            messages=messages,
            prediction=prediction,
            termination=termination,
            rounds=round_idx,
            start_time=start_time,
            response_truncated_once=response_truncated_once,
            format_error_once=format_error_once,
        )
        
        print("\n[GenImageAgent] ===== Trajectory Completed =====")
        print(f"[GenImageAgent] Rounds: {round_idx}")
        print(f"[GenImageAgent] Termination: {termination}")
        print(f"[GenImageAgent] Time: {result['time_taken']:.1f}s")
        print(
            "[GenImageAgent] Token usage: "
            f"max_response_tokens={result['token_usage'].get('max_response_tokens', 0)}, "
            f"response_token_lengths={result['token_usage'].get('response_token_lengths', [])}, "
            f"tool_response_token_lengths={result['token_usage'].get('tool_response_token_lengths', [])}"
        )
        
        return result
    
    def reset(self):
        """Reset agent state."""
        self.img_manager = None
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.response_token_lengths = []
        self.tool_response_token_lengths = []
        self.model_call_durations = []
        self.tool_call_durations = []
        self.total_model_time = 0.0
        self.total_tool_time = 0.0
