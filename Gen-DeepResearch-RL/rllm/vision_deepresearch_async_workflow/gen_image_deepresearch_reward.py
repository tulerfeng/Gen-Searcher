"""
Gen Image Reward: GPT-4.1 scoring aligned with gpt_eval_knowgen (same SYSTEM_PROMPT, same overall formula).
Scoring uses the original task prompt (question), not the model-generated gen_prompt.
"""
from __future__ import annotations

import os
import re
import json
import random
import base64
import io
import traceback
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import BoundedSemaphore, Lock
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Any

from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

try:
    from openai import OpenAI, RateLimitError
except ImportError:
    OpenAI = None
    RateLimitError = Exception

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None

from rllm.agents.agent import Episode

# ---------- KnowGen-aligned: GPT-4.1 prompt + scoring (aligned with gpt_eval_knowgen.py) ----------
WORLDGEN_MAX_SIDE = 4096
WORLDGEN_JPEG_QUALITY = 100
WORLDGEN_LLM_TIMEOUT = 300.0
WORLDGEN_LLM_MAX_TRY = 30
WORLDGEN_LLM_MAX_TOKENS = 8192
WORLDGEN_LLM_TEMPERATURE = 0.0


WORLDGEN_SYSTEM_PROMPT = r"""You are a strict and professional expert evaluator for AI-generated image grounded with world knowledge (MODEL EVALUATION).

You will receive:
1) A task prompt (what the image must show).
2) Image 1: the generated image (model output to be evaluated).
3) Image 2: the ground-truth reference image (a strong reference implementation).

All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

Critical clarification (VERY IMPORTANT):
- This is NOT a pixel-level similarity task.
- Image 2 (GT) is a REFERENCE for intended identity, key grounded details, and stable visual attributes.
  Image 1 may use a different camera angle/layout as long as it still satisfies the prompt.
- Focus on whether prompt-required, externally-checkable (search-grounded) details are correctly AND verifiably realized in Image 1.
- Do NOT assume correctness if a key detail is not clearly visible/readable. If unverifiable, score lower.

Output format (MUST follow exactly):
Output ONLY one valid JSON object with EXACTLY these keys:
{
  "rationale": string,
  "faithfulness": number,
  "visual_correctness": number,
  "text_accuracy": number,
  "aesthetics": number
  "text_accuracy_na": boolean,
}
SCORING SCALE (VERY IMPORTANT):
- Each score MUST be exactly one of: 0, 0.5, 1
- 1 (Exemplary) is rare and requires perfect success for that dimension.
- 0.5 (Conditional) means mostly correct but not perfect.
- 0 (Rejected) means failed on important requirements.

- "rationale" must be 5–10 short sentences, evidence-based, referring only to what is visible.
- "text_accuracy_na" should be true if the prompt does not require any readable text, otherwise it should be false.

Implicit required step (ENFORCED via rationale):
- In the rationale, you MUST explicitly list the extracted prompt hard constraints (2–5, or more if needed) BEFORE scoring.
  If you cannot identify the constraints, you must still list what you believe are the hard constraints.

Evaluation procedure (follow silently, but the rationale MUST reflect it):
1) Extract the prompt’s TOP hard constraints (2–5, or more if needed): required subjects/identities, setting/props,
   relations/counts, required style, and any externally-checkable requirements (readable text/landmark/logo/badge/version/year/etc.).
2) Score Image 1 against the constraints. Use Image 2 only as a reference for stable identity/visual attributes and grounded evidence.
3) If a key requirement is not verifiable (too small/blurred/occluded/warped), do NOT assume it is correct; score lower.
4) Assessment of the primary subjects' visual identity correctness and consistency is mandatory in every case.

Boundary between visual_correctness vs text_accuracy:
- Visual-only grounded cues (subject visual features, logo SHAPE, badge EMBLEM geometry, landmark facade/massing, outfit/weapon silhouette, object geometry)
  belong to visual_correctness.
- Any grounded cue that must be READ as text (spelling, year numbers, titles, institution names, badge text) belongs to text_accuracy.

==========================
STRICT 3-LEVEL RUBRICS
(Each dimension uses ONLY {0, 0.5, 1})
==========================

1) faithfulness (overall prompt adherence: presence & structure only; not GT-identity correctness):
- This score does NOT require matching GT’s exact identity or fine-grained visual features; it focuses on whether Image 1 includes the prompt-requested elements and scene structure (who/what is present, what is happening, where it happens, and the required style/format).

(Exemplary) Score = 1 ONLY IF:
- Image 1 clearly includes everything the prompt asks for in terms of visible content and structure:
  all required subjects/entities are present, the required setting and key props appear,
  required actions/relations/counts are shown, and the required style/format is followed.
- Any required in-scene evidence elements requested by the prompt (e.g., a plaque/sign, a map, a report paper, a badge) are present as elements.

(Conditional) Score = 0.5 ONLY IF:
- Image 1 includes almost all prompt-requested content and structure, with only minor omissions or minor staging differences
  that do not change what the scene is supposed to depict (e.g., small placement differences, slight simplification of a secondary prop).

(Rejected) Score = 0 IF:
- One or more prompt-requested essential elements are not shown at all, or the scene structure clearly does not match the prompt’s request
  (e.g., missing a required subject/entity, missing the required setting, missing the required key prop/evidence element,
  missing the requested action/relationship/count, or not following the requested style/format).

2) visual_correctness (GT visual-feature agreement is the core; extremely strict):
(Exemplary) Score = 1 ONLY IF:
- The prompt-required primary subjects/objects in Image 1 match the GT reference (Image 2) in visual characteristics
  with NO substantive changes.
- This means: the same face/hairstyle silhouette, the same armor/clothing design and key colors/patterns,
  the same distinctive props/object geometry, the same emblem/logo/landmark facade/massing cues when applicable, etc.
- Any meaningful difference in these stable visual features disqualifies a score of 1.

(Conditional) Score = 0.5 ONLY IF:
- Image 1 can still be considered the same overall visual instance as the GT, and the differences are limited to relatively minor variations, allowing some changes to the visual features (face, hairstyle, armor design, key colors/patterns, key prop shapes), while the overall identity and major visual features remain recognizable and broadly consistent.
- IMPORTANT: "same role archetype" (generic knight/princess/warrior) alone does NOT qualify for 0.5.

(Rejected) Score = 0 IF:
- Any substantive mismatch vs GT in stable visual features of the required subjects/objects
  (different face/hair/armor design/color scheme/emblem/prop geometry/landmark cues),
  even if the overall scene still looks plausible or stylistically similar.

3) text_accuracy (required readable text; ALL relevant text must be correct AND very clearly readable; NO partial credit for wrong text):
Rule:
- If the prompt does NOT require any readable text: you MUST output "text_accuracy_na": true and "text_accuracy": 0.5 in the JSON. In your rationale state that the prompt did not require readable text.
- If the prompt DOES require readable text: output "text_accuracy_na": false and score "text_accuracy" (0, 0.5, or 1) per the criteria below.

(Exemplary) Score = 1 ONLY IF:
- ALL required text AND any prompt-involved text elements are:
  (a) present,
  (b) very clearly readable (crisp, unambiguous),
  (c) correct and consistent with the prompt’s requirements.
(Conditional) Score = 0.5 ONLY IF:
- Much of the required/prompt-involved text is readable and generally correct, and although parts may contain inaccuracies, omissions, or deviations, the overall meaning remains clear and is not seriously inconsistent with the prompt requirements.
(Rejected) Score = 0 IF:
- Any required/prompt-involved text is missing, unclear, not very readable, gibberish, placeholder, OR incorrect.
- Even if perfectly readable, if content is not correct, text_accuracy MUST be 0.

4) aesthetics:
(Exemplary) Score = 1 ONLY IF:
- Masterpiece-level composition and polish, AND Image 1 is NOT worse than GT in overall aesthetic quality.
(Conditional) Score = 0.5 ONLY IF:
- Very beautiful and polished, but slightly worse than GT (ONLY slightly) OR slightly less refined than top-tier.
(Rejected) Score = 0 IF:
- Anything clearly worse than GT in a noticeable way, OR merely average/OK-looking, OR cluttered/awkward framing,
  OR visible artifacts/noise that harm the overall appeal.

Rationale requirements (MANDATORY):
- Start with: "Constraints:" and list the extracted constraints (2–5, or more if needed).
- State whether the prompt required readable text; if not required, output "text_accuracy_na": true and "text_accuracy": 0.5 in the JSON and say so in the rationale.
- Mention 2–5 key comparisons (or more if needed) to GT focused on stable identity/visual traits (NOT demanding identical layout).
- Keep within 10 sentences.

Output JSON only. No markdown. No extra text."""


def _encode_image_to_data_url_reward(path: str, max_side: int = WORLDGEN_MAX_SIDE, quality: int = WORLDGEN_JPEG_QUALITY) -> str:
    if not Image or not ImageOps:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        w, h = im.size
        longest = max(w, h)
        if longest > max_side:
            scale = max_side / float(longest)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in (im.info or {})):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            im = im.convert("RGBA")
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(quality), optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"


def _build_user_message_worldgen(sample_id: str, prompt: str, gen_path: str, gt_path: str) -> dict:
    text = (
        f"Sample id: {sample_id}\n\n"
        f"Task prompt (image requirement):\n{prompt}\n\n"
        "Image 1: the **generated image** (to be evaluated).\n"
        "Image 2: the **ground-truth reference image**.\n\n"
        "Output a single JSON object with the three scores and a short rationale."
    )
    content = [{"type": "text", "text": text}]
    for p in [gen_path, gt_path]:
        try:
            url = _encode_image_to_data_url_reward(p)
            content.append({"type": "image_url", "image_url": {"url": url}})
        except Exception as e:
            content.append({"type": "text", "text": f"\n[WARN] failed to load image: {p}, err={e}\n"})
    return {"role": "user", "content": content}


def _parse_llm_json_reward(content: str) -> dict:
    content = (content or "").strip()
    if not content:
        raise ValueError("empty_content")
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z0-9]*\s*", "", content)
        content = re.sub(r"\s*```$", "", content).strip()
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        content = content[start : end + 1]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    content = re.sub(r",\s*}", "}", content)
    content = re.sub(r",\s*]", "]", content)
    return json.loads(content)


def _normalize_scores_reward(obj: Any) -> Tuple[float, float, Optional[float], float, str]:
    """(faithfulness, visual_correctness, text_accuracy|None, aesthetics, rationale). text_accuracy None => N/A."""
    out: Tuple[float, float, Optional[float], float, str] = (0.0, 0.0, 0.0, 0.0, "")
    if not isinstance(obj, dict):
        return out

    def clip01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    def round01(x: float) -> float:
        return round(clip01(x), 2)

    f = round01(float(obj.get("faithfulness", 0)))
    v = round01(float(obj.get("visual_correctness", 0)))
    text_na = obj.get("text_accuracy_na")
    if text_na in (True, "true", "True", 1):
        t: Optional[float] = None
    else:
        try:
            t = round01(float(obj.get("text_accuracy", 0)))
        except (TypeError, ValueError):
            t = None
    a = round01(float(obj.get("aesthetics", 0)))
    r = str(obj.get("rationale", ""))[:1000]
    return (f, v, t, a, r)


def call_gpt41_worldgen_score(
    generated_image_path: str,
    gt_image_path: str,
    prompt: str,
    sample_id: str,
    api_key: str,
    api_base: str,
    model: str = "gpt-4.1",
) -> Tuple[float, Optional[dict]]:
    """
    Aligned with gpt_eval_knowgen:
    same SYSTEM_PROMPT, same user-message construction, same overall formula.

    overall = 0.1*faithfulness + 0.4*visual_correctness + 0.4*text_accuracy + 0.1*aesthetics
    (when text is N/A, text_accuracy uses 0.5).
    """
    if not OpenAI:
        print("[GenReward] openai not installed, worldgen score unavailable")
        return 0.0, None
    try:
        user_msg = _build_user_message_worldgen(sample_id, prompt, generated_image_path, gt_image_path)
        messages = [
            {"role": "system", "content": WORLDGEN_SYSTEM_PROMPT},
            user_msg,
        ]
        base_url = (api_base or "").strip()
        if base_url and not base_url.endswith("/"):
            base_url = base_url + "/"
        client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=WORLDGEN_LLM_TIMEOUT)
        max_try = int(os.environ.get("GEN_REWARD_MAX_TRY", str(WORLDGEN_LLM_MAX_TRY)))
        last_exc: Optional[BaseException] = None
        for attempt in range(1, max_try + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=WORLDGEN_LLM_TEMPERATURE,
                    max_tokens=WORLDGEN_LLM_MAX_TOKENS,
                    timeout=WORLDGEN_LLM_TIMEOUT,
                )
                msg = getattr(resp, "choices", [None])[0]
                if msg is not None:
                    msg = getattr(msg, "message", None)
                content = getattr(msg, "content", None) if msg else None
                if not isinstance(content, str):
                    raise RuntimeError("empty_content")
                obj = _parse_llm_json_reward(content)
                if not isinstance(obj, dict):
                    raise ValueError("LLM output is not a JSON object.")
                f, v, t, a, _ = _normalize_scores_reward(obj)
                t_val = 0.5 if t is None else t
                overall = round(0.1 * f + 0.4 * v + 0.4 * t_val + 0.1 * a, 2)
                overall = max(0.0, min(1.0, overall))
                print(f"[GenReward] GPT-4.1 worldgen scores: f={f} v={v} t={t_val} a={a} overall={overall}")
                # Return the score and the full LLM JSON (saved into result.json as reward_response)
                return float(overall), obj
            except RateLimitError as e:
                last_exc = e
                if attempt % 5 == 0:
                    print(f"[GenReward] [429 RETRY] attempt={attempt}/{max_try}")
                time.sleep(min(10.0, 0.5 * (2 ** (attempt - 1))) * (0.8 + 0.4 * random.random()))
            except Exception as e:
                last_exc = e
                print(f"[GenReward] [LLM ERROR] attempt={attempt}/{max_try} err={str(e)[:150]}")
                if attempt >= max_try:
                    break
                time.sleep(min(10.0, 0.5 * (2 ** (attempt - 1))) * (0.8 + 0.4 * random.random()))
        if last_exc:
            print(f"[GenReward] GPT-4.1 worldgen score failed: {last_exc}")
        return 0.0, None
    except Exception as e:
        print(f"[GenReward] GPT-4.1 worldgen score failed: {e}")
        traceback.print_exc()
        return 0.0, None


# ---------- Text-based reward: no generated image; evaluate whether model <answer> is sufficient to produce the GT ----------
TEXT_REWARD_VALID_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)

TEXT_REWARD_SYSTEM_PROMPT = r"""You are an expert evaluator for a text-based image generation pipeline.

You will receive:
1) Task prompt: the original user requirement (what image we want to generate).
2) Ground-truth reference image: the target image we want the pipeline to produce.
3) Model's answer: the model's output in <answer>, containing:
   - gen_prompt: a natural-language prompt for an image generator (composition, style, subjects, etc.).
   - reference_images: a list of chosen reference images (each with img_id, title, note, etc.) that the model selected from search to guide generation.

Your task (TEXT + VISUAL):
- From both TEXT and VISUAL perspectives, judge how well this answer (gen_prompt + reference image choices) would support generating an image that matches the ground-truth.
- You are NOT evaluating an actual generated image here. You are evaluating whether the model's textual output (search choices + generation prompt) is well-aligned with the task and the GT: i.e., if we had a perfect image generator, would this answer be sufficient to produce the GT?
- Consider: Does the gen_prompt capture the key requirements from the task and the GT? Are the chosen reference images (by their titles/notes and what they typically show) appropriate for producing the GT? Are there critical missing or wrong elements?

Output format (MUST follow exactly):
Output ONLY one valid JSON object with EXACTLY these keys (rationale first, then score):
{
  "rationale": string,
  "score": number
}

Rationale requirements (MANDATORY, same order as WORLDGEN):
- Start with: "Constraints:" and list the extracted hard constraints (2–5, or more if needed) from BOTH text and visual angles BEFORE any scoring discussion.
  - Text angle: extract from the task prompt and the gen_prompt (required subjects/identities, setting, style, key props, readable text if any, etc.).
  - Visual angle: extract from the GT image (key visual features, identity cues, composition, style, details that the answer should support producing).
- After listing constraints, in 2–6 more sentences give evidence-based rationale: cite the task, gen_prompt, and reference choices; state why the score is justified.
- If you cannot identify the constraints, you must still list what you believe are the hard constraints.
- Total rationale: 5–10 short sentences.

SCORING SCALE (VERY IMPORTANT):
- "score" MUST be exactly one of: 0, 0.25, 0.5, 0.75, 1.0

1.0 (Exemplary): The answer is fully sufficient. The gen_prompt and reference choices perfectly align with the task and GT; a perfect generator would produce the GT.

0.75 (Very good): Strong alignment; at most minor gaps or imprecisions.

0.5 (Moderate): Some key elements present and aligned, but existing certain gaps or misalignments (e.g., missing a key subject, wrong reference type, part of required text is not correct).

0.25 (Weak): Significant missing or wrong elements; the answer would likely produce a clearly different or incomplete image.

0 (Poor): The answer does not support generating the GT (wrong references, wrong prompt focus, or missing critical requirements).

Output JSON only. No markdown. No extra text."""


def _build_user_message_text_reward(sample_id: str, task_prompt: str, gt_image_path: str, answer_content: str) -> dict:
    """User message for text-based reward: task prompt, GT image, and model answer text."""
    text = (
        f"Sample id: {sample_id}\n\n"
        f"Task prompt (what image we want to generate):\n{task_prompt}\n\n"
        "Ground-truth reference image (Image 1 below): the target image the pipeline should produce.\n\n"
        "Model's <answer> content (gen_prompt + reference_images):\n"
        f"{answer_content}\n\n"
        "From both text and visual angles, score how well this answer would support generating an image that matches the GT (0, 0.25, 0.5, 0.75, or 1.0). Output JSON with 'score' and 'rationale'."
    )
    content = [{"type": "text", "text": text}]
    try:
        url = _encode_image_to_data_url_reward(gt_image_path)
        content.append({"type": "image_url", "image_url": {"url": url}})
    except Exception as e:
        content.append({"type": "text", "text": f"\n[WARN] failed to load GT image: {gt_image_path}, err={e}\n"})
    return {"role": "user", "content": content}


def _parse_text_reward_score(obj: Any) -> float:
    """Parse LLM output to a score in TEXT_REWARD_VALID_SCORES. Default 0.0 if invalid."""
    if not isinstance(obj, dict):
        return 0.0
    try:
        s = float(obj.get("score", 0))
    except (TypeError, ValueError):
        return 0.0
    # snap to nearest valid score
    best = 0.0
    for v in TEXT_REWARD_VALID_SCORES:
        if abs(s - v) < abs(s - best):
            best = v
    return max(0.0, min(1.0, best))


def call_text_reward_score(
    task_prompt: str,
    gt_image_path: str,
    answer_content: str,
    sample_id: str,
    api_key: str,
    api_base: str,
    model: str = "gpt-4.1",
) -> Tuple[float, Optional[dict]]:
    """
    Text-based reward: given the original prompt, GT image, and model <answer>,
    evaluate (from text + visual perspectives) whether the answer is sufficient to produce the GT.

    Returns (score, response_dict). score is one of 0 / 0.25 / 0.5 / 0.75 / 1.0.
    """
    if not OpenAI:
        print("[GenReward] openai not installed, text reward unavailable")
        return 0.0, None
    try:
        user_msg = _build_user_message_text_reward(sample_id, task_prompt, gt_image_path, answer_content)
        messages = [
            {"role": "system", "content": TEXT_REWARD_SYSTEM_PROMPT},
            user_msg,
        ]
        base_url = (api_base or "").strip()
        if base_url and not base_url.endswith("/"):
            base_url = base_url + "/"
        client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=WORLDGEN_LLM_TIMEOUT)
        max_try = int(os.environ.get("GEN_REWARD_MAX_TRY", str(WORLDGEN_LLM_MAX_TRY)))
        last_exc: Optional[BaseException] = None
        for attempt in range(1, max_try + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=WORLDGEN_LLM_TEMPERATURE,
                    max_tokens=WORLDGEN_LLM_MAX_TOKENS,
                    timeout=WORLDGEN_LLM_TIMEOUT,
                )
                msg = getattr(resp, "choices", [None])[0]
                if msg is not None:
                    msg = getattr(msg, "message", None)
                content = getattr(msg, "content", None) if msg else None
                if not isinstance(content, str):
                    raise RuntimeError("empty_content")
                obj = _parse_llm_json_reward(content)
                if not isinstance(obj, dict):
                    raise ValueError("LLM output is not a JSON object.")
                score = _parse_text_reward_score(obj)
                print(f"[GenReward] Text reward score: {score} (raw: {obj.get('score')}) rationale: {str(obj.get('rationale', ''))[:120]}...")
                return float(score), obj
            except RateLimitError as e:
                last_exc = e
                if attempt % 5 == 0:
                    print(f"[GenReward] [429 RETRY] text reward attempt={attempt}/{max_try}")
                time.sleep(min(10.0, 0.5 * (2 ** (attempt - 1))) * (0.8 + 0.4 * random.random()))
            except Exception as e:
                last_exc = e
                print(f"[GenReward] [LLM ERROR] text reward attempt={attempt}/{max_try} err={str(e)[:150]}")
                if attempt >= max_try:
                    break
                time.sleep(min(10.0, 0.5 * (2 ** (attempt - 1))) * (0.8 + 0.4 * random.random()))
        if last_exc:
            print(f"[GenReward] Text reward failed: {last_exc}")
        return 0.0, None
    except Exception as e:
        print(f"[GenReward] Text reward failed: {e}")
        traceback.print_exc()
        return 0.0, None


_GEN_API_SEMAPHORE: Optional[BoundedSemaphore] = None
_GEN_API_SEMAPHORE_SIZE: Optional[int] = None
_GEN_API_SEMAPHORE_LOCK = Lock()
_GEN_QUEUE_STATE_LOCK = Lock()
_GEN_WAITING_COUNT = 0
_GEN_INFLIGHT_COUNT = 0
_GEN_BLOCKED_COUNT = 0
_GEN_TOTAL_COUNT = 0


def _shorten_text(text: str, max_len: int = 100) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


def _format_prediction_as_answer_text(prediction: Dict) -> str:
    """Format prediction (gen_prompt + reference_images) into answer text for text reward."""
    if not prediction or not isinstance(prediction, dict):
        return ""
    parts = []
    gen_prompt = prediction.get("gen_prompt", "")
    if gen_prompt:
        parts.append('"gen_prompt": ' + json.dumps(gen_prompt, ensure_ascii=False))
    refs = prediction.get("reference_images", [])
    if refs:
        ref_list = []
        for r in refs:
            if isinstance(r, dict):
                ref_list.append({
                    "img_id": r.get("img_id", ""),
                    "title": r.get("title", ""),
                    "note": r.get("note", ""),
                    "page_url": r.get("page_url", ""),
                })
            else:
                ref_list.append(str(r))
        parts.append('"reference_images": ' + json.dumps(ref_list, ensure_ascii=False, indent=2))
    return "\n".join(parts) if parts else ""


# Qwen generation uses up to the first 3 reference images, aligned with gen_image_from_results eval
MAX_REF_IMAGES_FOR_QWEN = 3

# Qwen multi-URL concurrency control: limit per URL (default 8)
_QWEN_URL_LOCK = Lock()
_QWEN_URL_LIMIT = 8
_QWEN_URL_SEMAPHORES: Dict[str, BoundedSemaphore] = {}


def get_qwen_edit_app_url_list() -> List[str]:
    raw = os.environ.get("QWEN_EDIT_APP_URL", "").strip()
    if not raw:
        print("[GenReward] WARNING: QWEN_EDIT_APP_URL not set")
        return []
    try:
        urls = json.loads(raw)
        if isinstance(urls, list):
            print(f"[GenReward] Parsed {len(urls)} QWEN_EDIT_APP_URL entries")
            return [u.strip() for u in urls if u.strip()]
    except:
        pass
    urls = [u.strip() for u in raw.split(",") if u.strip()]
    print(f"[GenReward] Parsed {len(urls)} QWEN_EDIT_APP_URL entries (comma-separated)")
    return urls


def _acquire_qwen_url_slot(urls: List[str]) -> tuple[str, BoundedSemaphore]:
    """Pick a Qwen Edit service URL with available capacity and acquire one concurrency slot.

    Strategy:
    - Shuffle URLs and prefer one with a free slot (non-blocking acquire).
    - If all URLs are full, block on one URL until a slot becomes available.
    """
    global _QWEN_URL_SEMAPHORES
    shuffled = list(urls)
    random.shuffle(shuffled)
    chosen_url: str
    sem: BoundedSemaphore
    with _QWEN_URL_LOCK:
        # Initialize semaphore per URL
        for u in shuffled:
            if u not in _QWEN_URL_SEMAPHORES:
                _QWEN_URL_SEMAPHORES[u] = BoundedSemaphore(_QWEN_URL_LIMIT)
        # Prefer non-blocking acquire
        for u in shuffled:
            s = _QWEN_URL_SEMAPHORES[u]
            if s.acquire(blocking=False):
                return u, s
        # All full: pick the first URL and wait for a slot
        chosen_url = shuffled[0]
        sem = _QWEN_URL_SEMAPHORES[chosen_url]
    # Block outside the lock to avoid blocking other threads
    sem.acquire()
    return chosen_url, sem


def _get_int_env(name: str, default: int, min_value: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"[GenReward] Invalid {name}={raw!r}, fallback to {default}")
        return default
    if value < min_value:
        print(f"[GenReward] Invalid {name}={value}, fallback to {default}")
        return default
    return value


def get_gen_api_semaphore() -> BoundedSemaphore:
    global _GEN_API_SEMAPHORE, _GEN_API_SEMAPHORE_SIZE
    size = _get_int_env("GEN_API_CONCURRENCY", 20, min_value=1)
    with _GEN_API_SEMAPHORE_LOCK:
        if _GEN_API_SEMAPHORE is None or _GEN_API_SEMAPHORE_SIZE != size:
            _GEN_API_SEMAPHORE = BoundedSemaphore(size)
            _GEN_API_SEMAPHORE_SIZE = size
            print(f"[GenReward] GEN_API_CONCURRENCY={size}")
    return _GEN_API_SEMAPHORE


def call_qwen_edit_to_generate_image(
    prompt: str,
    reference_images: List[str],
    timeout: int = 1800,
    request_tag: str = "",
) -> Optional[str]:
    global _GEN_WAITING_COUNT, _GEN_INFLIGHT_COUNT
    if not requests:
        print("[GenReward] requests not installed")
        return None
    urls = get_qwen_edit_app_url_list()
    if not urls:
        print("[GenReward] No QWEN_EDIT_APP_URL configured")
        return None
    # Pick a URL with available capacity; per-URL concurrency is capped by _QWEN_URL_LIMIT
    base_url, url_sem = _acquire_qwen_url_slot(urls)
    base_url = base_url.rstrip("/")
    # Aligned with QWenLocalGenerator: request /generate
    path = (os.environ.get("QWEN_EDIT_APP_PATH") or "").strip() or "/generate"
    if not path.startswith("/"):
        path = "/" + path
    url = base_url + path
    print(f"[GenReward] Selected endpoint: {url}")
    try:
        ref_images_b64: List[str] = []
        ref_list = reference_images[:MAX_REF_IMAGES_FOR_QWEN]
        for img_path in ref_list:
            if not img_path or not os.path.exists(img_path):
                print(f"[GenReward] Image not found: {img_path}")
                continue
            try:
                with open(img_path, "rb") as f:
                    img_data = f.read()
                b64 = base64.b64encode(img_data).decode("utf-8")
                ref_images_b64.append(b64)
                print(f"[GenReward] Loaded ref image: {img_path} ({len(img_data)} bytes)")
            except Exception as e:
                print(f"[GenReward] Failed to load {img_path}: {e}")
        print(f"[GenReward] Total {len(ref_images_b64)} reference images (Qwen max {MAX_REF_IMAGES_FOR_QWEN})")
        # Aligned with QWenLocalGenerator.gen_payload: image_urls uses data:image/jpeg;base64,xxx; other params are matched
        image_urls = [f"data:image/jpeg;base64,{b}" for b in ref_images_b64] if ref_images_b64 else None
        payload = {
            "image_urls": image_urls,
            "prompt": prompt,
            "seed": 0,
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 40,
            "guidance_scale": 1.0,
            "num_images_per_prompt": 1,
        }
        print(f"[GenReward] Calling Qwen Edit service...")
        print(f"[GenReward] Prompt preview: {_shorten_text(prompt, 150)}")
        semaphore = get_gen_api_semaphore()
        capacity = _GEN_API_SEMAPHORE_SIZE or _get_int_env("GEN_API_CONCURRENCY", 20, min_value=1)
        req = request_tag or "unknown"

        wait_start = time.perf_counter()
        with _GEN_QUEUE_STATE_LOCK:
            _GEN_WAITING_COUNT += 1
            waiting_now = _GEN_WAITING_COUNT
            inflight_now = _GEN_INFLIGHT_COUNT
        print(
            f"[GenReward][Queue] request={req} enqueue "
            f"waiting={waiting_now} inflight={inflight_now} capacity={capacity}"
        )

        semaphore.acquire()
        wait_elapsed = time.perf_counter() - wait_start
        with _GEN_QUEUE_STATE_LOCK:
            _GEN_WAITING_COUNT -= 1
            _GEN_INFLIGHT_COUNT += 1
            waiting_now = _GEN_WAITING_COUNT
            inflight_now = _GEN_INFLIGHT_COUNT
        print(
            f"[GenReward][Queue] request={req} dequeue "
            f"waiting={waiting_now} inflight={inflight_now} capacity={capacity} "
            f"wait_s={wait_elapsed:.3f}"
        )

        gen_start = time.perf_counter()
        try:
            print(f"[GenReward] Acquired generation queue slot, calling API...")
            response = requests.post(url, json=payload, timeout=timeout)
        finally:
            gen_elapsed = time.perf_counter() - gen_start
            total_elapsed = wait_elapsed + gen_elapsed
            with _GEN_QUEUE_STATE_LOCK:
                _GEN_INFLIGHT_COUNT -= 1
                waiting_now = _GEN_WAITING_COUNT
                inflight_now = _GEN_INFLIGHT_COUNT
            semaphore.release()
            # Release per-URL concurrency slot
            try:
                url_sem.release()
            except Exception:
                pass
            print(
                f"[GenReward][Queue] request={req} release "
                f"waiting={waiting_now} inflight={inflight_now} capacity={capacity} "
                f"gen_s={gen_elapsed:.3f} total_s={total_elapsed:.3f}"
            )
        print(f"[GenReward] Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"[GenReward] Error: {response.text[:500]}")
            return None
        result = response.json()
        # Aligned with QWenLocalGenerator.deal_response: success + image
        if not result.get("success"):
            print(f"[GenReward] API success=False: {result.get('message', result)}")
            return None
        img_b64 = result.get("image") or ""
        if not img_b64:
            print(f"[GenReward] No image in response: {result}")
            return None
        if img_b64.startswith("data:image"):
            img_b64 = img_b64.split(",", 1)[-1]
        print(f"[GenReward] Generated image size: {len(img_b64)} chars")
        return img_b64
    except Exception as e:
        print(f"[GenReward] Qwen Edit call failed: {e}")
        traceback.print_exc()
        return None


def _image_path_to_base64_jpeg(path: str) -> Optional[str]:
    """Convert an image file to JPEG base64 for Nano API usage."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            raw = f.read()
        if Image is not None and ImageOps is not None:
            im = Image.open(io.BytesIO(raw))
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=95)
            raw = buf.getvalue()
        return base64.b64encode(raw).decode("utf-8")
    except Exception:
        return None


def _extract_inline_image_b64_from_gemini_generate_content(data: Dict[str, Any]) -> Optional[str]:
    """Parse Gemini `generateContent` JSON; return base64 image bytes from the first inlineData part."""
    if not isinstance(data, dict):
        return None
    for cand in data.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        content = cand.get("content") or {}
        if not isinstance(content, dict):
            continue
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict):
                b64 = inline.get("data")
                if b64:
                    return str(b64)
    return None


def call_nano_to_generate_image(
    prompt: str,
    reference_images: List[str],
    timeout: int = 1200,
    request_tag: str = "",
    api_key: str = "",
    model_name: str = "gemini-3-pro-image-preview",
    max_poll_seconds: int = 180,
    poll_interval: float = 1.0,
    max_try: int = 100,
) -> Optional[str]:
    """
    Google Generative Language API: POST .../models/{model}:generateContent (same payload shape as Gemini REST).
    Returns base64-encoded image from the response inlineData. Retries up to max_try within timeout.
    """
    global _GEN_WAITING_COUNT, _GEN_INFLIGHT_COUNT
    if not requests or not api_key:
        print("[GenReward] Nano: requests not installed or GEN_IMAGE_NANO_API_KEY not set")
        return None
    # Official endpoint (no env): https://ai.google.dev/api/rest/v1beta/models.generateContent
    usage_app_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    )
    parts = [{"text": prompt}]
    for img_path in reference_images:
        b64 = _image_path_to_base64_jpeg(img_path)
        if b64:
            parts.append({"inlineData": {"mimeType": "image/jpeg", "data": b64}})
    payload = json.dumps({"contents": [{"parts": parts}]}, ensure_ascii=False)
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    semaphore = get_gen_api_semaphore()
    req = request_tag or "nano"
    wait_start = time.perf_counter()
    with _GEN_QUEUE_STATE_LOCK:
        _GEN_WAITING_COUNT += 1
    semaphore.acquire()
    wait_elapsed = time.perf_counter() - wait_start
    with _GEN_QUEUE_STATE_LOCK:
        _GEN_WAITING_COUNT -= 1
        _GEN_INFLIGHT_COUNT += 1
    print(f"[GenReward][Queue] request={req} Nano dequeue wait_s={wait_elapsed:.3f}")

    try:
        total_deadline = time.time() + timeout
        req_timeout = min(timeout, 120)
        last_err: Optional[Exception] = None
        for attempt in range(max_try):
            if time.time() >= total_deadline:
                print(f"[GenReward] Nano total timeout ({timeout}s) reached after {attempt} attempts")
                return None
            try:
                r = requests.post(usage_app_url, data=payload, headers=headers, timeout=req_timeout)
                if r is None or r.status_code != 200:
                    raise RuntimeError(
                        f"Nano generateContent failed: {r.status_code if r else 'NoResponse'} "
                        f"{getattr(r, 'text', '')[:500]}"
                    )
                data = r.json()
                if isinstance(data, dict) and data.get("error"):
                    raise RuntimeError(f"Nano API error: {data.get('error')}")
                img_b64 = _extract_inline_image_b64_from_gemini_generate_content(data)
                if not img_b64:
                    raise RuntimeError(f"Nano: no inline image in response: {str(data)[:500]}")
                print(f"[GenReward] Nano generated image size: {len(img_b64)} chars (attempt {attempt + 1})")
                return img_b64
            except Exception as e:
                last_err = e
                if (attempt + 1) % 10 == 0:
                    print(f"[GenReward] Nano attempt {attempt + 1}/{max_try} failed: {e}", flush=True)
                if attempt < max_try - 1 and time.time() < total_deadline - 5:
                    # Original: sleep_s = min(30.0, 3.0 + 3.0 * 1.1**attempt)
                    # Now: shorten by 5x and cap max sleep from 30s to 20s
                    base_sleep = 3.0 + 3.0 * (1.1 ** attempt)
                    sleep_s = min(20.0, base_sleep / 5.0)
                    time.sleep(sleep_s)
        if last_err:
            print(f"[GenReward] Nano all {max_try} attempts failed: {last_err}")
        return None
    finally:
        with _GEN_QUEUE_STATE_LOCK:
            _GEN_INFLIGHT_COUNT -= 1
        semaphore.release()
        print(f"[GenReward][Queue] request={req} Nano release")


def _safe_sample_id(s: Optional[str]) -> str:
    """Sanitize sample_id for use in path (no slashes, no empty)."""
    if not s or not str(s).strip():
        return "unknown"
    s = re.sub(r'[^\w\-.]', "_", str(s).strip())
    return s[:200] if len(s) > 200 else s


def _sanitize_messages_for_save(messages: List[dict]) -> List[dict]:
    """Strip base64 from messages so saved JSON does not contain raw image data."""
    out: List[dict] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        content = m.get("content")
        if isinstance(content, str):
            if "data:image" in content or (len(content) > 500 and "base64" in content.lower()):
                content = content[:200] + " ... [base64/image omitted for save]"
            out.append({"role": role, "content": content})
        elif isinstance(content, list):
            sanitized_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "image_url" and "image_url" in part:
                        sanitized_parts.append({"type": "image_ref", "placeholder": True, "note": "base64 omitted"})
                    elif part.get("type") == "text" and "text" in part:
                        sanitized_parts.append(part)
                    else:
                        sanitized_parts.append(part)
                else:
                    sanitized_parts.append(part)
            out.append({"role": role, "content": sanitized_parts})
        else:
            out.append({"role": role, "content": content})
    return out


def save_trajectory_result(
    trajectory_id: str,
    question: str,
    messages: List[dict],
    prediction: Dict,
    reward: float,
    generated_image_path: Optional[str],
    termination: str = "",
    sample_id: Optional[str] = None,
    token_usage: Optional[Dict] = None,
    timing: Optional[Dict] = None,
    reward_response: Optional[Dict] = None,
    global_step: Optional[int] = None,
    image_reward: Optional[float] = None,
    text_reward: Optional[float] = None,
    text_reward_response: Optional[Dict] = None,
) -> None:
    """
    Save per-trajectory result JSON (each turn's messages + prediction without verified_facts + reward).
    No base64 in saved messages. Written to GEN_IMAGE_OUTPUT_DIR/step{N}/{sample_id}/[PREFIX]_{trajectory_id}/result.json.
    PREFIX: [TOOLONG] for max_response_length_exceeded, [ERROR] for error, [UNKNOWN] for unknown, otherwise empty.
    When text reward is used (image_reward/text_reward/text_reward_response provided), the JSON also records
    image_reward, text_reward, text_reward_response, and the final reward as a weighted total.
    """
    try:
        output_dir = os.environ.get("GEN_IMAGE_OUTPUT_DIR", "./gen_images")
        sid = _safe_sample_id(sample_id or trajectory_id)
        
        # Determine step folder name
        step_folder = f"step{global_step}" if global_step is not None else "step0"
        
        # Determine folder prefix by termination
        prefix = ""
        if termination == "max_response_length_exceeded":
            prefix = "[TOOLONG]_"
        elif termination == "repeated_response":
            prefix = "[REPEAT]_"
        elif termination == "error":
            prefix = "[ERROR]_"
        elif termination == "unknown":
            prefix = "[UNKNOWN]_"
        
        traj_folder_name = f"{prefix}{trajectory_id}"
        # Path: step{N}/{sample_id}/{trajectory_folder}
        traj_dir = Path(output_dir) / step_folder / sid / traj_folder_name
        traj_dir.mkdir(parents=True, exist_ok=True)
        prediction_out = {
            "gen_prompt": prediction.get("gen_prompt", ""),
            "reference_images": prediction.get("reference_images", []),
        }
        obj = {
            "sample_id": sid,
            "trajectory_id": trajectory_id,
            "question": question,
            "messages": _sanitize_messages_for_save(messages),
            "prediction": prediction_out,
            "reward": reward,
            "reward_response": reward_response or {},
            "generated_image_path": generated_image_path,
            "termination": termination,
            "token_usage": token_usage or {},
            "timing": timing or {},
        }
        if image_reward is not None:
            obj["image_reward"] = image_reward
        if text_reward is not None:
            obj["text_reward"] = text_reward
        if text_reward_response is not None:
            obj["text_reward_response"] = text_reward_response
        path = traj_dir / "result.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print(f"[GenReward] Saved result JSON: {path}")
    except Exception as e:
        print(f"[GenReward] Failed to save result.json: {e}")
        traceback.print_exc()


def save_generated_image(
    image_b64: str,
    trajectory_id: str,
    question: str,
    sample_id: Optional[str] = None,
    termination: str = "",
    global_step: Optional[int] = None,
) -> Optional[str]:
    try:
        output_dir = os.environ.get("GEN_IMAGE_OUTPUT_DIR", "./gen_images")
        sid = _safe_sample_id(sample_id or trajectory_id)
        
        # Determine step folder name
        step_folder = f"step{global_step}" if global_step is not None else "step0"
        
        # Determine folder prefix by termination
        prefix = ""
        if termination == "max_response_length_exceeded":
            prefix = "[TOOLONG]_"
        elif termination == "repeated_response":
            prefix = "[REPEAT]_"
        elif termination == "error":
            prefix = "[ERROR]_"
        elif termination == "unknown":
            prefix = "[UNKNOWN]_"
        
        traj_folder_name = f"{prefix}{trajectory_id}"
        # Path: step{N}/{sample_id}/{trajectory_folder}
        traj_dir = Path(output_dir) / step_folder / sid / traj_folder_name
        traj_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_question = "".join(c if c.isalnum() or c in " _-" else "_" for c in question[:50])
        filename = f"gen_{timestamp}_{safe_question}.png"
        filepath = traj_dir / filename
        img_data = base64.b64decode(image_b64)
        with open(filepath, "wb") as f:
            f.write(img_data)
        print(f"[GenReward] Saved image: {filepath} ({len(img_data)} bytes)")
        return str(filepath)
    except Exception as e:
        print(f"[GenReward] Failed to save image: {e}")
        traceback.print_exc()
        return None


class GenImageRewardFn:
    def __init__(self):
        self.timeout = int(os.environ.get("GEN_IMAGE_TIMEOUT", "1800"))
        self.min_input_images = _get_int_env("GEN_MIN_INPUT_IMAGES", 1, min_value=1)
        self.max_input_images = _get_int_env("GEN_MAX_INPUT_IMAGES", 4, min_value=1)
        if self.max_input_images < self.min_input_images:
            print(
                f"[GenReward] Invalid image range: min={self.min_input_images}, max={self.max_input_images}. "
                f"Use max=min={self.min_input_images}"
            )
            self.max_input_images = self.min_input_images
        self.gpt_api_key = os.environ.get("GEN_REWARD_API_KEY", "").strip()
        self.gpt_api_base = os.environ.get("GEN_REWARD_API_BASE_URL", "").strip()
        self.gpt_model = (os.environ.get("GEN_REWARD_MODEL", "gpt-4.1") or "gpt-4.1").strip()
        self.gen_service = (os.environ.get("GEN_IMAGE_SERVICE", "qwen_image") or "qwen_image").strip().lower()
        self.nano_api_key = os.environ.get("GEN_IMAGE_NANO_API_KEY", "").strip()
        self.nano_model = (os.environ.get("GEN_IMAGE_NANO_MODEL", "gemini-3-pro-image-preview") or "gemini-3-pro-image-preview").strip()
        self.nano_timeout = int(os.environ.get("GEN_IMAGE_NANO_TIMEOUT", "1200"))
        self.nano_max_try = int(os.environ.get("GEN_IMAGE_NANO_MAX_TRY", "30"))
        self.nano_max_poll = int(os.environ.get("GEN_IMAGE_NANO_MAX_POLL", "180"))
        # Text reward coefficient:
        # final reward = (1 - text_coef) * image_reward + text_coef * text_reward;
        # 0 means do not call text reward
        _text_coef = os.environ.get("GEN_REWARD_TEXT_COEF", "0").strip()
        try:
            self.text_coef = max(0.0, min(1.0, float(_text_coef)))
        except (TypeError, ValueError):
            self.text_coef = 0.0
        print(
            f"[GenReward] Initialized with timeout={self.timeout}s, scorer=GPT-4.1 worldgen (model={self.gpt_model}), "
            f"gen_service={self.gen_service}, input_images=[{self.min_input_images},{self.max_input_images}], text_reward_coef={self.text_coef}"
        )
    
    def compute_reward(self, episode: Episode, trajectory_id: str) -> float:
        try:
            global _GEN_TOTAL_COUNT, _GEN_BLOCKED_COUNT
            print(f"\n[GenReward] ===== Computing Reward =====")
            print(f"[GenReward] Trajectory ID: {trajectory_id}")
            
            # Get dataset sample from episode.task
            task_data = episode.task if isinstance(episode.task, dict) else {}
            sample_id = task_data.get("id") or task_data.get("sample_id") or getattr(episode, "id", "") or trajectory_id
            messages = (episode.info or {}).get("messages", [])
            token_usage = (episode.info or {}).get("token_usage", {})
            timing = dict((episode.info or {}).get("timing", {}) or {})
            timing.setdefault("model_call_durations", [])
            timing.setdefault("tool_call_durations", [])
            timing.setdefault("total_model_time", 0.0)
            timing.setdefault("total_tool_time", 0.0)
            timing.setdefault("trajectory_total_time", 0.0)
            timing.setdefault("image_gen_time", 0.0)
            timing.setdefault("score_time", 0.0)
            
            # Get global training step (used to organize output folders)
            global_step = (episode.info or {}).get("global_step")
            termination = (episode.info or {}).get("termination", "")

            def _finalize_timing() -> Dict:
                model_total = float(timing.get("total_model_time", 0.0) or 0.0)
                tool_total = float(timing.get("total_tool_time", 0.0) or 0.0)
                image_gen_time = float(timing.get("image_gen_time", 0.0) or 0.0)
                score_time = float(timing.get("score_time", 0.0) or 0.0)
                timing["model_call_durations"] = [round(float(x), 1) for x in timing.get("model_call_durations", [])]
                timing["tool_call_durations"] = [round(float(x), 1) for x in timing.get("tool_call_durations", [])]
                timing["total_model_time"] = round(model_total, 1)
                timing["total_tool_time"] = round(tool_total, 1)
                timing["trajectory_total_time"] = round(float(timing.get("trajectory_total_time", 0.0) or 0.0), 1)
                timing["image_gen_time"] = round(image_gen_time, 1)
                timing["score_time"] = round(score_time, 1)
                timing["timing_components_sum"] = round(model_total + tool_total + image_gen_time + score_time, 1)
                if episode.info is not None:
                    episode.info["timing"] = timing
                return timing

            with _GEN_QUEUE_STATE_LOCK:
                _GEN_TOTAL_COUNT += 1
                total_now = _GEN_TOTAL_COUNT
            
            # Get ground-truth image path
            gt_image_path = task_data.get("gt_image", "")
            if not gt_image_path:
                print(f"[GenReward] ERROR: No gt_image in task data")
                print(f"[GenReward] Task data keys: {list(task_data.keys())}")
                save_trajectory_result(
                    trajectory_id, "", messages, task_data.get("prediction", {}), 0.0, None,
                    (episode.info or {}).get("termination", ""), sample_id=sample_id, token_usage=token_usage, timing=_finalize_timing()
                )
                return 0.0
            
            if not os.path.exists(gt_image_path):
                print(f"[GenReward] ERROR: Ground truth image not found: {gt_image_path}")
                save_trajectory_result(
                    trajectory_id, task_data.get("question", ""), messages, task_data.get("prediction", {}), 0.0, None,
                    (episode.info or {}).get("termination", ""), sample_id=sample_id, token_usage=token_usage, timing=_finalize_timing()
                )
                return 0.0
            
            print(f"[GenReward] Ground truth image: {gt_image_path}")
            
            # Get original prompt
            original_prompt = task_data.get("question", "")
            if not original_prompt:
                print(f"[GenReward] ERROR: No question/prompt in task data")
                save_trajectory_result(
                    trajectory_id, "", messages, task_data.get("prediction", {}), 0.0, None,
                    (episode.info or {}).get("termination", ""), sample_id=sample_id, token_usage=token_usage, timing=_finalize_timing()
                )
                return 0.0
            
            print(f"[GenReward] Original prompt: {_shorten_text(original_prompt, 150)}")
            
            # Get generated prompt and reference images from prediction
            prediction = task_data.get("prediction", {})
            if not prediction or not isinstance(prediction, dict):
                print(f"[GenReward] Invalid prediction: {prediction}")
                save_trajectory_result(
                    trajectory_id, original_prompt, messages, {"error": str(prediction)}, 0.0, None,
                    termination, sample_id=sample_id, token_usage=token_usage, timing=_finalize_timing(), global_step=global_step
                )
                return 0.0
            if "error" in prediction:
                print(f"[GenReward] Prediction error: {prediction['error']}")
                save_trajectory_result(
                    trajectory_id, original_prompt, messages, prediction, 0.0, None,
                    termination, sample_id=sample_id, token_usage=token_usage, timing=_finalize_timing(), global_step=global_step
                )
                return 0.0
            
            gen_prompt = prediction.get("gen_prompt", "")
            reference_images = prediction.get("reference_images", [])
            if not gen_prompt:
                print(f"[GenReward] Empty gen_prompt")
                save_trajectory_result(
                    trajectory_id, original_prompt, messages, prediction, 0.0, None,
                    termination, sample_id=sample_id, token_usage=token_usage, timing=_finalize_timing(), global_step=global_step
                )
                return 0.0
            
            print(f"[GenReward] gen_prompt: {_shorten_text(gen_prompt, 150)}")
            print(f"[GenReward] reference_images count: {len(reference_images)}")
            
            ref_image_paths: List[str] = []
            for r in reference_images:
                if isinstance(r, dict):
                    local_path = r.get("local_path", "")
                    if local_path and os.path.exists(local_path):
                        ref_image_paths.append(local_path)
            print(f"[GenReward] Valid ref images: {len(ref_image_paths)}")
            if len(ref_image_paths) < self.min_input_images or len(ref_image_paths) > self.max_input_images:
                with _GEN_QUEUE_STATE_LOCK:
                    _GEN_BLOCKED_COUNT += 1
                    blocked_now = _GEN_BLOCKED_COUNT
                print(
                    f"[GenReward] Invalid ref image count: {len(ref_image_paths)} "
                    f"(required [{self.min_input_images}, {self.max_input_images}]), skip generation and return 0.0"
                )
                print(
                    f"[GenReward][Blocked] sample_id={sample_id} trajectory_id={trajectory_id} "
                    f"blocked={blocked_now} total={total_now}"
                )
                save_trajectory_result(
                    trajectory_id, original_prompt, messages, prediction, 0.0, None,
                    termination, sample_id=sample_id, token_usage=token_usage, timing=_finalize_timing(), global_step=global_step
                )
                return 0.0
            
            image_gen_start = time.perf_counter()
            if self.gen_service == "nano":
                print(f"[GenReward] Step 1: Call Nano to generate image")
                generated_img_b64 = call_nano_to_generate_image(
                    gen_prompt,
                    ref_image_paths,
                    timeout=min(self.timeout, self.nano_timeout),
                    request_tag=f"{sample_id}:{trajectory_id}",
                    api_key=self.nano_api_key,
                    model_name=self.nano_model,
                    max_poll_seconds=self.nano_max_poll,
                    max_try=self.nano_max_try,
                )
            else:
                print(f"[GenReward] Step 1: Call Qwen Edit to generate image (up to {MAX_REF_IMAGES_FOR_QWEN} reference images)")
                generated_img_b64 = call_qwen_edit_to_generate_image(
                    gen_prompt,
                    ref_image_paths,
                    self.timeout,
                    request_tag=f"{sample_id}:{trajectory_id}",
                )
            timing["image_gen_time"] = time.perf_counter() - image_gen_start
            if not generated_img_b64:
                print(f"[GenReward] Image generation failed")
                save_trajectory_result(
                    trajectory_id, original_prompt, messages, prediction, 0.0, None,
                    termination, sample_id=sample_id, token_usage=token_usage, timing=_finalize_timing(), global_step=global_step
                )
                return 0.0
            
            print(f"[GenReward] Step 2: Save generated image")
            saved_path = save_generated_image(generated_img_b64, trajectory_id, original_prompt, sample_id=sample_id, termination=termination, global_step=global_step)
            if not saved_path:
                print(f"[GenReward] Failed to save image")
                save_trajectory_result(
                    trajectory_id, original_prompt, messages, prediction, 0.0, None,
                    termination, sample_id=sample_id, token_usage=token_usage, timing=_finalize_timing(), global_step=global_step
                )
                return 0.0
            
            print(f"[GenReward] Step 3: Score with GPT-4.1 KnowGen (original prompt; same prompt/formula as gpt_eval_knowgen)")
            if not self.gpt_api_key:
                print(f"[GenReward] GEN_REWARD_API_KEY not set, returning 0.0")
                timing["score_time"] = 0.0
                save_trajectory_result(
                    trajectory_id, original_prompt, messages, prediction, 0.0, saved_path,
                    termination, sample_id=sample_id, token_usage=token_usage, timing=_finalize_timing(), global_step=global_step
                )
                return 0.0
            score_start = time.perf_counter()
            image_reward, reward_obj = call_gpt41_worldgen_score(
                saved_path,
                gt_image_path,
                original_prompt,
                str(sample_id),
                self.gpt_api_key,
                self.gpt_api_base,
                self.gpt_model,
            )
            timing["score_time"] = time.perf_counter() - score_start
            reward_response_out = {"worldgen": reward_obj or {}}

            # Optional: text-based reward that does not depend on generated image quality
            text_reward = 0.0
            text_reward_obj = None
            if self.text_coef > 0 and self.gpt_api_key:
                answer_content = _format_prediction_as_answer_text(prediction)
                text_reward_start = time.perf_counter()
                text_reward, text_reward_obj = call_text_reward_score(
                    original_prompt,
                    gt_image_path,
                    answer_content,
                    str(sample_id),
                    self.gpt_api_key,
                    self.gpt_api_base,
                    self.gpt_model,
                )
                timing["text_reward_time"] = round(time.perf_counter() - text_reward_start, 1)
                reward_response_out["text_reward"] = text_reward_obj or {}
                reward_response_out["image_reward"] = image_reward
                reward_response_out["text_reward_score"] = text_reward
                final_reward = (1.0 - self.text_coef) * image_reward + self.text_coef * text_reward
                print(f"[GenReward] Image reward (worldgen): {image_reward:.3f}")
                print(f"[GenReward] Text reward: {text_reward:.3f}")
                print(f"[GenReward] Blended (coef={self.text_coef}): final={final_reward:.3f}")
                if episode.info is None:
                    episode.info = {}
                episode.info["image_reward"] = image_reward
                episode.info["text_reward"] = text_reward
                episode.info["reward_weighted"] = final_reward
            else:
                final_reward = image_reward

            print(f"[GenReward] ===== Final Reward: {final_reward:.3f} =====\n")
            # Save this trajectory's result.json; when text_coef>0 also record image_reward, text_reward,
            # text_reward_response, and the weighted final reward
            save_kwargs = dict(
                trajectory_id=trajectory_id,
                question=original_prompt,
                messages=messages,
                prediction=prediction,
                reward=final_reward,
                generated_image_path=saved_path,
                termination=termination,
                sample_id=sample_id,
                token_usage=token_usage,
                timing=_finalize_timing(),
                reward_response=reward_response_out,
                global_step=global_step,
            )
            if self.text_coef > 0:
                save_kwargs["image_reward"] = image_reward
                save_kwargs["text_reward"] = text_reward
                save_kwargs["text_reward_response"] = text_reward_obj
            save_trajectory_result(**save_kwargs)
            return final_reward
        except Exception as e:
            print(f"[GenReward] Reward computation failed: {e}")
            traceback.print_exc()
            return 0.0


def _compute_reward_one(args: Tuple[Any, str, int, int]) -> Tuple[int, float]:
    """Compute reward for one episode (thread-pool worker). Returns (idx, reward)."""
    episode, trajectory_id, idx, total = args
    reward_fn = GenImageRewardFn()
    reward = reward_fn.compute_reward(episode, trajectory_id)
    return (idx, reward)


async def gen_image_deepresearch_reward_fn_async(episodes: List[Episode], **kwargs) -> List[float]:
    print(f"\n[GenReward] ===== Batch Reward Computation =====")
    print(f"[GenReward] Episodes count: {len(episodes)}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(episodes)
    # Run reward for episodes in parallel (including image generation) to avoid one slow sample blocking the whole batch.
    # Concurrency is limited by the GEN_API_CONCURRENCY semaphore.
    max_workers = min(total, 64)
    rewards: List[float] = [0.0] * total
    args_list = [
        (ep, f"traj_{timestamp}_{idx:04d}", idx, total)
        for idx, ep in enumerate(episodes)
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_compute_reward_one, a): a[2] for a in args_list}
        for fut in as_completed(futures):
            idx, reward = fut.result()
            rewards[idx] = reward
            print(f"[GenReward] Episode {idx + 1}/{total} reward: {reward:.3f}")
    print(f"\n[GenReward] ===== Batch Complete =====")
    print(f"[GenReward] Average reward: {sum(rewards) / len(rewards):.3f}")
    return rewards
