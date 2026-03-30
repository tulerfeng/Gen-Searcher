#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KnowGen evaluation: GPT-4.1 scoring for image-generation results.
Input: results.json (e.g. from run_gen_image_from_results.sh): prompt, output_path (generated image), gt_image (GT).
One LLM call per sample; outputs per-sample scores and category-group summaries.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI, RateLimitError
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------- Tunables -----------------
LLM_TIMEOUT_SEC = 300.0
LLM_MAX_TRY = 20
LLM_MAX_TOKENS = 8192
LLM_TEMPERATURE = 0.0
MAX_SIDE = 4096
JPEG_QUALITY = 100
SCORE_KEYS = ("faithfulness", "visual_correctness", "text_accuracy", "aesthetics")

# Top-level group mapping
CATEGORY_MAP = {
    "science_and_knowledge": {
        "astronomy", "biology", "chemistry", "physics", "engineering",
        "medicine", "industry", "architecture", "history", "geography",
        "religion", "politics", "culture", "art", "sports"
    },
    "pop_culture_and_news": {
        "anime", "game", "film", "celebrities", "posters",
        "multi-subject-anime", "multi-subject-celebrities", "multi-subject-game",
        "news"
    }
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _read_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json(path: str, obj: Any) -> None:
    d = os.path.dirname(path)
    if d:
        _ensure_dir(d)
    tmp_path = f"{path}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _round_01(x: float) -> float:
    return round(_clip01(x), 2)


# ----------------- Image encoding -----------------
def _encode_image_to_data_url(path: str, max_side: int = MAX_SIDE, quality: int = JPEG_QUALITY) -> str:
    import base64
    import io
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        w, h = im.size
        longest = max(w, h)
        if longest > max_side:
            scale = max_side / float(longest)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
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


def _build_user_message(sample_id: str, prompt: str, gen_path: str, gt_path: str) -> dict:
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
            url = _encode_image_to_data_url(p)
            content.append({"type": "image_url", "image_url": {"url": url}})
        except Exception as e:
            content.append({"type": "text", "text": f"\n[WARN] failed to load image: {p}, err={e}\n"})
    return {"role": "user", "content": content}


SYSTEM_PROMPT = r"""You are a strict and professional expert evaluator for AI-generated image grounded with world knowledge (MODEL EVALUATION).

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


def _parse_llm_json(content: str) -> dict:
    """Parse LLM-returned JSON, tolerating extra commas and surrounding text."""
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
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise ValueError("LLM output is not valid JSON.")


def _call_llm_json(client: OpenAI, model: str, messages: List[dict], max_try: int = LLM_MAX_TRY) -> dict:
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_try + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                timeout=LLM_TIMEOUT_SEC,
            )
            msg = getattr(resp, "choices", [None])[0]
            if msg is not None:
                msg = getattr(msg, "message", None)
            content = getattr(msg, "content", None) if msg else None
            if not isinstance(content, str):
                raise RuntimeError("empty_content")
            obj = _parse_llm_json(content)
            if not isinstance(obj, dict):
                raise ValueError("LLM output is not a JSON object.")
            return obj
        except RateLimitError as e:
            last_exc = e
            if attempt % 5 == 0:
                _log(f"[429 RETRY] attempt={attempt}/{max_try}")
            time.sleep(min(10.0, 0.5 * (2 ** (attempt - 1))) * (0.8 + 0.4 * random.random()))
        except Exception as e:
            last_exc = e
            _log(f"[LLM ERROR] attempt={attempt}/{max_try} err={str(e)[:150]}")
            if attempt >= max_try:
                break
            time.sleep(min(10.0, 0.5 * (2 ** (attempt - 1))) * (0.8 + 0.4 * random.random()))
    raise last_exc if last_exc else RuntimeError("LLM call failed")


def _normalize_scores(obj: Any) -> Tuple[float, float, Optional[float], float, str]:
    """Return (faithfulness, visual_correctness, text_accuracy|None, aesthetics, rationale).
    text_accuracy=None means N/A (determined by text_accuracy_na).
    """
    out: Tuple[float, float, Optional[float], float, str] = (0.0, 0.0, 0.0, 0.0, "")
    if not isinstance(obj, dict):
        return out
    f = _round_01(float(obj.get("faithfulness", 0)))
    v = _round_01(float(obj.get("visual_correctness", 0)))
    text_na = obj.get("text_accuracy_na")
    if text_na in (True, "true", "True", 1):
        t: Optional[float] = None
    else:
        try:
            t = _round_01(float(obj.get("text_accuracy", 0)))
        except (TypeError, ValueError):
            t = None
    a = _round_01(float(obj.get("aesthetics", 0)))
    r = str(obj.get("rationale", ""))[:1000]
    return (f, v, t, a, r)


def _resolve_path(path: Optional[str], base_dir: Path) -> Optional[str]:
    if not path or not str(path).strip():
        return None
    p = Path(path)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return str(p) if p.exists() else None


def load_results(results_path: str) -> Tuple[List[dict], Path]:
    data = _read_json(results_path)
    if not isinstance(data, list):
        raise ValueError("results.json must be a JSON array.")
    base_dir = Path(results_path).resolve().parent
    return data, base_dir


def run_one_eval(
    entry: dict,
    base_dir: Path,
    client: OpenAI,
    model: str,
) -> Tuple[str, bool, dict]:
    """Returns (sample_id, ok, result_dict)."""
    sid = str(entry.get("id", ""))
    prompt = (entry.get("prompt") or "").strip()
    out_path = _resolve_path(entry.get("output_path"), base_dir)
    gt_path = _resolve_path(entry.get("gt_image"), base_dir)

    if not prompt:
        return sid, False, {"error": "empty_prompt"}
    if not out_path or not os.path.isfile(out_path):
        return sid, False, {"error": "missing_generated_image"}
    if not gt_path or not os.path.isfile(gt_path):
        return sid, False, {"error": "missing_gt_image"}

    try:
        user_msg = _build_user_message(sid, prompt, out_path, gt_path)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            user_msg,
        ]
        obj = _call_llm_json(client, model, messages)
        f, v, t, a, r = _normalize_scores(obj)
        t_val = 0.5 if t is None else t
        text_accuracy_na = t is None
        overall = round(0.1 * f + 0.4 * v + 0.4 * t_val + 0.1 * a, 2)
        payload = {
            "rationale": r,
            "faithfulness": f,
            "visual_correctness": v,
            "text_accuracy": t_val,
            "aesthetics": a,
            "overall": overall,
        }
        if text_accuracy_na:
            payload["text_accuracy_na"] = True
        return sid, True, payload
    except Exception as e:
        return sid, False, {"error": str(e)[:300]}


def get_category_group(meta: dict) -> str:
    """Return 'science_and_knowledge' or 'pop_culture_and_news'; raise ValueError if no match."""
    cat_raw = (meta.get("category") or "").strip().lower()
    if not cat_raw:
        raise ValueError(f"meta has no category field or it is empty: {meta}")
    for group_name, cats in CATEGORY_MAP.items():
        if cat_raw in cats:
            return group_name
    raise ValueError(f"category '{cat_raw}' does not match the two top-level groups; please check data or extend mapping")


def build_summary_by_groups(rows: List[dict]) -> dict:
    """
    Compute mean scores grouped by the two top-level groups.
    rows: [{"id":..., "meta":{...}, "scores":{...}, "eval_success": bool}, ...]
    Returns: {
        "science_and_knowledge": {"faithfulness": mean, ..., "overall": mean, "count": N},
        "pop_culture_and_news": {...},
        "overall_avg": {...}  # mean of the two group means
    }
    """
    groups = {"science_and_knowledge": [], "pop_culture_and_news": []}
    for r in rows:
        if not r.get("eval_success"):
            continue
        meta = r.get("meta") or {}
        scores = r.get("scores") or {}
        if not isinstance(scores, dict):
            continue
        try:
            grp = get_category_group(meta)
            groups[grp].append(scores)
        except ValueError as e:
            _log(f"[WARN] Skip sample {r.get('id')}: {e}")
            continue

    def avg_scores(slist: List[dict]) -> dict:
        if not slist:
            return {k: 0.0 for k in list(SCORE_KEYS) + ["overall"]}
        out = {}
        for k in ("faithfulness", "visual_correctness", "aesthetics"):
            vals = [float(s.get(k, 0)) for s in slist if isinstance(s.get(k), (int, float))]
            out[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
        text_vals = [float(s.get("text_accuracy", 0)) for s in slist if not s.get("text_accuracy_na") and isinstance(s.get("text_accuracy"), (int, float))]
        out["text_accuracy"] = round(sum(text_vals) / len(text_vals), 4) if text_vals else 0.5
        out["overall"] = round(0.1 * out["faithfulness"] + 0.4 * out["visual_correctness"] + 0.4 * out["text_accuracy"] + 0.1 * out["aesthetics"], 4)
        return out

    summary = {}
    for grp_name, slist in groups.items():
        summary[grp_name] = {**avg_scores(slist), "count": len(slist)}

    # Mean of the two group means
    sk_avg = summary["science_and_knowledge"]
    pc_avg = summary["pop_culture_and_news"]
    overall_avg = {}
    for k in list(SCORE_KEYS) + ["overall"]:
        overall_avg[k] = round((sk_avg[k] + pc_avg[k]) / 2.0, 4)
    overall_avg["count"] = sk_avg["count"] + pc_avg["count"]
    summary["overall_avg"] = overall_avg
    return summary


def build_output_with_summary(data: List[dict], results_by_id: Dict[str, dict]) -> Tuple[List[dict], dict]:
    """Build output list in original order and append 3 summary items at the end."""
    sorted_items: List[dict] = []
    for i, e in enumerate(data):
        sid = str(e.get("id", i))
        rec = results_by_id.get(sid)
        if rec is None:
            rec = dict(e)
            rec["eval_success"] = False
            rec["scores"] = None
            rec["error"] = "not_evaluated"
        sorted_items.append(rec)

    valid_rows = [r for r in sorted_items if r.get("eval_success") is True and isinstance(r.get("scores"), dict)]
    summary = build_summary_by_groups(valid_rows)

    output_items = list(sorted_items)
    output_items.append({
        "summary_type": "science_and_knowledge",
        "avg_scores": {k: summary["science_and_knowledge"][k] for k in list(SCORE_KEYS) + ["overall"]},
        "count": summary["science_and_knowledge"]["count"]
    })
    output_items.append({
        "summary_type": "pop_culture_and_news",
        "avg_scores": {k: summary["pop_culture_and_news"][k] for k in list(SCORE_KEYS) + ["overall"]},
        "count": summary["pop_culture_and_news"]["count"]
    })
    output_items.append({
        "summary_type": "overall_avg",
        "avg_scores": {k: summary["overall_avg"][k] for k in list(SCORE_KEYS) + ["overall"]},
        "count": summary["overall_avg"]["count"]
    })
    return output_items, summary


def main():
    parser = argparse.ArgumentParser(description="KnowGen results eval: prompt + generated image + GT image -> GPT-4.1 scoring")
    parser.add_argument("--results", "-r", required=True, help="Path to results.json produced by run_gen_image_from_results")
    parser.add_argument("--output-json", "-o", default=None, help="Output JSON path (default: <results_dir>/results_eval.json)")
    parser.add_argument("--api-key", required=True, help="OpenAI-compatible API key (or set OPENAI_API_KEY)")
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible base_url (or set OPENAI_API_BASE)")
    parser.add_argument("--model", default="gpt-4.1", help="Model name")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallelism")
    parser.add_argument("--resume", action="store_true", help="Skip ids that already have evaluation results")
    args = parser.parse_args()

    api_key = (args.api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    api_base = (args.api_base or os.getenv("OPENAI_API_BASE", "")).strip()
    if not api_key:
        raise SystemExit("Please provide --api-key or set OPENAI_API_KEY")
    if api_base and not api_base.endswith("/"):
        api_base += "/"

    results_path = Path(args.results).resolve()
    if not results_path.exists():
        raise SystemExit(f"results file not found: {results_path}")
    data, base_dir = load_results(str(results_path))
    eval_path = Path(args.output_json).resolve() if args.output_json else (base_dir / "results_eval.json")

    existing = {}
    if args.resume and eval_path.exists():
        raw = _read_json(str(eval_path))
        if isinstance(raw, dict):
            existing = raw
        elif isinstance(raw, list):
            for r in raw:
                iid = r.get("id")
                if iid is not None:
                    existing[str(iid)] = r

    # Only evaluate samples with success=True and valid output_path + gt_image
    to_run = []
    for entry in data:
        if not entry.get("success"):
            continue
        sid = str(entry.get("id", ""))
        if sid in existing and isinstance((existing[sid] or {}).get("scores"), dict):
            continue
        gt_path = _resolve_path(entry.get("gt_image"), base_dir)
        out_path = _resolve_path(entry.get("output_path"), base_dir)
        if not gt_path or not out_path:
            continue
        to_run.append(entry)

    _log(f"results total={len(data)}, pending_eval={len(to_run)}")

    client = OpenAI(api_key=api_key, base_url=api_base or None, timeout=LLM_TIMEOUT_SEC)
    model = (args.model or "gpt-4.1").strip()
    results_by_id = dict(existing)

    if to_run:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {
                ex.submit(run_one_eval, entry, base_dir, client, model): entry
                for entry in to_run
            }
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Eval"):
                entry = futs[fut]
                try:
                    sid, ok, payload = fut.result()
                    rec = dict(entry)  # Keep all original fields
                    rec["eval_success"] = ok
                    if ok:
                        rec["scores"] = payload
                        rec["error"] = None
                    else:
                        rec["scores"] = None
                        rec["error"] = payload.get("error")
                    results_by_id[sid] = rec
                except Exception as e:
                    sid = str(entry.get("id", ""))
                    rec = dict(entry)
                    rec["eval_success"] = False
                    rec["scores"] = None
                    rec["error"] = str(e)[:200]
                    results_by_id[sid] = rec
                # Recompute summary and save after each sample to keep resume reliable.
                output_items, _ = build_output_with_summary(data, results_by_id)
                _write_json(str(eval_path), output_items)

    output_items, summary = build_output_with_summary(data, results_by_id)
    _write_json(str(eval_path), output_items)
    _log(f"Eval output (with summaries): {eval_path}")
    _log(f"Science & Knowledge: {summary['science_and_knowledge']}")
    _log(f"Pop Culture & News: {summary['pop_culture_and_news']}")
    _log(f"Overall Avg: {summary['overall_avg']}")


if __name__ == "__main__":
    main()
