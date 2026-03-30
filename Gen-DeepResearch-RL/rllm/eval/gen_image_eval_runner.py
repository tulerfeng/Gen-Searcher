"""
Gen Image eval: run the Gen workflow and only produce trajectory logs, gen_prompt, and reference_images.
No image generation and no Gemini-based scoring.
Input: a JSON file, each line is one sample {id, prompt, meta, gen_image}.
Output: results.json (appended after each sample; includes id, trajectory_messages, gen_prompt, reference_images, termination, etc.).
Supports --resume: skip ids that are already completed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout import OpenAIEngine
from vision_deepresearch_async_workflow.gen_image_deepresearch_tools_executor import (
    create_gen_image_tools,
)
from vision_deepresearch_async_workflow.gen_image_deepresearch_workflow import (
    GenImageDeepResearchWorkflow,
)


# ---------------------- Data loading ---------------------- #


def load_tasks_from_json(path: str, max_samples: int | None = None) -> List[dict]:
    """Load JSON input. Supports a JSON array or JSONL (one JSON object per line)."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    tasks: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("JSON root must be an array")
        tasks = data
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))

    if not tasks:
        raise ValueError("No valid tasks in input file")

    for i, t in enumerate(tasks):
        if "prompt" not in t:
            raise ValueError(f"Task {i} missing 'prompt'")
        if "id" not in t:
            t["id"] = i

    if max_samples is not None:
        tasks = tasks[:max_samples]

    return tasks


def task_to_workflow_input(record: dict) -> dict:
    """Convert an input record into the workflow task format (question, etc.)."""
    return {
        "id": record.get("id"),
        "question": record.get("prompt", ""),
        "prompt": record.get("prompt", ""),
        "meta": record.get("meta", {}),
        "gen_image": record.get("gen_image"),  # Not used for now
    }


# ---------------------- Sanitize & output ---------------------- #


def _sanitize_content(msg: dict) -> dict:
    """Sanitize message content by trimming oversized fields."""
    out = {"role": msg.get("role", ""), "content": ""}
    content = msg.get("content", "")
    if isinstance(content, str):
        out["content"] = content[:50000] + "..." if len(content) > 50000 else content
    else:
        out["content"] = str(content)[:50000]
    if "images" in msg:
        out["images"] = [
            (p if isinstance(p, str) else p.get("image", ""))[:200]
            for p in (msg["images"] or [])[:10]
        ]
    return out


def _copy_ref_images_and_build_list(
    prediction: dict,
    sample_id: str,
    ref_images_dir: Path,
) -> List[dict]:
    """Copy images in reference_images into ref_images_dir and return the updated list with new paths."""
    ref_images_dir.mkdir(parents=True, exist_ok=True)
    out = []
    raw_refs = prediction.get("reference_images", []) if isinstance(prediction, dict) else []
    for r in raw_refs:
        if not isinstance(r, dict):
            continue
        local_path = (r.get("local_path") or "").strip()
        img_id = r.get("img_id", "").strip() or "img"
        note = r.get("note", "")
        url = r.get("url", "")
        title = r.get("title", "")
        new_path = ""
        if local_path and os.path.exists(local_path):
            ext = Path(local_path).suffix or ".jpg"
            safe_img_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in img_id)
            dest_name = f"{sample_id}_{safe_img_id}{ext}"
            dest_path = ref_images_dir / dest_name
            try:
                shutil.copy2(local_path, dest_path)
                new_path = str(dest_path)
            except Exception as e:
                print(f"[GenEval] Failed to copy ref image {local_path} -> {dest_path}: {e}")
                new_path = local_path
        else:
            new_path = local_path
        out.append({
            "img_id": img_id,
            "note": note,
            "local_path": new_path,
            "url": url,
            "title": title,
        })
    return out


def episode_to_output_record(
    episode: Any,
    original_task: dict,
    ref_images_dir: Path | None = None,
) -> dict:
    """Extract fields to save from an Episode."""
    info = episode.info or {}
    messages = info.get("messages", [])
    prediction = info.get("prediction", {})
    termination = info.get("termination") or (
        episode.termination_reason.value if hasattr(episode, "termination_reason") and episode.termination_reason else "unknown"
    )

    trajectory_messages = [_sanitize_content(m) for m in messages]
    gen_prompt = prediction.get("gen_prompt", "") if isinstance(prediction, dict) else ""
    sample_id = str(original_task.get("id", "unknown"))

    if ref_images_dir:
        reference_images = _copy_ref_images_and_build_list(prediction, sample_id, ref_images_dir)
    else:
        reference_images = []
        if isinstance(prediction, dict) and "reference_images" in prediction:
            for r in prediction["reference_images"]:
                if isinstance(r, dict):
                    reference_images.append({
                        "img_id": r.get("img_id", ""),
                        "note": r.get("note", ""),
                        "local_path": r.get("local_path", ""),
                        "url": r.get("url", ""),
                        "title": r.get("title", ""),
                    })

    # Open-source data convention: GT image path is stored at top-level task field `gt_image`
    gt_image = original_task.get("gt_image", "")

    return {
        "id": original_task.get("id"),
        "prompt": original_task.get("prompt", ""),
        "meta": original_task.get("meta", {}),
        "termination": termination,
        "trajectory_messages": trajectory_messages,
        "gen_prompt": gen_prompt,
        "reference_images": reference_images,
        "gt_image": gt_image,
    }


# ---------------------- Rollout ---------------------- #


def build_rollout_engine(args: argparse.Namespace) -> OpenAIEngine:
    sampling_params = {
        "temperature": getattr(args, "temperature", 0.7),
        "top_p": getattr(args, "top_p", 1.0),
    }
    max_tokens = getattr(args, "max_tokens", None)
    if max_tokens is not None:
        sampling_params["max_tokens"] = max_tokens
    return OpenAIEngine(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_prompt_length=getattr(args, "max_prompt_length", 32768),
        max_response_length=getattr(args, "max_response_length", 32768),
        sampling_params=sampling_params,
    )


# ---------------------- Main ---------------------- #


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gen Image eval: produce trajectory logs, gen_prompt, reference_images (no generation/scoring)"
    )
    parser.add_argument("--input", "--json", default=None, dest="input_json", help="Input JSON path (array or JSONL)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max number of samples to evaluate")
    parser.add_argument("--model", default=None, help="Model name (inference service)")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible inference service base_url")
    parser.add_argument("--api-key", default=None, help="API Key")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=32768, help="Max prompt length (default 32k)")
    parser.add_argument("--max-response-length", type=int, default=32768, help="Max response length (default 32k)")
    parser.add_argument("--parallel-tasks", type=int, default=4, help="Number of parallel tasks")
    parser.add_argument("--output-dir", default="./gen_eval_outputs", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Skip completed ids and resume from checkpoint")
    return parser.parse_args()


def _append_and_save(results: List[dict], rec: dict, results_path: Path) -> None:
    """Append one result and immediately write back to the JSON file."""
    results.append(rec)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


async def _run_one(
    workflow_engine: AgentWorkflowEngine,
    task: dict,
    task_id: str,
):
    """Run a single task and return the episode."""
    _, _, episode = await workflow_engine.process_task_with_retry(task, task_id, 0)
    return episode


async def main():
    args = parse_args()

    input_path = args.input_json or os.environ.get("GEN_EVAL_INPUT_JSON")
    if not input_path:
        raise ValueError("Please provide --input/--json or set GEN_EVAL_INPUT_JSON")

    model = args.model or os.environ.get("GEN_EVAL_MODEL", "Vision-DeepResearch-8B")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

    args.model = model
    args.base_url = base_url
    args.api_key = api_key

    tasks_raw = load_tasks_from_json(input_path, args.max_samples)
    tasks = [task_to_workflow_input(t) for t in tasks_raw]
    task_ids = [str(t.get("id", i)) for i, t in enumerate(tasks_raw)]
    id_to_task = {str(t.get("id", i)): t for i, t in enumerate(tasks_raw)}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_images_dir = out_dir / "ref_images"
    results_path = out_dir / "results.json"

    results: List[dict] = []
    done_ids: set = set()
    if args.resume and results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            done_ids = {str(r.get("id")) for r in results if r.get("id") is not None}
            print(f"[GenEval] Resume: loaded {len(results)} records, skipping {len(done_ids)} ids")
        except Exception as e:
            print(f"[GenEval] Resume load failed: {e}; starting from scratch")

    pending = [
        (t, tid)
        for t, tid in zip(tasks, task_ids)
        if tid not in done_ids
    ]
    if not pending:
        print("[GenEval] No pending samples; already completed")
        return

    tools = create_gen_image_tools()
    rollout_engine = build_rollout_engine(args)
    workflow_engine = AgentWorkflowEngine(
        workflow_cls=GenImageDeepResearchWorkflow,
        workflow_args={
            "tools": tools,
            "reward_function": None,
        },
        rollout_engine=rollout_engine,
        n_parallel_tasks=args.parallel_tasks,
        retry_limit=2,
    )
    await workflow_engine.initialize_pool()

    write_lock = asyncio.Lock()
    total_count = len(results) + len(pending)
    pbar = tqdm(total=len(pending), desc="GenEval", unit="sample")

    async def run_and_save(task: dict, task_id: str) -> None:
        episode = await _run_one(workflow_engine, task, task_id)
        orig = id_to_task.get(task_id, {"id": task_id, "prompt": "", "meta": {}})
        rec = episode_to_output_record(episode, orig, ref_images_dir=ref_images_dir)
        async with write_lock:
            _append_and_save(results, rec, results_path)
        pbar.update(1)
        pbar.set_postfix(completed=len(results), total=total_count)

    print(f"[GenEval] Pending {len(pending)} samples; starting")
    await asyncio.gather(*[run_and_save(t, tid) for t, tid in pending])
    pbar.close()
    print(f"[GenEval] Done. Results saved to {results_path}; ref images saved to {ref_images_dir}")


if __name__ == "__main__":
    asyncio.run(main())
