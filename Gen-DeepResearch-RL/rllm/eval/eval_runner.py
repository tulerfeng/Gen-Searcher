"""
Benchmark evaluation runner for Vision DeepResearch (single workflow).

Goals:
- Use the same workflow/tools/reward as training (`DeepResearchWorkflow`, `deepresearch_reward_fn`).
- Default rollout: OpenAI-compatible (can point to local vLLM server via base_url).
- Input: Parquet only with columns question/answer/(images).
- Save full trajectories (sanitized) and metrics under eval/outputs/<timestamp>/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse
from urllib.request import urlopen

from datasets import load_dataset

import yaml
from PIL import Image

from vision_deepresearch_async_workflow.deepresearch_tools_async_executor import (
    get_all_tools,
)
from vision_deepresearch_async_workflow.deepresearch_workflow import (
    DeepResearchWorkflow,
)
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout import OpenAIEngine
from rllm.rewards.reward_fn import deepresearch_reward_fn


# ---------------------- Data loading ---------------------- #


def _extract_from_content(content: list) -> tuple[str, List[Any]]:
    """Convert OpenAI-style content array (text + image_url) to question + images list."""
    texts: List[str] = []
    images: List[Any] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and "text" in item:
            texts.append(str(item["text"]))
        elif item.get("type") == "image_url":
            url = item.get("image_url", {}) or {}
            if isinstance(url, dict) and "url" in url:
                images.append({"url": url["url"]})
    question = "\n".join([t for t in texts if t.strip()])
    return question, images


def _url_to_bytes(url: str) -> dict:
    try:
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"}:
            with urlopen(url, timeout=10) as resp:
                data = resp.read()
                return {"bytes": data, "origin_url": url}
    except Exception:
        pass
    return {"url": url}


def _normalize_images(images: List[Any]) -> List[Any]:
    normalized: List[Any] = []
    for img in images:
        if isinstance(img, str) and img.startswith(("http://", "https://")):
            normalized.append(_url_to_bytes(img))
        elif isinstance(img, dict) and "url" in img and isinstance(img["url"], str):
            normalized.append(_url_to_bytes(img["url"]))
        else:
            normalized.append(img)
    return normalized


def _record_to_task(record: dict) -> dict:
    """Strict mapping to the unified schema question/answer/images."""
    question = record.get("question", "")
    answer = record.get("answer", "")
    images: List[Any] = []

    if "images" in record and record["images"] is not None:
        imgs = record["images"]
        images.extend(imgs if isinstance(imgs, list) else [imgs])

    if not isinstance(question, str):
        question = str(question)
    if not isinstance(answer, str):
        answer = str(answer)

    return {
        "id": record.get("id") or record.get("idx"),
        "question": question,
        "answer": answer,
        "images": _normalize_images(images),
    }


def _record_from_parquet(rec: dict, idx: int) -> dict:
    # Strict allowed fields
    if "question" not in rec or "answer" not in rec:
        raise ValueError(f"Record {idx} missing required fields 'question'/'answer'")

    question_raw = rec.get("question")
    answer_raw = rec.get("answer")

    question = str(question_raw) if question_raw is not None else ""
    if not question.strip():
        raise ValueError(f"Record {idx} has empty question")

    answer = str(answer_raw) if answer_raw is not None else ""
    if not answer.strip():
        raise ValueError(f"Record {idx} has empty answer")
    images_raw = rec.get("images", [])
    if images_raw is None:
        images_raw = []
    images_list: List[Any] = (
        images_raw if isinstance(images_raw, list) else [images_raw]
    )

    return {
        "id": rec.get("id") or rec.get("idx") or rec.get("_id"),
        "question": question,
        "answer": answer,
        "images": _normalize_images(images_list),
    }


def load_tasks(args) -> List[dict]:
    if not args.parquet:
        raise ValueError(
            "Parquet path is required. Please set --parquet or data.parquet."
        )

    ds_dict = load_dataset("parquet", data_files=str(args.parquet))
    ds = ds_dict["train"]

    allowed_cols = {"question", "answer", "images", "id", "idx", "_id","image_caption","question_original"}
    required_cols = {"question", "answer"}
    cols = set(ds.column_names)

    missing = required_cols - cols
    if missing:
        raise ValueError(f"Parquet file missing required columns: {sorted(missing)}")

    extras = cols - allowed_cols
    if extras:
        raise ValueError(
            f"Parquet file contains unsupported columns (allowed: question/answer/images/id): {sorted(extras)}"
        )

    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    tasks: List[dict] = []
    for idx, rec in enumerate(ds):
        rec_dict = dict(rec)
        task = _record_from_parquet(rec_dict, idx)
        tasks.append(task)

    if not tasks:
        raise ValueError("No valid tasks loaded from Parquet.")
    return tasks


# ---------------------- Rollout setup ---------------------- #


def build_rollout_engine(args):
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.max_tokens is not None:
        sampling_params["max_tokens"] = args.max_tokens
    # OpenAIEngine also works with local OpenAI-compatible servers (e.g., vLLM)
    return OpenAIEngine(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        sampling_params=sampling_params,
    )


# ---------------------- Serialization helpers ---------------------- #


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, Image.Image):
        return {"type": "PIL.Image", "size": obj.size}
    if isinstance(obj, bytes):
        return f"<bytes:{len(obj)}>"
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def episode_to_dict(ep) -> Dict[str, Any]:
    trajectories = []
    for tr in ep.trajectories or []:
        steps = []
        for st in tr.steps or []:
            action_val = getattr(st.action, "action", st.action)
            steps.append(
                {
                    "chat_completions": _sanitize(st.chat_completions),
                    "model_response": _sanitize(st.model_response),
                    "action": _sanitize(action_val),
                    "observation": _sanitize(st.observation),
                    "reward": st.reward,
                }
            )
        trajectories.append(
            {
                "name": tr.name,
                "reward": tr.reward,
                "info": _sanitize(getattr(tr, "info", {})),
                "steps": steps,
            }
        )

    return {
        "id": ep.id,
        "task": _sanitize(ep.task),
        "termination_reason": (
            ep.termination_reason.value if ep.termination_reason else None
        ),
        "is_correct": ep.is_correct,
        "metrics": _sanitize(ep.metrics),
        "info": _sanitize(ep.info),
        "trajectories": trajectories,
    }


# ---------------------- Metrics & IO ---------------------- #


def compute_metrics(episodes: Iterable[Any]) -> dict:
    episodes = list(episodes)
    total = len(episodes)
    correct = sum(1 for ep in episodes if getattr(ep, "is_correct", False))
    termination = {}
    rewards: List[float] = []
    for ep in episodes:
        reason = ep.termination_reason.value if ep.termination_reason else "unknown"
        termination[reason] = termination.get(reason, 0) + 1
        if ep.trajectories:
            rewards.extend(
                [tr.reward for tr in ep.trajectories if tr.reward is not None]
            )
    avg_reward = (sum(rewards) / len(rewards)) if rewards else None
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "termination_distribution": termination,
        "average_reward": avg_reward,
    }


def save_outputs(episodes: List[Any], metrics: dict, output_dir: Path, config: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    ep_path = output_dir / "episodes.jsonl"
    with ep_path.open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(episode_to_dict(ep), ensure_ascii=False) + "\n")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved episodes to {ep_path}")
    print(f"📊 Metrics: {metrics}")


# ---------------------- Arg parsing ---------------------- #


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepResearch evaluation (Parquet only)"
    )
    parser.add_argument(
        "--config", default="eval/config/eval_hle.yaml", help="YAML config path"
    )

    # Data (Parquet only)
    parser.add_argument("--parquet", default=None, help="Local Parquet path (required)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples")

    # Rollout
    parser.add_argument(
        "--provider",
        default=None,
        help="openai (default) | vllm (still uses OpenAIEngine base_url)",
    )
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument(
        "--temperature", type=float, default=None, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=None, help="Top-p")
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="Max tokens for completion"
    )

    # Execution
    parser.add_argument(
        "--parallel-tasks", type=int, default=None, help="Parallel tasks"
    )
    parser.add_argument("--retry-limit", type=int, default=None, help="Retry limit")
    parser.add_argument(
        "--output-dir", default=None, help="Output dir (relative to eval/)"
    )

    return parser.parse_args()


def load_config(path: Path) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def merge_args_with_config(cfg: dict, args) -> argparse.Namespace:
    # Rollout source
    provider = args.provider or cfg.get("provider", "openai")
    rollout_cfg = cfg.get(provider, {}) if isinstance(cfg, dict) else {}

    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    exec_cfg = cfg.get("execution", {}) if isinstance(cfg, dict) else {}
    out_cfg = cfg.get("output", {}) if isinstance(cfg, dict) else {}

    merged = argparse.Namespace(
        provider=provider,
        model=args.model or rollout_cfg.get("model") or "gpt-4o",
        base_url=args.base_url
        or rollout_cfg.get("base_url")
        or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=args.api_key
        or rollout_cfg.get("api_key")
        or os.getenv("OPENAI_API_KEY", ""),
        temperature=(
            args.temperature
            if args.temperature is not None
            else rollout_cfg.get("sampling_params", {}).get("temperature", 0.6)
        ),
        top_p=(
            args.top_p
            if args.top_p is not None
            else rollout_cfg.get("sampling_params", {}).get("top_p", 0.95)
        ),
        max_tokens=(
            args.max_tokens
            if args.max_tokens is not None
            else rollout_cfg.get("sampling_params", {}).get("max_tokens")
        ),
        parquet=args.parquet or data_cfg.get("parquet"),
        max_samples=(
            args.max_samples
            if args.max_samples is not None
            else data_cfg.get("max_samples")
        ),
        parallel_tasks=(
            args.parallel_tasks
            if args.parallel_tasks is not None
            else exec_cfg.get("parallel_tasks", 4)
        ),
        retry_limit=(
            args.retry_limit
            if args.retry_limit is not None
            else exec_cfg.get("retry_limit", 1)
        ),
        output_dir=args.output_dir or out_cfg.get("dir") or "./outputs",
    )
    return merged


# ---------------------- Main ---------------------- #


async def main():
    args_cli = parse_args()
    cfg = load_config(Path(args_cli.config))
    args = merge_args_with_config(cfg, args_cli)

    tasks = load_tasks(args)
    if not tasks:
        raise ValueError("No tasks loaded. Please check dataset or JSONL path.")

    tools = get_all_tools()
    rollout_engine = build_rollout_engine(args)

    workflow_args = {
        "tools": tools,
        "reward_function": deepresearch_reward_fn,
    }

    workflow_engine = AgentWorkflowEngine(
        workflow_cls=DeepResearchWorkflow,
        workflow_args=workflow_args,
        rollout_engine=rollout_engine,
        n_parallel_tasks=args.parallel_tasks,
        retry_limit=args.retry_limit,
    )

    print(f"🚀 Running DeepResearch evaluation with {len(tasks)} tasks")
    episodes = await workflow_engine.execute_tasks(tasks)

    metrics = compute_metrics(episodes)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / args.output_dir / timestamp
    save_outputs(list(episodes), metrics, output_dir, cfg)


if __name__ == "__main__":
    asyncio.run(main())
