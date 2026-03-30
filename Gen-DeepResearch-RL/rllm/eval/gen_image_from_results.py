"""
Generate images from a result JSON (with gen_prompt/reference_images) or a prompt-only JSON using an API backend or local diffusers.
- If gen_prompt exists and reference_images is non-empty: generate with gen_prompt + reference images
- Otherwise: generate from the original prompt (text-only)
- If input only contains prompt: generate directly from prompt
- diffuser_flux: unified FLUX text-to-image backend (text-only)
Output: save images under OUTPUT_DIR/model_xxx/ and write results.json.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Optional

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Utilities: image base64 (no external api_generator dependency)
# ---------------------------------------------------------------------------

try:
    from PIL import Image
except ImportError:
    Image = None


def _img2base64(img: str | "Image.Image", format: str = "JPEG") -> str:
    if Image is None:
        raise RuntimeError("PIL is required for image encoding")
    if isinstance(img, str):
        if not os.path.exists(img):
            raise FileNotFoundError(f"File not found: {img}")
        Image.MAX_IMAGE_PIXELS = None
        img = Image.open(img)
    if getattr(img, "mode", "") == "RGBA":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=format)
    return __import__("base64").b64encode(buf.getvalue()).decode()


def _base642img(b64: str) -> "Image.Image":
    if Image is None:
        raise RuntimeError("PIL is required")
    data = __import__("base64").b64decode(b64)
    return Image.open(io.BytesIO(data))


# ---------------------------------------------------------------------------
# Data loading and "which backend to use" logic
# ---------------------------------------------------------------------------


def load_records(path: str) -> List[dict]:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("JSON root must be an array")
        records = data
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError("No valid records in input")
    return records


def get_effective_prompt_and_images(record: dict) -> tuple[str, List[str]]:
    """
    Return (prompt, image_paths).
    - If gen_prompt exists and reference_images is non-empty: use gen_prompt + all valid local_path values
    - Otherwise: use record["prompt"], with image_paths empty (text-only)
    """
    prompt = record.get("prompt") or ""
    gen_prompt = (record.get("gen_prompt") or "").strip()
    refs = record.get("reference_images") or []
    if not isinstance(refs, list):
        refs = []
    paths = []
    for r in refs:
        if not isinstance(r, dict):
            continue
        p = (r.get("local_path") or "").strip()
        if p and os.path.exists(p):
            paths.append(p)
    if gen_prompt and paths:
        return gen_prompt, paths
    return prompt, []


# ---------------------------------------------------------------------------
# Generator interface and API retries
# ---------------------------------------------------------------------------


class ImageGeneratorBase:
    """Generation interface: generate(prompt, image_paths) -> PIL.Image. Empty image_paths means text-only."""

    def generate(self, prompt: str, image_paths: Optional[List[str]] = None):
        raise NotImplementedError


def _api_request_with_retry(
    method: str,
    url: str,
    payload: str,
    headers: dict,
    timeout: int = 120,
    max_try: int = 5,
    print_log: bool = False,
) -> Any:
    import requests
    for i in range(max_try):
        try:
            if print_log and max_try > 1:
                print(f"  Request {i + 1}/{max_try}: {url[:80]}...", flush=True)
            r = requests.post(url, data=payload.encode("utf-8") if isinstance(payload, str) else payload, headers=headers, timeout=timeout)
            return r
        except Exception as e:
            if print_log:
                print(f"  Request exception: {e}", flush=True)
            time.sleep(min(30, 3 + 3 * (1.1 ** i)))
    return None


# ---------------------------------------------------------------------------
# API generators: Nano / Seed / GPT (with retries)
# ---------------------------------------------------------------------------


def _extract_inline_image_b64_from_gemini_generate_content(data: Any) -> Optional[str]:
    """Parse Gemini `generateContent` JSON; return base64 image data from the first inlineData part."""
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


class NanoAPIGenerator(ImageGeneratorBase):
    """Nano (e.g. gemini-3-pro-image-preview): Google Generative Language `generateContent` (official REST)."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        timeout: int = 60,
        max_try: int = 5,
        print_log: bool = False,
        poll_interval: float = 1.0,
        max_poll_seconds: int = 180,
    ):
        import requests
        self.requests = requests
        self.api_key = api_key
        self.model_name = model_name
        # Official endpoint (hardcoded): https://ai.google.dev/api/rest/v1beta/models.generateContent
        self.usage_app_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        )
        self.timeout = timeout
        self.max_try = max_try
        self.print_log = print_log
        self.poll_interval = poll_interval
        self.max_poll_seconds = max_poll_seconds

    def generate(self, prompt: str, image_paths: Optional[List[str]] = None):
        images = []
        if image_paths:
            for p in image_paths:
                images.append(Image.open(p) if os.path.isfile(p) else None)
            images = [x for x in images if x is not None]
        parts = [{"text": prompt}]
        for img in images:
            parts.append({
                "inlineData": {"mimeType": "image/jpeg", "data": _img2base64(img)}
            })
        payload = json.dumps({"contents": [{"parts": parts}]}, ensure_ascii=False)
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        last_err = None
        total_deadline = time.time() + self.timeout  # Total time across retries must not exceed timeout
        for attempt in range(self.max_try):
            if time.time() >= total_deadline:
                raise last_err or TimeoutError(f"Total duration exceeded {self.timeout}s")
            req_timeout = min(self.timeout, max(60, int(total_deadline - time.time())))
            try:
                r = _api_request_with_retry(
                    "POST", self.usage_app_url, payload, headers, req_timeout, 1, self.print_log
                )  # 1=no inner retry; outer max_try controls retries
                if r is None or r.status_code != 200:
                    raise RuntimeError(
                        f"Nano generateContent failed: {r.status_code if r else 'NoResponse'} {getattr(r, 'text', '')}"
                    )
                data = r.json()
                if isinstance(data, dict) and data.get("error"):
                    raise RuntimeError(f"Nano API error: {data.get('error')}")
                img_b64 = _extract_inline_image_b64_from_gemini_generate_content(data)
                if not img_b64:
                    raise RuntimeError(f"Nano: no inline image in response: {str(data)[:500]}")
                raw = __import__("base64").b64decode(img_b64)
                return Image.open(io.BytesIO(raw))
            except Exception as e:
                last_err = e
                if (attempt + 1) % 10 == 0:
                    print(f"  [Nano] Still failing after {attempt + 1}/{self.max_try} retries: {e}", flush=True)
                if attempt < self.max_try - 1 and time.time() < total_deadline - 5:
                    time.sleep(min(30, 3 + 3 * (1.1 ** attempt)))
        raise last_err


class SeedAPIGenerator(ImageGeneratorBase):
    """Seedream (Doubao) via Volcengine Ark: POST .../images/generations (OpenAI-style), returns URL or bytes."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "doubao-seedream-4-0-250828",
        timeout: int = 60,
        max_try: int = 5,
        print_log: bool = False,
    ):
        import requests
        self.requests = requests
        self.api_key = api_key
        self.model_name = model_name
        # Volcengine Ark API v3 (images/generations)
        self.usage_app_url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        self.timeout = timeout
        self.max_try = max_try
        self.print_log = print_log

    def generate(self, prompt: str, image_paths: Optional[List[str]] = None):
        images_base64 = []
        if image_paths:
            for p in image_paths:
                if os.path.isfile(p):
                    images_base64.append(_img2base64(p))
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "size": "2K",
            "response_format": "url",
            "watermark": False,
            "sequential_image_generation": "disabled",
        }
        if images_base64:
            payload["image"] = [f"data:image/jpg;base64,{b}" for b in images_base64]
        payload_str = json.dumps(payload)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_err = None
        total_deadline = time.time() + self.timeout  # Total time across retries must not exceed timeout
        for attempt in range(self.max_try):
            if time.time() >= total_deadline:
                raise last_err or TimeoutError(f"Total duration exceeded {self.timeout}s")
            req_timeout = min(self.timeout, max(60, int(total_deadline - time.time())))
            try:
                r = _api_request_with_retry(
                    "POST", self.usage_app_url, payload_str, headers, req_timeout, 1, self.print_log
                )  # 1=no inner retry; outer max_try controls retries
                if r is None:
                    raise RuntimeError("Seed: no response")
                data = r.json()
                if r.status_code != 200 or "data" not in data:
                    raise RuntimeError(f"Seed API error: {data}")
                image_url = data["data"][0]["url"]
                img_r = self.requests.get(image_url, timeout=req_timeout)
                if img_r.status_code != 200:
                    raise RuntimeError(f"Seed: failed to download image: {img_r.status_code}")
                return Image.open(io.BytesIO(img_r.content))
            except Exception as e:
                last_err = e
                if (attempt + 1) % 10 == 0:
                    print(f"  [Seed] Still failing after {attempt + 1}/{self.max_try} retries: {e}", flush=True)
                if attempt < self.max_try - 1 and time.time() < total_deadline - 5:
                    time.sleep(min(30, 3 + 3 * (1.1 ** attempt)))
        raise last_err


class GPTImageAPIGenerator(ImageGeneratorBase):
    """OpenAI Images API: edits when reference images are provided; otherwise generations."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-image-1",
        timeout: int = 300,
        max_try: int = 5,
        print_log: bool = False,
    ):
        import requests
        self.requests = requests
        self.api_key = api_key
        self.model_name = model_name
        # OpenAI official API v1 base; endpoints: /v1/images/generations, /v1/images/edits
        self.base_url = "https://api.openai.com/v1"
        self.timeout = timeout
        self.max_try = max_try
        self.print_log = print_log

    def generate(self, prompt: str, image_paths: Optional[List[str]] = None):
        if image_paths and len(image_paths) > 0:
            images_b64 = [_img2base64(p) for p in image_paths if os.path.isfile(p)]
            if not images_b64:
                image_paths = None
        else:
            images_b64 = []
        if image_paths and images_b64:
            url = f"{self.base_url}/images/edits"
            payload = {
                "model": "gpt-image-1",
                "prompt": prompt,
                "image": images_b64,
                "n": 1,
                "size": "auto",
                "quality": "auto",
            }
        else:
            url = f"{self.base_url}/images/generations"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024",
                "quality": "standard" if self.model_name == "dall-e-3" else "auto",
            }
            if self.model_name == "gpt-image-1":
                payload["background"] = "auto"
                payload["moderation"] = "low"
                payload["output_compression"] = 100
                payload["size"] = "auto"
        payload_str = json.dumps(payload)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_err = None
        total_deadline = time.time() + self.timeout  # Total time across retries must not exceed timeout
        for attempt in range(self.max_try):
            if time.time() >= total_deadline:
                raise last_err or TimeoutError(f"Total duration exceeded {self.timeout}s")
            req_timeout = min(self.timeout, max(60, int(total_deadline - time.time())))
            try:
                r = _api_request_with_retry(
                    "POST", url, payload_str, headers, req_timeout, 1, self.print_log
                )  # 1=no inner retry; outer max_try controls retries
                if r is None:
                    raise RuntimeError("GPT Image: no response")
                data = r.json()
                if r.status_code != 200 or "data" not in data:
                    raise RuntimeError(f"GPT Image API error: {data}")
                img_data = data["data"][0]
                if "b64_json" in img_data:
                    return _base642img(img_data["b64_json"])
                if "url" in img_data:
                    img_r = self.requests.get(img_data["url"], timeout=req_timeout)
                    if img_r.status_code != 200:
                        raise RuntimeError(f"GPT Image: download failed: {img_r.status_code}")
                    return Image.open(io.BytesIO(img_r.content))
                raise RuntimeError("GPT Image response contains no image data")
            except Exception as e:
                last_err = e
                if (attempt + 1) % 10 == 0:
                    print(
                        f"  [GPT Image] Still failing after {attempt + 1}/{self.max_try} retries: {e}",
                        flush=True,
                    )
                if attempt < self.max_try - 1 and time.time() < total_deadline - 5:
                    time.sleep(min(30, 3 + 3 * (1.1 ** attempt)))
        raise last_err


# ---------------------------------------------------------------------------
# Local diffusers: Qwen (separate gen/edit), LongCat, Z-Image, Z-Image-Turbo, FLUX
# ---------------------------------------------------------------------------


class DiffuserQwenGenerator(ImageGeneratorBase):
    """
    Text-only uses the gen model (e.g. Qwen-Image) on gen_device.
    Text+image uses the edit model (e.g. Qwen-Image-Edit-2509) on the edit device pool.
    gen_device: e.g. "cuda:0"
    edit_device: e.g. "cuda:2,cuda:3" (comma-separated)

    Scheduling strategy:
    - Text-only tasks use gen resources (concurrency=1)
    - Text+image tasks use the edit pool (concurrency = number of edit devices)
    - Total concurrency cap = min(3, 1 + number of edit devices)
    """

    def __init__(
        self,
        gen_model: str = "Qwen/Qwen-Image",
        edit_model: str = "Qwen/Qwen-Image-Edit-2509",
        gen_device: str = "cuda:0",
        edit_device: str = "cuda:1",
        torch_dtype: Optional[str] = None,
    ):
        import torch
        self.torch = torch
        self.gen_device = gen_device
        self.edit_devices = [d.strip() for d in str(edit_device).split(",") if d.strip()]
        if not self.edit_devices:
            self.edit_devices = ["cuda:1"]
        dtype = getattr(torch, torch_dtype or "bfloat16", torch.bfloat16)
        self._gen_pipe = None
        self._edit_pipes = [None for _ in self.edit_devices]
        self._gen_model_id = (gen_model or "").strip() or "Qwen/Qwen-Image"
        self._edit_model_id = (edit_model or "").strip() or "Qwen/Qwen-Image-Edit-2509"
        self._dtype = dtype
        self._pipe_init_lock = threading.Lock()
        self._gen_lock = threading.Lock()
        import queue
        self._edit_slots = queue.Queue()
        for i in range(len(self.edit_devices)):
            self._edit_slots.put(i)
        self.max_parallel = min(3, 1 + len(self.edit_devices))

    def _get_gen_pipe(self):
        if self._gen_pipe is None:
            with self._pipe_init_lock:
                if self._gen_pipe is None:
                    from diffusers import DiffusionPipeline
                    self._gen_pipe = DiffusionPipeline.from_pretrained(
                        self._gen_model_id, torch_dtype=self._dtype
                    ).to(self.gen_device)
        return self._gen_pipe

    def _get_edit_pipe(self, slot_idx: int):
        if self._edit_pipes[slot_idx] is None:
            with self._pipe_init_lock:
                if self._edit_pipes[slot_idx] is None:
                    from diffusers import QwenImageEditPlusPipeline
                    pipe = QwenImageEditPlusPipeline.from_pretrained(
                        self._edit_model_id, torch_dtype=self._dtype
                    ).to(self.edit_devices[slot_idx])
                    pipe.set_progress_bar_config(disable=None)
                    self._edit_pipes[slot_idx] = pipe
        return self._edit_pipes[slot_idx]

    def generate(self, prompt: str, image_paths: Optional[List[str]] = None):
        if not image_paths or len(image_paths) == 0:
            with self._gen_lock:
                pipe = self._get_gen_pipe()
                positive_magic = ", Ultra HD, 4K, cinematic composition."
                image = pipe(
                    prompt=prompt + positive_magic,
                    negative_prompt=" ",
                    width=1664,
                    height=928,
                    num_inference_steps=50,
                    true_cfg_scale=4.0,
                    generator=self.torch.Generator(device=self.gen_device).manual_seed(0),
                ).images[0]
                return image

        image_paths = image_paths[:3]  # Qwen supports at most 3 reference images
        images = [Image.open(p) for p in image_paths if os.path.isfile(p)]
        if not images:
            return self.generate(prompt, None)
        slot_idx = self._edit_slots.get()
        inputs = {
            "image": images,
            "prompt": prompt,
            "generator": self.torch.Generator(device=self.edit_devices[slot_idx]).manual_seed(0),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 40,
            "guidance_scale": 1.0,
            "num_images_per_prompt": 1,
        }
        try:
            pipe = self._get_edit_pipe(slot_idx)
            with self.torch.inference_mode():
                out = pipe(**inputs)
            return out.images[0]
        finally:
            self._edit_slots.put(slot_idx)


class DiffuserLongCatGenerator(ImageGeneratorBase):
    """
    LongCat: text-only uses LongCatImagePipeline (gen); text+image uses LongCatImageEditPipeline (edit).
    Similar to Qwen, gen and edit can use different devices.
    """

    def __init__(
        self,
        gen_model: str = "meituan-longcat/LongCat-Image",
        edit_model: str = "meituan-longcat/LongCat-Image-Edit",
        gen_device: str = "cuda:0",
        edit_device: str = "cuda:1",
        torch_dtype: Optional[str] = None,
    ):
        import torch
        self.torch = torch
        self.gen_device = gen_device
        self.edit_device = edit_device
        dtype = getattr(torch, torch_dtype or "bfloat16", torch.bfloat16)
        self._gen_pipe = None
        self._edit_pipe = None
        self._gen_model_id = (gen_model or "").strip() or "meituan-longcat/LongCat-Image"
        self._edit_model_id = (edit_model or "").strip() or "meituan-longcat/LongCat-Image-Edit"
        self._dtype = dtype

    def _get_gen_pipe(self):
        if self._gen_pipe is None:
            from diffusers import LongCatImagePipeline
            self._gen_pipe = LongCatImagePipeline.from_pretrained(
                self._gen_model_id, torch_dtype=self._dtype
            ).to(self.gen_device)
        return self._gen_pipe

    def _get_edit_pipe(self):
        if self._edit_pipe is None:
            from diffusers import LongCatImageEditPipeline
            self._edit_pipe = LongCatImageEditPipeline.from_pretrained(
                self._edit_model_id, torch_dtype=self._dtype
            ).to(self.edit_device)
        return self._edit_pipe

    def generate(self, prompt: str, image_paths: Optional[List[str]] = None):
        if not image_paths or len(image_paths) == 0:
            pipe = self._get_gen_pipe()
            image = pipe(
                prompt,
                height=768,
                width=1344,
                guidance_scale=4.0,
                num_inference_steps=50,
                num_images_per_prompt=1,
                generator=self.torch.Generator(self.gen_device).manual_seed(0),
                enable_cfg_renorm=True,
                enable_prompt_rewrite=True,
            ).images[0]
            return image
        pipe = self._get_edit_pipe()
        images = [Image.open(p).convert("RGB") for p in image_paths if os.path.isfile(p)]
        if not images:
            return self.generate(prompt, None)
        img = images[0]
        image = pipe(
            img,
            prompt,
            negative_prompt="",
            guidance_scale=4.5,
            num_inference_steps=50,
            num_images_per_prompt=1,
            generator=self.torch.Generator(self.edit_device).manual_seed(0),
        ).images[0]
        return image


class DiffuserZImageGenerator(ImageGeneratorBase):
    """Z-Image: text-only. model_id can be a local path or a HuggingFace id (auto-download)."""

    def __init__(
        self,
        model_id: str = "Tongyi-MAI/Z-Image",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
    ):
        import torch
        self.torch = torch
        self.device = device
        load_id = (model_id or "").strip() or "Tongyi-MAI/Z-Image"
        dtype = getattr(torch, torch_dtype, torch.bfloat16)
        from diffusers import ZImagePipeline
        self.pipe = ZImagePipeline.from_pretrained(
            load_id, torch_dtype=dtype, low_cpu_mem_usage=False
        ).to(device)
        self._model_id = load_id

    def generate(self, prompt: str, image_paths: Optional[List[str]] = None):
        image = self.pipe(
            prompt=prompt,
            negative_prompt="",
            height=1280,
            width=720,
            cfg_normalization=False,
            num_inference_steps=50,
            guidance_scale=4,
            generator=self.torch.Generator(self.device).manual_seed(0),
        ).images[0]
        return image


class DiffuserZImageTurboGenerator(ImageGeneratorBase):
    """Z-Image-Turbo: text-only. model_id can be a local path or a HuggingFace id (auto-download)."""

    def __init__(
        self,
        model_id: str = "Tongyi-MAI/Z-Image-Turbo",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
    ):
        import torch
        self.torch = torch
        self.device = device
        load_id = (model_id or "").strip() or "Tongyi-MAI/Z-Image-Turbo"
        dtype = getattr(torch, torch_dtype, torch.bfloat16)
        from diffusers import ZImagePipeline
        self.pipe = ZImagePipeline.from_pretrained(
            load_id, torch_dtype=dtype, low_cpu_mem_usage=False
        ).to(device)

    def generate(self, prompt: str, image_paths: Optional[List[str]] = None):
        image = self.pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=self.torch.Generator(self.device).manual_seed(0),
        ).images[0]
        return image


def _resolve_local_model_path_case_insensitive_contains(path_hint: str, keyword: str) -> Optional[str]:
    """
    Case-insensitive "contains" matching for local model paths.
    - If path_hint exists and contains keyword, return it.
    - If path_hint is a directory, recursively match subdirectories.
    - If path_hint does not exist, fall back to the nearest existing parent directory and recurse.
    """
    if not path_hint:
        return None
    keyword_l = (keyword or "").strip().lower()
    if not keyword_l:
        return None

    hint = os.path.expanduser(path_hint.strip())
    if os.path.exists(hint) and keyword_l in hint.lower():
        return hint

    roots = []
    if os.path.isdir(hint):
        roots.append(hint)
    else:
        cur = hint
        while cur and cur != os.path.dirname(cur):
            parent = os.path.dirname(cur)
            if parent and os.path.isdir(parent):
                roots.append(parent)
                break
            cur = parent

    matches = []
    for root in roots:
        for dirpath, dirnames, _ in os.walk(root):
            for d in dirnames:
                full = os.path.join(dirpath, d)
                if keyword_l in full.lower():
                    matches.append(full)

    if not matches:
        return None
    matches.sort(key=lambda x: (len(x), x.lower()))
    return matches[0]


class DiffuserFluxGenerator(ImageGeneratorBase):
    """
    Unified FLUX text-to-image backend (text-only):
    - FLUX.1-dev
    - FLUX.1-Krea-dev
    - FLUX.2-klein-9B
    - FLUX.2-klein-4B
    The specific variant is chosen by case-insensitive substring matching in model_path.
    """

    force_text_only = True

    _VARIANTS = [
        ("flux.1-krea-dev", "black-forest-labs/FLUX.1-Krea-dev"),
        ("flux.1-dev", "black-forest-labs/FLUX.1-dev"),
        ("flux.2-klein-9b", "black-forest-labs/FLUX.2-klein-9B"),
        ("flux.2-klein-4b", "black-forest-labs/FLUX.2-klein-4B"),
    ]

    def __init__(self, model_path: str, device: str = "cuda", torch_dtype: str = "bfloat16"):
        import torch
        self.torch = torch
        self.device = device
        self.dtype = getattr(torch, torch_dtype, torch.bfloat16)

        hint = (model_path or "").strip()
        hint_l = hint.lower()
        variant = None
        model_id = None
        resolved_local = None
        for k, hf_id in self._VARIANTS:
            if k in hint_l:
                variant = k
                model_id = hf_id
                resolved_local = _resolve_local_model_path_case_insensitive_contains(hint, k)
                break
        if variant is None:
            supported = ", ".join(k for k, _ in self._VARIANTS)
            raise ValueError(
                f"diffuser_flux cannot infer model variant from path: {hint}. "
                f"It must contain one of: {supported}"
            )

        self.variant = variant
        self.model_ref = resolved_local or model_id
        if "flux.2-klein" in self.variant:
            from diffusers import Flux2KleinPipeline
            self.pipe = Flux2KleinPipeline.from_pretrained(self.model_ref, torch_dtype=self.dtype)
        else:
            from diffusers import FluxPipeline
            self.pipe = FluxPipeline.from_pretrained(self.model_ref, torch_dtype=self.dtype)
        self.pipe = self.pipe.to(self.device)

    def generate(self, prompt: str, image_paths: Optional[List[str]] = None):
        # FLUX backend is text-only by design; ignore image_paths.
        if self.variant == "flux.1-dev":
            return self.pipe(
                prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=self.torch.Generator("cpu").manual_seed(0),
            ).images[0]

        if self.variant == "flux.1-krea-dev":
            return self.pipe(
                prompt,
                height=1024,
                width=1024,
                guidance_scale=4.5,
            ).images[0]

        return self.pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=1.0,
            num_inference_steps=4,
            generator=self.torch.Generator(device=self.device).manual_seed(0),
        ).images[0]


class DiffuserLumina2Generator(ImageGeneratorBase):
    """
    Lumina-Image-2.0 text-to-image (text-only).
    - Model: Alpha-VLLM/Lumina-Image-2.0 or a local path
    - Uses enable_model_cpu_offload to reduce VRAM usage
    Inference parameters follow the provided reference settings.
    """

    force_text_only = True

    def __init__(self, model_path: str, device: str = "cuda", torch_dtype: str = "bfloat16"):
        import torch
        from diffusers import Lumina2Pipeline

        self.torch = torch
        self.device = device
        self.dtype = getattr(torch, torch_dtype, torch.bfloat16)

        load_id = (model_path or "").strip() or "Alpha-VLLM/Lumina-Image-2.0"
        self.pipe = Lumina2Pipeline.from_pretrained(load_id, torch_dtype=self.dtype)
        # Offload to CPU as needed (aligned with the reference settings)
        self.pipe.enable_model_cpu_offload()

    def generate(self, prompt: str, image_paths: Optional[List[str]] = None):
        image = self.pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=4.0,
            num_inference_steps=50,
            cfg_trunc_ratio=0.25,
            cfg_normalization=True,
            generator=self.torch.Generator("cpu").manual_seed(0),
        ).images[0]
        return image


class DiffuserSD3Generator(ImageGeneratorBase):
    """
    Stable Diffusion 3.5 text-to-image (large / medium, text-only).
    Distinguish large vs medium by substring matching in model_path:
    - *3.5-large*  -> stabilityai/stable-diffusion-3.5-large
    - *3.5-medium* -> stabilityai/stable-diffusion-3.5-medium
    Inference parameters follow the provided reference settings.
    """

    force_text_only = True

    _VARIANTS = [
        ("3.5-large", "stabilityai/stable-diffusion-3.5-large"),
        ("3.5-medium", "stabilityai/stable-diffusion-3.5-medium"),
    ]

    def __init__(self, model_path: str, device: str = "cuda", torch_dtype: str = "bfloat16"):
        import torch
        from diffusers import StableDiffusion3Pipeline

        self.torch = torch
        self.device = device
        self.dtype = getattr(torch, torch_dtype, torch.bfloat16)

        hint = (model_path or "").strip()
        hint_l = hint.lower()
        variant = None
        model_id = None
        resolved_local = None
        for k, hf_id in self._VARIANTS:
            if k.lower() in hint_l:
                variant = k
                model_id = hf_id
                resolved_local = _resolve_local_model_path_case_insensitive_contains(hint, k)
                break
        if variant is None:
            # If cannot infer from path, default to large
            variant = "3.5-large"
            model_id = "stabilityai/stable-diffusion-3.5-large"

        self.variant = variant
        self.model_ref = resolved_local or model_id
        self.pipe = StableDiffusion3Pipeline.from_pretrained(self.model_ref, torch_dtype=self.dtype)
        self.pipe = self.pipe.to(self.device)

    def generate(self, prompt: str, image_paths: Optional[List[str]] = None):
        if self.variant == "3.5-medium":
            image = self.pipe(
                prompt,
                num_inference_steps=40,
                guidance_scale=4.5,
            ).images[0]
            return image
        image = self.pipe(
            prompt,
            num_inference_steps=28,
            guidance_scale=3.5,
        ).images[0]
        return image


class HunyuanImage3Generator(ImageGeneratorBase):
    """
    HunyuanImage-3 text-to-image (Transformers AutoModelForCausalLM).
    - Text prompt only; reference images are ignored.
    - Model path comes from diffuser_gen_model_path; default is ./HunyuanImage-3.
    """

    force_text_only = True

    def __init__(self, model_path: str, device_spec: Optional[str] = None):
        import os
        from transformers import AutoModelForCausalLM

        self.model_id = (model_path or "").strip() or "./HunyuanImage-3"
        dev = (device_spec or "").strip()
        if dev:
            dev_l = dev.lower()
            if dev_l != "auto":
                # Supported forms:
                # - "cuda:0,1,2"
                # - "0,1,2"
                # - "cuda:3"
                parts = [p.strip() for p in dev.split(",") if p.strip()]
                indices = []
                for p in parts:
                    if p.startswith("cuda:"):
                        p = p[len("cuda:") :]
                    if p.isdigit():
                        indices.append(p)
                if indices:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(indices)

        kwargs = dict(
            attn_implementation="sdpa",
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            moe_impl="eager",
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self.model.load_tokenizer(self.model_id)

    def generate(self, prompt: str, image_paths: Optional[List[str]] = None):
        image = self.model.generate_image(prompt=prompt, stream=True)
        return image


# ---------------------------------------------------------------------------
# Bagel: text-to-image generation
# ---------------------------------------------------------------------------


class BagelTextGenerator(ImageGeneratorBase):
    """
    Bagel text-to-image (text-only).
    - force_text_only=True: ignore reference_images
    - mode=1: normal mode (aligned with eval/new_test/t2i_infer_batch.py defaults)
    - Model path is passed via --diffuser-gen-model-path to gen_image_from_results.py
    """

    force_text_only = True

    def __init__(
        self,
        model_path: str,
        mode: int = 1,
        device_spec: Optional[str] = None,
        image_shapes: tuple = (1024, 1024),
        cfg_text_scale: float = 4.0,
        cfg_interval: tuple = (0.4, 1.0),
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        seed: int = 0,
    ):
        import os
        self.model_path = (model_path or "").strip()
        if not self.model_path:
            raise ValueError("bagel backend requires a model path (pass via --diffuser-gen-model-path)")
        # mode=1 means normal mode. For compatibility, only normal mode is implemented.
        if mode != 1:
            raise ValueError("bagel backend currently only supports normal mode (mode=1)")

        # Reuse DIFFUSER_GEN_DEVICE format (e.g. cuda:0 / cuda:0,1 / 0,1)
        dev = (device_spec or "").strip() or "cuda:0"
        dev_l = dev.lower().strip()
        requested_gpu_count = None
        if dev_l and dev_l != "auto":
            parts = [p.strip() for p in dev.split(",") if p.strip()]
            indices = []
            for p in parts:
                if p.startswith("cuda:"):
                    p = p[len("cuda:") :]
                if p.isdigit():
                    indices.append(p)
            if indices:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(indices)
                requested_gpu_count = len(indices)

        import random
        import numpy as np
        import torch
        from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

        from data.transforms import ImageTransform
        from data.data_utils import add_special_tokens
        from modeling.bagel import (
            Bagel,
            BagelConfig,
            Qwen2Config,
            Qwen2ForCausalLM,
            SiglipVisionConfig,
            SiglipVisionModel,
        )
        from modeling.autoencoder import load_ae
        from modeling.qwen2 import Qwen2Tokenizer
        from inferencer import InterleaveInferencer

        self.os = os
        self.random = random
        self.np = np
        self.torch = torch

        # ----------------- Initialize following the official Bagel pipeline -----------------
        llm_config = Qwen2Config.from_json_file(os.path.join(self.model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(self.model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        vae_model, vae_config = load_ae(local_path=os.path.join(self.model_path, "ae.safetensors"))

        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
        )

        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        tokenizer = Qwen2Tokenizer.from_pretrained(self.model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        # Dynamically estimate max_memory from free memory on visible GPUs to avoid forcing weights onto saturated GPUs
        max_mem_per_gpu = os.environ.get("BAGEL_MAX_MEM_PER_GPU", "40GiB")
        min_free_gib = int(os.environ.get("BAGEL_MIN_FREE_GIB", "8"))
        if torch.cuda.device_count() > 0:
            gib = 1024 ** 3
            max_memory = {}
            free_report = []
            for i in range(torch.cuda.device_count()):
                try:
                    free_bytes, _ = torch.cuda.mem_get_info(i)
                    # Reserve 2GiB buffer; keep at least 1GiB to avoid hitting the exact limit
                    usable_gib = max(1, int(free_bytes / gib) - 2)
                    max_memory[i] = f"{usable_gib}GiB"
                    free_report.append((i, round(free_bytes / gib, 2), usable_gib))
                except Exception:
                    max_memory[i] = max_mem_per_gpu
                    free_report.append((i, -1.0, -1))
            # Allow CPU offload to reduce OOM risk
            max_memory["cpu"] = os.environ.get("BAGEL_MAX_MEM_CPU", "256GiB")
            print(f"[bagel] visible_cuda={torch.cuda.device_count()}, requested={requested_gpu_count or 'auto'}", flush=True)
            for idx, free_gib, usable in free_report:
                print(f"[bagel] cuda:{idx} free={free_gib}GiB usable_for_map={usable}GiB", flush=True)

            # If user explicitly requested 3-GPU parallelism, all 3 GPUs must have sufficient free memory
            if requested_gpu_count == 3:
                low_free = [idx for idx, free_gib, _ in free_report if free_gib >= 0 and free_gib < min_free_gib]
                if low_free:
                    raise RuntimeError(
                        f"Bagel requires 3-GPU parallelism, but these visible GPUs have free memory < {min_free_gib}GiB: {low_free}. "
                        f"Please free the corresponding GPUs and retry."
                    )
                if torch.cuda.device_count() < 3:
                    raise RuntimeError(
                        f"Bagel requires 3-GPU parallelism, but only {torch.cuda.device_count()} GPUs are visible."
                    )
        else:
            max_memory = None

        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )

        same_device_modules = [
            "language_model.model.embed_tokens",
            "time_embedder",
            "latent_pos_embed",
            "vae2llm",
            "llm2vae",
            "connector",
            "vit_pos_embed",
        ]
        if torch.cuda.device_count() <= 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                device_map[k] = device_map.get(k, first_device)
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device

        try:
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=os.path.join(self.model_path, "ema.safetensors"),
                device_map=device_map,
                offload_buffers=True,
                dtype=torch.bfloat16,
                force_hooks=True,
                offload_folder="/tmp/offload",
            )
        except Exception as e:
            raise RuntimeError(f"Bagel checkpoint load failed: {e}") from e
        self.model = model.eval()
        self.inferencer = InterleaveInferencer(
            model=self.model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )
        self.image_shapes = image_shapes
        self.cfg_text_scale = cfg_text_scale
        self.cfg_interval = cfg_interval
        self.timestep_shift = timestep_shift
        self.num_timesteps = num_timesteps
        self.cfg_renorm_min = cfg_renorm_min
        self.cfg_renorm_type = cfg_renorm_type
        self.seed = seed

    def generate(self, prompt: str, image_paths: Optional[List[str]] = None):
        # Text-to-image only; do not use reference_images
        _ = image_paths
        self.random.seed(self.seed)
        self.np.random.seed(self.seed)
        self.torch.manual_seed(self.seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed(self.seed)
            self.torch.cuda.manual_seed_all(self.seed)
        self.torch.backends.cudnn.deterministic = True
        self.torch.backends.cudnn.benchmark = False

        output_dict = self.inferencer(
            text=prompt,
            think=True,
            max_think_token_n=1024,
            do_sample=False,
            text_temperature=0.3,
            cfg_text_scale=self.cfg_text_scale,
            cfg_img_scale=1.0,
            cfg_interval=list(self.cfg_interval),
            timestep_shift=self.timestep_shift,
            num_timesteps=self.num_timesteps,
            cfg_renorm_min=self.cfg_renorm_min,
            cfg_renorm_type=self.cfg_renorm_type,
            image_shapes=self.image_shapes,
        )
        image = output_dict.get("image") if isinstance(output_dict, dict) else None
        if image is None:
            raise RuntimeError("bagel inference failed: no image returned")
        return image


# ---------------------------------------------------------------------------
# Build generator
# ---------------------------------------------------------------------------


def build_generator(args: argparse.Namespace) -> ImageGeneratorBase:
    backend = (getattr(args, "backend", None) or os.environ.get("GEN_IMAGE_BACKEND", "api")).strip().lower()
    api_type = (getattr(args, "api_type", None) or os.environ.get("GEN_IMAGE_API_TYPE", "gpt")).strip().lower()

    if backend == "api":
        api_key = getattr(args, "api_key", None) or os.environ.get("GEN_IMAGE_API_KEY", "")
        model_name = getattr(args, "model_name", None) or os.environ.get("GEN_IMAGE_MODEL", "gpt-image-1")
        timeout = int(getattr(args, "timeout", None) or os.environ.get("GEN_IMAGE_TIMEOUT", "120"))
        max_try = int(getattr(args, "max_try", None) or os.environ.get("GEN_IMAGE_MAX_TRY", "5"))
        if api_type == "nano":
            return NanoAPIGenerator(api_key=api_key, model_name=model_name, timeout=timeout, max_try=max_try, print_log=args.print_log)
        if api_type == "seed":
            return SeedAPIGenerator(api_key=api_key, model_name=model_name, timeout=timeout, max_try=max_try, print_log=args.print_log)
        return GPTImageAPIGenerator(api_key=api_key, model_name=model_name, timeout=timeout, max_try=max_try, print_log=args.print_log)

    gen_model = (getattr(args, "diffuser_gen_model_path", None) or "").strip() or "Qwen/Qwen-Image"
    edit_model = (getattr(args, "diffuser_edit_model_path", None) or "").strip() or "Qwen/Qwen-Image-Edit-2509"
    gen_device = getattr(args, "diffuser_gen_device", None) or "cuda:0"
    edit_device = getattr(args, "diffuser_edit_device", None) or "cuda:1"

    if backend == "diffuser_qwen":
        return DiffuserQwenGenerator(
            gen_model=gen_model,
            edit_model=edit_model,
            gen_device=gen_device,
            edit_device=edit_device,
        )

    if backend == "diffuser_longcat":
        gen_model_lc = (getattr(args, "diffuser_gen_model_path", None) or "").strip() or "meituan-longcat/LongCat-Image"
        edit_model_lc = (getattr(args, "diffuser_edit_model_path", None) or "").strip() or "meituan-longcat/LongCat-Image-Edit"
        return DiffuserLongCatGenerator(
            gen_model=gen_model_lc,
            edit_model=edit_model_lc,
            gen_device=gen_device,
            edit_device=edit_device,
        )

    if backend == "diffuser_zimage":
        return DiffuserZImageGenerator(model_id=gen_model, device=gen_device)

    if backend == "diffuser_zimage_turbo":
        return DiffuserZImageTurboGenerator(model_id=gen_model, device=gen_device)

    if backend == "diffuser_flux":
        return DiffuserFluxGenerator(model_path=gen_model, device=gen_device)

    if backend == "diffuser_lumina2":
        return DiffuserLumina2Generator(model_path=gen_model, device=gen_device)

    if backend == "diffuser_sd3":
        return DiffuserSD3Generator(model_path=gen_model, device=gen_device)

    if backend == "hunyuan_image3":
        return HunyuanImage3Generator(model_path=gen_model, device_spec=gen_device)

    if backend == "bagel":
        # Open-source build: bagel backend is disabled
        raise ValueError("bagel backend is disabled (open-source build does not support it)")

    raise ValueError(f"Unsupported backend: {backend}")


# ---------------------------------------------------------------------------
# Main: read JSON -> generate per record -> write images and results.json
# ---------------------------------------------------------------------------


def _safe_model_suffix(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", s).strip("_") or "model"


def _pick_gt_image(rec: dict) -> Optional[str]:
    """Compatibility for GT fields: prediction.gen_image / top-level gt_image / top-level gen_image."""
    return (
        (rec.get("prediction") or {}).get("gen_image")
        or rec.get("gt_image")
        or rec.get("gen_image")
    )


def _run_one_record(
    generator: ImageGeneratorBase,
    rec: dict,
    index: int,
    images_dir: Path,
    print_log: bool = False,
) -> tuple[int, dict]:
    """Process one record and return (index, entry)."""
    rid = rec.get("id", index)
    sid = str(rid)
    prompt, image_paths = get_effective_prompt_and_images(rec)
    if getattr(generator, "force_text_only", False):
        image_paths = []

    out_name = f"{sid}.png"
    out_path = (images_dir / out_name).resolve()
    entry = {
        "id": rid,
        "prompt": rec.get("prompt", ""),
        "gen_prompt": rec.get("gen_prompt", ""),
        "meta": rec.get("meta", {}),
        "gt_image": _pick_gt_image(rec),
        "used_prompt": prompt,
        "used_images": image_paths,
        "output_path": str(out_path),
        "success": False,
    }
    try:
        img = generator.generate(prompt, image_paths if image_paths else None)
        img.save(out_path)
        entry["success"] = True
    except Exception as e:
        entry["success"] = False
        entry["error"] = str(e)
        if print_log:
            print(f"[{sid}] Generation failed: {e}", flush=True)
    return (index, entry)


def _parse_device_list(device_spec: str) -> List[str]:
    """Parse 'cuda:1,cuda:2,cuda:3' / '1,2,3' into ['1','2','3']."""
    raw = (device_spec or "").strip()
    if not raw:
        return []
    out: List[str] = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        if p.startswith("cuda:"):
            p = p[len("cuda:") :]
        if p.isdigit():
            out.append(p)
    return out


def _run_bagel_worker_mode(args: argparse.Namespace) -> int:
    """
    Internal worker:
    - Read shard input (JSON array)
    - Run bagel generation serially
    - Write results to _bagel_worker_output
    """
    if not args._bagel_worker_output:
        raise ValueError("worker mode missing --_bagel-worker-output")

    records = load_records(args.input)
    out_root = Path(args.output_dir)
    folder_name = _safe_model_suffix(getattr(args, "suffix", "default"))
    save_dir = out_root / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    images_dir = save_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    results_path = save_dir / "results.json"

    generator = build_generator(args)
    worker_results = []
    for i, rec in tqdm(list(enumerate(records)), desc=f"BagelWorker-{args._bagel_worker_id}", unit="sample"):
        _, entry = _run_one_record(generator, rec, i, images_dir, args.print_log)
        try:
            entry["_orig_index"] = int(rec.get("__orig_index", i))
        except Exception:
            entry["_orig_index"] = i
        worker_results.append(entry)

    with open(args._bagel_worker_output, "w", encoding="utf-8") as f:
        json.dump(worker_results, f, ensure_ascii=False, indent=2)
    return 0


def _run_bagel_data_parallel(args: argparse.Namespace, records: list, results_path: Path) -> bool:
    """
    Bagel data-parallel entry:
    - When --diffuser-gen-device includes multiple GPUs, shard samples and launch worker subprocesses
    - Each worker sees only one GPU and loads its own model copy
    - Workers write partial outputs; the main process merges them into results.json

    Returns True if handled and caller can return; False if this branch is not used.
    """
    if getattr(args, "_bagel_worker_mode", False):
        return False
    if getattr(args, "backend", "") != "bagel":
        return False

    # Main process: prefer CLI device list; fall back to env DIFFUSER_GEN_DEVICE
    raw_device_spec = getattr(args, "diffuser_gen_device", None) or os.environ.get("DIFFUSER_GEN_DEVICE", "")
    device_list = _parse_device_list(raw_device_spec)
    if len(device_list) <= 1:
        return False

    success_ids = set()
    id_to_index = {}
    existing_results = []
    if args.resume and results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            for idx, r in enumerate(existing_results):
                iid = r.get("id")
                if iid is not None:
                    sid = str(iid)
                    id_to_index[sid] = idx
                    if r.get("success") is True:
                        success_ids.add(sid)
        except Exception:
            existing_results = []
            id_to_index = {}
            success_ids = set()

    pending = []
    for i, rec in enumerate(records):
        sid = str(rec.get("id", i))
        if args.resume and sid in success_ids:
            continue
        rec_copy = dict(rec)
        rec_copy["__orig_index"] = i
        pending.append(rec_copy)

    if not pending:
        if existing_results:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(existing_results, f, ensure_ascii=False, indent=2)
        else:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
        return True

    shards = [[] for _ in device_list]
    for idx, rec in enumerate(pending):
        shards[idx % len(device_list)].append(rec)

    temp_dir = Path(tempfile.mkdtemp(prefix="bagel_dp_"))
    procs = []
    worker_out_files = []

    try:
        script_path = Path(__file__).resolve()
        for worker_id, (gpu_id, shard) in enumerate(zip(device_list, shards)):
            if not shard:
                continue
            shard_in = temp_dir / f"shard_{worker_id}.json"
            shard_out = temp_dir / f"worker_{worker_id}_results.json"
            with open(shard_in, "w", encoding="utf-8") as f:
                json.dump(shard, f, ensure_ascii=False, indent=2)

            cmd = [
                sys.executable,
                str(script_path),
                "--input",
                str(shard_in),
                "--output-dir",
                str(args.output_dir),
                "--suffix",
                str(args.suffix),
                "--backend",
                "bagel",
                "--diffuser-gen-model-path",
                str(args.diffuser_gen_model_path or ""),
                "--diffuser-gen-device",
                # Note: in the worker process CUDA_VISIBLE_DEVICES is controlled by env;
                # use auto here to avoid BagelTextGenerator overriding env again
                "auto",
                "--_bagel-worker-mode",
                "--_bagel-worker-id",
                str(worker_id),
                "--_bagel-worker-output",
                str(shard_out),
            ]
            if args.print_log:
                cmd.append("--print-log")
            if getattr(args, "diffuser_edit_model_path", None):
                cmd.extend(["--diffuser-edit-model-path", str(args.diffuser_edit_model_path)])
            if getattr(args, "diffuser_edit_device", None):
                cmd.extend(["--diffuser-edit-device", str(args.diffuser_edit_device)])

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            if args.print_log:
                print(
                    f"[bagel-dp] start worker_id={worker_id} gpu_id={gpu_id} shard_size={len(shard)} CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}",
                    flush=True,
                )
            p = subprocess.Popen(cmd, env=env)
            procs.append((worker_id, p))
            worker_out_files.append(shard_out)

        for worker_id, p in procs:
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f"bagel worker {worker_id} exited with non-zero code: {rc}")

        merged = [None] * len(records)
        if args.resume and existing_results:
            for i, rec in enumerate(records):
                sid = str(rec.get("id", i))
                if sid in id_to_index:
                    old = existing_results[id_to_index[sid]]
                    if old.get("success") is True:
                        merged[i] = old
        for out_file in worker_out_files:
            if not out_file.exists():
                continue
            with open(out_file, "r", encoding="utf-8") as f:
                part = json.load(f)
            for entry in part:
                orig_index = int(entry.pop("_orig_index", -1))
                if 0 <= orig_index < len(records):
                    merged[orig_index] = entry

        # Keep one-to-one correspondence with input records to avoid holes
        for i, rec in enumerate(records):
            if merged[i] is None:
                merged[i] = {"id": rec.get("id", i), "pending": True}

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Generate images from result JSON (API or local diffuser)")
    parser.add_argument("--input", "-i", required=True, help="Input JSON/JSONL (id/prompt/gen_prompt/reference_images or prompt-only)")
    parser.add_argument("--output-dir", "-o", required=True, help="Output root directory")
    parser.add_argument("--suffix", default="default", help="Output subdirectory name; results under OUTPUT_DIR/{suffix}/")
    parser.add_argument("--resume", action="store_true", help="Skip ids that already have output")
    parser.add_argument("--print-log", action="store_true", help="Print request/retry logs")

    parser.add_argument(
        "--backend",
        choices=[
            "api",
            "diffuser_qwen",
            "diffuser_longcat",
            "diffuser_zimage",
            "diffuser_zimage_turbo",
            "diffuser_flux",
            "diffuser_lumina2",
            "diffuser_sd3",
            "hunyuan_image3",
        ],
        default="api",
    )
    parser.add_argument("--api-type", choices=["nano", "seed", "gpt"], default="gpt")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max-try", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=12, help="API-mode parallelism (effective only when backend=api)")

    parser.add_argument("--diffuser-gen-model-path", default=None, help="Diffuser gen model: local path or HuggingFace id (e.g. Qwen/Qwen-Image); HF ids will be downloaded")
    parser.add_argument("--diffuser-edit-model-path", default=None, help="Diffuser edit model: local path or HuggingFace id")
    parser.add_argument("--diffuser-gen-device", default=None, help="Diffuser gen device, e.g. cuda:0")
    parser.add_argument("--diffuser-edit-device", default=None, help="Diffuser edit device, e.g. cuda:1")
    # Open-source build disables bagel, so hidden bagel-worker args are not kept
    args = parser.parse_args()

    # if args._bagel_worker_mode:
    #     _run_bagel_worker_mode(args)
    #     return

    records = load_records(args.input)
    out_root = Path(args.output_dir)
    folder_name = _safe_model_suffix(getattr(args, "suffix", "default"))
    save_dir = out_root / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    images_dir = save_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    results_path = save_dir / "results.json"

    # Bagel multi-GPU data-parallel mode: disabled

    generator = build_generator(args)
    backend = getattr(args, "backend", "api")
    if backend == "api":
        worker_count = max(1, int(getattr(args, "parallel", 1)))
    elif backend == "diffuser_qwen":
        worker_count = max(1, int(getattr(generator, "max_parallel", 1)))
    else:
        worker_count = 1
    use_parallel = worker_count > 1

    if use_parallel:
        # Parallel mode (API or qwen): keep results in input order and update by index
        by_id = {}
        if args.resume and results_path.exists():
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                by_id = {str(r.get("id")): r for r in loaded if r.get("id") is not None}
            except Exception:
                pass
        results = [None] * len(records)
        for i, rec in enumerate(records):
            sid = str(rec.get("id", i))
            if sid in by_id and by_id[sid].get("success") is True:
                results[i] = by_id[sid]
        pending_indices = [i for i in range(len(records)) if results[i] is None or results[i].get("success") is not True]
        write_lock = threading.Lock()

        def _write_results():
            out = []
            for i, r in enumerate(results):
                if r is not None:
                    out.append(r)
                else:
                    out.append({"id": records[i].get("id", i), "pending": True})
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

        with ThreadPoolExecutor(max_workers=worker_count) as ex:
            futures = {
                ex.submit(
                    _run_one_record,
                    generator,
                    records[i],
                    i,
                    images_dir,
                    args.print_log,
                ): i
                for i in pending_indices
            }
            for fut in tqdm(as_completed(futures), total=len(pending_indices), desc="GenImage", unit="sample"):
                try:
                    i, entry = fut.result()
                    with write_lock:
                        results[i] = entry
                        _write_results()
                except Exception as e:
                    idx = futures[fut]
                    rec = records[idx]
                    with write_lock:
                        results[idx] = {
                            "id": rec.get("id", idx),
                            "prompt": rec.get("prompt", ""),
                            "gen_prompt": rec.get("gen_prompt", ""),
                            "meta": rec.get("meta", {}),
                            "gt_image": _pick_gt_image(rec),
                            "success": False,
                            "error": str(e),
                        }
                        _write_results()
        # Final write (aligned with input order; unfinished entries keep pending=True)
        final_results = [
            results[i] if results[i] is not None else {"id": records[i].get("id", i), "pending": True}
            for i in range(len(records))
        ]
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
    else:
        # Serial mode (non-API backends or parallel=1)
        results = []
        success_ids = set()
        id_to_index = {}
        if args.resume and results_path.exists():
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                for idx, r in enumerate(results):
                    iid = r.get("id")
                    if iid is not None:
                        sid = str(iid)
                        id_to_index[sid] = idx
                        if r.get("success") is True:
                            success_ids.add(sid)
            except Exception:
                pass

        to_run = [
            (i, rec)
            for i, rec in enumerate(records)
            if not (args.resume and str(rec.get("id", i)) in success_ids)
        ]
        for i, rec in tqdm(to_run, desc="GenImage", unit="sample"):
            _, entry = _run_one_record(generator, rec, i, images_dir, args.print_log)
            rid = rec.get("id", i)
            sid = str(rid)
            if args.resume and sid in id_to_index:
                results[id_to_index[sid]] = entry
            else:
                results.append(entry)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Done. Output dir: {save_dir}; results: {results_path}")


if __name__ == "__main__":
    main()
