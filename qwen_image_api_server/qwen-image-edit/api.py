import os
import re
import base64
import time
import random
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from io import BytesIO
import requests
import argparse
import threading
from urllib.parse import urlparse
import traceback  # <<< ADDED >>>
import gc  # <<< ADDED >>>

# ----------------------------
# Global variables & config
# ----------------------------
app = FastAPI(title="Qwen Image Edit API")

# Model and GPU state
pipelines = []  # List of (pipeline, device)
gpu_locks = []  # One lock per GPU; only one request at a time per GPU
gpu_available = []  # Per-GPU idle flags (optional optimization)
model_name = "Qwen/Qwen-Image-Edit-2509"
dtype = torch.bfloat16

# Thread pool for blocking model inference
executor = ThreadPoolExecutor(max_workers=8)  # Tune max_workers for load

# <<< ADDED >>>
# Dedicated reload thread pool (avoid reload starvation under inference load)
reload_executor = ThreadPoolExecutor(max_workers=16)

# <<< ADDED >>>
# Reload dedup & isolation
reload_pending = []   # Reload needed (timeout / fatal flag)
reload_inflight = []  # Reload already submitted (dedup)

# <<< ADDED >>>
# GPU failure / recovery (disable bad GPUs; auto re-enable after recovery)
gpu_disabled = []          # True: GPU temporarily excluded from scheduling
gpu_fail_count = []        # Consecutive failure count
GPU_FAIL_DISABLE_THRESHOLD = int(os.environ.get("GPU_FAIL_DISABLE_THRESHOLD", 3))  # <<< ADDED >>>
GPU_DISABLE_COOLDOWN = float(os.environ.get("GPU_DISABLE_COOLDOWN", 300))  # <<< ADDED >>>
gpu_disabled_until = []    # Earliest time to retry recovery (after cooldown)

# GPU wait timeout (seconds)
GPU_WAIT_TIMEOUT = float(os.environ.get("GPU_WAIT_TIMEOUT", 6000))  # <<< ADDED >>>

# Inference timeout (seconds)
INFER_TIMEOUT = float(os.environ.get("INFER_TIMEOUT", 1800))  # <<< ADDED >>>

# CUDA fatal error substrings (trigger pipeline restart)
FATAL_CUDA_KEYWORDS = [
    "unspecified launch failure",
    "illegal memory access",
    "device-side assert",
    "device side assert",
    "cudnn",
    "cublas",
    "misaligned address",
    "an illegal memory access was encountered",
    "xid",
]

def _is_fatal_cuda_error(err: BaseException) -> bool:
    s = str(err).lower()
    # Restart if CUDA/cudnn/cublas-related and any keyword matches
    if "cuda" not in s and "cudnn" not in s and "cublas" not in s:
        return False
    return any(k in s for k in FATAL_CUDA_KEYWORDS)

# <<< ADDED >>>
def _disable_gpu(gpu_id: int, reason: str):
    """Mark GPU unavailable until cooldown, then retry recovery."""
    try:
        gpu_available[gpu_id] = False
        gpu_disabled[gpu_id] = True
        gpu_disabled_until[gpu_id] = time.time() + GPU_DISABLE_COOLDOWN
        print(f"[GPU-DISABLE] GPU {gpu_id} disabled for {GPU_DISABLE_COOLDOWN:.0f}s. Reason: {reason}")
    except Exception:
        pass

# <<< ADDED >>>
def _maybe_try_reenable_gpu(gpu_id: int):
    """After cooldown, try recovery: submit one reload; rejoin pool on success."""
    try:
        if not gpu_disabled[gpu_id]:
            return
        if time.time() < gpu_disabled_until[gpu_id]:
            return
        # Cooldown elapsed: set pending to trigger one reload (deduped)
        reload_pending[gpu_id] = True
        print(f"[GPU-RECOVER] Cooldown passed, trying to recover GPU {gpu_id} by reload...")
        _submit_reload_once(gpu_id)
    except Exception:
        pass

# <<< ADDED >>>
def _mark_gpu_success(gpu_id: int):
    """Clear failure count after success; ensure GPU is marked available."""
    try:
        gpu_fail_count[gpu_id] = 0
        gpu_available[gpu_id] = True
        # If previously disabled, success means recovered
        if gpu_disabled[gpu_id]:
            gpu_disabled[gpu_id] = False
            gpu_disabled_until[gpu_id] = 0.0
            print(f"[GPU-RECOVER] GPU {gpu_id} recovered and re-enabled.")
    except Exception:
        pass

# <<< ADDED >>>
def _mark_gpu_failure(gpu_id: int, reason: str):
    """Record failure; disable the GPU when threshold is reached (other GPUs carry load)."""
    try:
        gpu_fail_count[gpu_id] += 1
        print(f"[GPU-FAIL] GPU {gpu_id} fail_count={gpu_fail_count[gpu_id]} reason={reason}")
        if gpu_fail_count[gpu_id] >= GPU_FAIL_DISABLE_THRESHOLD:
            _disable_gpu(gpu_id, reason=reason)
    except Exception:
        pass

# <<< ADDED >>>
def _submit_reload_once(gpu_id: int):
    """At most one reload job per GPU running or queued at a time."""
    try:
        if gpu_id < 0 or gpu_id >= len(gpu_locks):
            return
        if not reload_pending[gpu_id]:
            return
        if reload_inflight[gpu_id]:
            return

        reload_inflight[gpu_id] = True

        def _job():
            lock = gpu_locks[gpu_id]
            with lock:
                try:
                    ok = _reload_pipeline_on_gpu(gpu_id)  # <<< CHANGED: returns bool
                    if ok:
                        # Reload ok: back in scheduling pool
                        gpu_available[gpu_id] = True
                        reload_pending[gpu_id] = False
                        gpu_fail_count[gpu_id] = 0
                        if gpu_disabled[gpu_id]:
                            gpu_disabled[gpu_id] = False
                            gpu_disabled_until[gpu_id] = 0.0
                            print(f"[GPU-RECOVER] GPU {gpu_id} re-enabled after successful reload.")
                    else:
                        # Reload failed: keep enabled if old pipeline still usable; else stay disabled
                        if pipelines[gpu_id] is not None:
                            gpu_available[gpu_id] = True
                            reload_pending[gpu_id] = False
                            print(f"[RESTART] Reload failed on GPU {gpu_id}, but old pipeline still usable; keeping GPU enabled.")
                        else:
                            _disable_gpu(gpu_id, reason="reload failed and no usable pipeline")
                finally:
                    reload_inflight[gpu_id] = False

        reload_executor.submit(_job)
        print(f"[RESTART] Submitted reload job for GPU {gpu_id} (deduped).")
    except Exception as e:
        print(f"[RESTART] Failed to submit reload for GPU {gpu_id}: {e}")
        traceback.print_exc()

def _reload_pipeline_on_gpu(gpu_id: int) -> bool:
    """
    Reload pipeline while holding gpu_locks[gpu_id] (avoid concurrent use of same GPU).

    Intended behavior:
    - Bad GPU: mark unavailable until fixed; then return to pool
    - Timeout: recover on successful reload; if reload fails but old pipeline works, that is acceptable
    """
    global pipelines
    device = f"cuda:{gpu_id}"
    try:
        print(f"[RESTART] Reloading pipeline on {device} ...")

        # 0) Keep old pipeline for rollback / continued use on failure
        old = None
        try:
            old = pipelines[gpu_id]
        except Exception:
            old = None

        # 1) Preload on CPU first (no VRAM yet)
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            use_safetensors=True,
            low_cpu_mem_usage=False,  # <<< ADDED >>>
        )
        pipe.set_progress_bar_config(disable=True)

        # 1.5) Health check: avoid leftover meta tensors (best-effort)
        def _has_meta(m):
            for _, p in m.named_parameters(recurse=True):
                if getattr(p, "is_meta", False):
                    return True
            return False

        if hasattr(pipe, "unet") and _has_meta(pipe.unet):
            raise RuntimeError("Meta tensors still present in new pipeline (unet).")
        if hasattr(pipe, "vae") and _has_meta(pipe.vae):
            raise RuntimeError("Meta tensors still present in new pipeline (vae).")

        # 2) Drop old pipeline (only one model resident on GPU)
        try:
            pipelines[gpu_id] = None
            if old is not None:
                del old
        except Exception:
            pass
        gc.collect()

        # 3) Clear CUDA cache / sync (best effort to reset bad state)
        try:
            torch.cuda.set_device(gpu_id)
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()  # <<< ADDED >>>
            except Exception:
                pass
        except Exception:
            pass

        # 4) Move to GPU
        pipe.to(device)
        pipelines[gpu_id] = pipe

        print(f"[RESTART] Pipeline reloaded successfully on {device}.")
        return True

    except Exception as e:
        print(f"[RESTART] Failed to reload pipeline on {device}: {e}")
        traceback.print_exc()
        # Roll back: restore old pipeline if possible; avoid leaving pipelines[gpu_id] permanently None
        try:
            if pipelines[gpu_id] is None:
                pipelines[gpu_id] = old
        except Exception:
            pass
        return False

def _schedule_reload_pipeline(gpu_id: int):
    """
    For timeout when the worker thread cannot be force-stopped:
    - Do not flood reload submits; only set flags and isolate this GPU
    - Reload is submitted once after inference ends (in finally), or after cooldown for recovery
    """
    try:
        reload_pending[gpu_id] = True
        gpu_available[gpu_id] = False
        print(f"[RESTART] Marked reload pending for GPU {gpu_id} (will reload once after inference ends).")
    except Exception as e:
        print(f"[RESTART] Failed to mark reload pending for GPU {gpu_id}: {e}")
        traceback.print_exc()

# ----------------------------
# Request body models
# ----------------------------
class GenerateRequest(BaseModel):
    image_urls: List[str]  # Two image URLs
    prompt: str
    seed: Optional[int] = 0
    true_cfg_scale: Optional[float] = 4.0
    negative_prompt: Optional[str] = " "
    num_inference_steps: Optional[int] = 40
    guidance_scale: Optional[float] = 1.0
    num_images_per_prompt: Optional[int] = 1

# ----------------------------
# Helpers
# ----------------------------
def load_image_from_url(url_or_data: str) -> Image.Image:
    """
    Two input forms:
    1. Normal URL: "https://example.com/image.jpg"
    2. Data URL: "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQE..."
    """
    try:
        # Case 1: data URL (starts with data:image/)
        if url_or_data.startswith("data:image/"):
            match = re.match(r"data:image/[^;]+;base64,(.*)", url_or_data)
            if not match:
                raise ValueError("Invalid data URL format")
            base64_data = match.group(1)
            image_data = base64.b64decode(base64_data)
            image = Image.open(BytesIO(image_data)).convert("RGB")
            return image

        # Case 2: normal HTTP(S) URL
        else:
            parsed = urlparse(url_or_data)
            if parsed.scheme in ("http", "https"):
                response = requests.get(url_or_data, timeout=600)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                return image
            else:
                raise ValueError(
                    f"Unsupported URL scheme: {parsed.scheme}. Only http/https and data:image/... supported."
                )

    except Exception as e:
        raise ValueError(f"Failed to load image from input '{url_or_data[:50]}...': {e}")

def run_inference_on_gpu(gpu_id: int, inputs: dict, queue_wait: float):
    """Run inference on one GPU (fatal CUDA errors trigger pipeline reload under this GPU lock)."""
    lock = gpu_locks[gpu_id]
    device = f"cuda:{gpu_id}"

    with lock:
        start_time = time.time()
        try:
            pipeline = pipelines[gpu_id]
            if pipeline is None:
                # Prior restart failed or not loaded yet
                ok = _reload_pipeline_on_gpu(gpu_id)  # <<< CHANGED >>>
                pipeline = pipelines[gpu_id]
                if pipeline is None:
                    _mark_gpu_failure(gpu_id, "pipeline unavailable after reload")
                    raise RuntimeError(f"Pipeline on GPU {gpu_id} is unavailable after reload.")
                else:
                    # Reload failed but old weights still usable; can continue serving
                    if ok:
                        _mark_gpu_success(gpu_id)

            generator = torch.Generator(device=device).manual_seed(inputs["seed"])
            with torch.inference_mode():
                output = pipeline(
                    image=inputs["images"],
                    prompt=inputs["prompt"],
                    generator=generator,
                    true_cfg_scale=inputs["true_cfg_scale"],
                    negative_prompt=inputs["negative_prompt"],
                    num_inference_steps=inputs["num_inference_steps"],
                    guidance_scale=inputs["guidance_scale"],
                    num_images_per_prompt=inputs["num_images_per_prompt"],
                )

            end_time = time.time()
            infer_time = end_time - start_time
            print(f"GPU {gpu_id} inference time: {infer_time:.6f} seconds; queue wait: {queue_wait:.6f} seconds")
            _mark_gpu_success(gpu_id)  # <<< ADDED >>>
            return output.images[0]  # PIL Image

        except Exception as e:
            # Print full traceback
            print(f"GPU {gpu_id} inference failed: {e}")
            traceback.print_exc()

            # Fatal CUDA: mark reload pending + bump count; may disable GPU
            if _is_fatal_cuda_error(e):
                print(f"[RESTART] Detected fatal CUDA error on GPU {gpu_id}, mark reload pending...")
                reload_pending[gpu_id] = True
                gpu_available[gpu_id] = False
                _mark_gpu_failure(gpu_id, "fatal cuda error")  # <<< ADDED >>>

            else:
                _mark_gpu_failure(gpu_id, "inference exception")  # <<< ADDED >>>

            # Re-raise for API layer to return an error
            raise
        finally:
            # After inference thread ends:
            # 1) If pending, submit one deduped reload (avoid piling up reloads on timeout)
            _submit_reload_once(gpu_id)
            # 2) If GPU was disabled and cooldown elapsed, try recovery
            _maybe_try_reenable_gpu(gpu_id)

# ----------------------------
# API routes
# ----------------------------
@app.post("/generate")
async def generate(request: GenerateRequest):
    # Find first idle GPU (with wait timeout)
    gpu_id = None
    wait_start = time.time()
    while True:
        for i, lock in enumerate(gpu_locks):
            # Only if: not locked + available + pipeline not None + not disabled
            _maybe_try_reenable_gpu(i)  # <<< ADDED >>> triggers recovery attempt when cooldown done
            if (not lock.locked()) and gpu_available[i] and (pipelines[i] is not None) and (not gpu_disabled[i]):
                gpu_id = i
                break
        if gpu_id is not None:
            break

        if time.time() - wait_start > GPU_WAIT_TIMEOUT:
            raise HTTPException(status_code=503, detail="All GPUs are busy (wait timeout)")

        await asyncio.sleep(5)

    # Queue wait: from entering generate until a GPU is acquired
    queue_wait = time.time() - wait_start  # <<< ADDED >>>

    try:
        images = [load_image_from_url(url) for url in request.image_urls]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    inputs = {
        "images": images,
        "prompt": request.prompt,
        "seed": request.seed,
        "true_cfg_scale": request.true_cfg_scale,
        "negative_prompt": request.negative_prompt,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        "num_images_per_prompt": request.num_images_per_prompt,
    }

    loop = asyncio.get_event_loop()
    try:
        output_image: Image.Image = await asyncio.wait_for(
            loop.run_in_executor(executor, run_inference_on_gpu, gpu_id, inputs, queue_wait),
            timeout=INFER_TIMEOUT
        )
    except asyncio.TimeoutError:
        # Timeout cannot force-stop the thread: mark pending, stop scheduling new work to this GPU; reload after inference ends
        print(f"[TIMEOUT] Inference timeout on GPU {gpu_id}, mark reload pending and temporarily avoid this GPU...")
        _schedule_reload_pipeline(gpu_id)
        _mark_gpu_failure(gpu_id, "timeout")  # <<< ADDED >>>
        # Temporarily disable to avoid routing to a slow / stuck GPU
        _disable_gpu(gpu_id, reason="timeout")  # <<< ADDED >>>
        raise HTTPException(status_code=504, detail="Inference timeout")
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        # Include traceback in response for debugging (same success=False style as before)
        return {"success": False, "message": f"Inference failed: {str(e)}\n{tb}"}

    buffer = BytesIO()
    output_image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return {"success": True, "image": img_base64}

@app.get("/health")
async def health_check():
    if not pipelines:
        return {"status": "unhealthy", "reason": "Models not loaded"}
    # <<< CHANGED >>> add disabled_gpus without removing existing fields
    return {
        "status": "healthy",
        "num_gpus": len(pipelines),
        "disabled_gpus": [i for i, d in enumerate(gpu_disabled) if d],
    }

# ----------------------------
# Initialization
# ----------------------------
def initialize_pipelines(num_gpus: int):
    global pipelines, gpu_locks
    if num_gpus == 0:
        raise ValueError("num_gpus must be >= 1")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a GPU machine.")

    available_gpus = torch.cuda.device_count()
    if num_gpus > available_gpus:
        print(f"Warning: Requested {num_gpus} GPUs, but only {available_gpus} available. Using {available_gpus}.")
        num_gpus = available_gpus

    print(f"Initializing {num_gpus} pipelines on GPUs 0 to {num_gpus - 1}...")

    for i in range(num_gpus):
        device = f"cuda:{i}"
        print(f"Loading model on {device}...")
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            use_safetensors=True,
            low_cpu_mem_usage=False,  # <<< ADDED >>>
        )
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        pipelines.append(pipe)
        gpu_locks.append(threading.Lock())
        gpu_available.append(True)
        reload_pending.append(False)
        reload_inflight.append(False)
        gpu_disabled.append(False)           # <<< ADDED >>>
        gpu_fail_count.append(0)            # <<< ADDED >>>
        gpu_disabled_until.append(0.0)      # <<< ADDED >>>

    print("All pipelines loaded successfully.")

# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    initialize_pipelines(args.num_gpus)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
