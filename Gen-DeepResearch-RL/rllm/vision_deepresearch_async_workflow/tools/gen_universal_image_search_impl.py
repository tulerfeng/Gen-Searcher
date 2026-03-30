"""
Universal image search implementation.

This module provides:
- searching for image results (Serper Google Images API)
- downloading images with optional proxy pool
- returning a list of {title, url, local_path, page_url} dicts

It intentionally avoids any provider/vendor identifiers in file/class/function names
to keep the open-source version clean.
"""

import hashlib
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import requests

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


_TL = threading.local()


def _get_thread_session() -> requests.Session:
    s = getattr(_TL, "sess", None)
    if s is None:
        s = requests.Session()
        _TL.sess = s
    return s


def _same_host_root(url: str) -> str:
    p = urlparse(url)
    if p.scheme and p.netloc:
        return f"{p.scheme}://{p.netloc}/"
    return "https://www.google.com/"


def _clean_html_b(text: str) -> str:
    return re.sub(r"</?b>", "", text or "")


def _safe_sample_prefix(sample_id: Optional[Union[int, str]]) -> str:
    if sample_id is None:
        return ""
    s = re.sub(r"[^0-9A-Za-z_\-]+", "_", str(sample_id).strip())
    return (s + "-") if s else ""


def _is_valid_image_file(path: str) -> bool:
    if not PIL_AVAILABLE:
        return os.path.exists(path) and os.path.getsize(path) >= 1024
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 1024:
            return False
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def _convert_to_rgb_for_jpeg(im: "Image.Image") -> "Image.Image":
    mode = (im.mode or "").upper()
    if mode in ("RGBA", "LA") or ("A" in mode):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        if "A" in im.getbands():
            alpha = im.getchannel("A")
            bg.paste(im.convert("RGBA"), mask=alpha)
        else:
            bg.paste(im.convert("RGB"))
        return bg
    if mode == "P" or mode != "RGB":
        return im.convert("RGB")
    return im


def _proxies_from_env() -> Optional[Dict[str, str]]:
    # Open-source build: do not use any proxy settings.
    return None


UA = os.environ.get(
    "IMG_DL_UA",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
)
DEFAULT_HEADERS = {
    "User-Agent": UA,
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "DNT": "1",
}


def download_image(
    img_url: str,
    save_dir: str = "saved_img",
    timeout: int = 20,
    sample_id: Optional[Union[int, str]] = None,
    page_url: Optional[str] = None,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    url_hash = hashlib.md5(img_url.encode("utf-8")).hexdigest()
    prefix = _safe_sample_prefix(sample_id)
    local_path = os.path.join(save_dir, f"{prefix}{url_hash}.jpg")
    if _is_valid_image_file(local_path):
        return local_path

    headers = dict(DEFAULT_HEADERS)
    if page_url and isinstance(page_url, str) and page_url.strip():
        headers["Referer"] = page_url.strip()
    else:
        headers["Referer"] = _same_host_root(img_url)

    sess = _get_thread_session()
    r = sess.get(
        img_url,
        timeout=min(timeout, 30),
        stream=True,
        headers=headers,
        allow_redirects=True,
        proxies=None,
    )
    r.raise_for_status()
    ct = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if ct and not ct.startswith("image/"):
        raise ValueError(f"Non-image content-type: {ct} url={img_url}")

    tmp_raw = local_path + ".raw.tmp"
    tmp_jpg = local_path + ".tmp"
    bytes_written = 0
    with open(tmp_raw, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)

    if bytes_written < 1024:
        raise ValueError("Downloaded file too small")

    if PIL_AVAILABLE:
        with Image.open(tmp_raw) as im:
            im.load()
            rgb = _convert_to_rgb_for_jpeg(im)
            rgb.save(tmp_jpg, format="JPEG", quality=85, optimize=True)
        if not _is_valid_image_file(tmp_jpg):
            raise ValueError("Converted JPG invalid")
        os.replace(tmp_jpg, local_path)
    else:
        os.replace(tmp_raw, local_path)

    try:
        if os.path.exists(tmp_raw):
            os.remove(tmp_raw)
    except Exception:
        pass
    return local_path


def _fetch_universal_image_results(query: str, topk: int, max_retry: int) -> List[dict]:
    """
    Fetch image search results via POST to IMAGE_SEARCH_API_BASE_URL (e.g. Serper /images),
    normalizing each hit to the schema expected by _download_from_items().
    Uses SERPER_KEY_ID as X-API-KEY.
    """
    api_key = (os.environ.get("SERPER_KEY_ID") or "").strip()
    if not api_key:
        raise ValueError("SERPER_KEY_ID is not set for image search")

    url = (os.environ.get("IMAGE_SEARCH_API_BASE_URL") or "").strip()
    if not url:
        raise ValueError("IMAGE_SEARCH_API_BASE_URL is not set for image search")

    topk = min(max(1, topk), 20)
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    # `num` is accepted by Serper for many endpoints; if the API rejects it, remove and slice client-side.
    payload = {
        "q": query,
        "num": topk,
    }

    empty_results_max_retry = 50
    empty_retry_cnt = 0

    for retry in range(max_retry):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30, proxies=None)

            if resp.status_code == 429:
                sleep_time = min(20, 1.2 * retry) + random.uniform(0, 10)
                print(
                    f"[ImageSearch] 429 Too Many Requests, retrying in {sleep_time:.2f}s query={query!r}",
                    flush=True,
                )
                time.sleep(sleep_time)
                continue

            resp.raise_for_status()
            data = resp.json()
            _err = None
            if isinstance(data, dict):
                _err = data.get("message") or data.get("error")
            if isinstance(data, dict) and _err and not data.get("images"):
                raise RuntimeError(f"Serper images error: {_err}")

            # Serper returns images[]; use `or []` so JSON null becomes []
            raw_items = data.get("images") or []
            if not isinstance(raw_items, list):
                raise RuntimeError(f"Unexpected results type: {type(raw_items)}")

            items: List[dict] = []
            for item in raw_items[:topk]:
                image_url = (
                    item.get("imageUrl")
                    or item.get("image_url")
                    or item.get("url")
                    or item.get("thumbnailUrl")
                    or item.get("thumbnail_url")
                    or ""
                )
                page_url = (
                    item.get("link")
                    or item.get("pageUrl")
                    or item.get("sourceUrl")
                    or item.get("source_url")
                    or ""
                )

                if not image_url:
                    continue

                # Serper schema (camelCase): title, imageUrl, thumbnailUrl, link, source, domain, googleUrl, ...
                title_txt = (item.get("title") or item.get("source") or "").strip() or "image"
                items.append(
                    {
                        "title": title_txt,
                        "imageUrl": image_url,
                        "thumbnailUrl": item.get("thumbnailUrl") or item.get("thumbnail_url") or "",
                        "link": page_url,
                        "sourceUrl": item.get("sourceUrl") or item.get("source_url") or "",
                    }
                )

            print(
                f"[ImageSearch] status_code={resp.status_code} results_len={len(items)} query={query!r}",
                flush=True,
            )

            if len(items) == 0 and empty_retry_cnt < empty_results_max_retry:
                empty_retry_cnt += 1
                sleep_time = random.uniform(0, 15)
                print(
                    f"[ImageSearch] EMPTY_RESULTS retry={empty_retry_cnt}/{empty_results_max_retry} "
                    f"sleep={sleep_time:.2f}s query={query!r}",
                    flush=True,
                )
                time.sleep(sleep_time)
                continue

            return items
        except Exception as e:
            print(f"[ImageSearch] _fetch_results retry={retry} error: {e}", flush=True)
            if retry == max_retry - 1:
                raise
            time.sleep(1 + random.uniform(0, 2))

    raise RuntimeError(f"Image search failed after {max_retry} retries")


def _download_from_items(
    items: List[dict],
    fetch_k: int,
    desired_k: int,
    save_dir: str,
    sample_id: Optional[Union[int, str]],
) -> List[Dict[str, str]]:
    download_timeout = 30
    candidates: List[dict] = []
    for idx, item in enumerate(items[:fetch_k]):
        title = _clean_html_b((item.get("title") or "image").strip())
        img_url = (
            item.get("imageUrl")
            or item.get("image_url")
            or item.get("thumbnailUrl")
            or item.get("thumbnail_url")
            or item.get("url")
            or ""
        )
        page_url = item.get("link") or item.get("pageUrl") or item.get("sourceUrl") or item.get("url") or ""
        if not img_url:
            continue
        candidates.append(
            {
                "idx": idx,
                "title": title,
                "img_url": img_url,
                "page_url": page_url,
            }
        )

    if not candidates:
        return []

    def _download_one(c: dict) -> Optional[Dict[str, Union[int, str]]]:
        try:
            local_path = download_image(
                c["img_url"],
                save_dir=save_dir,
                timeout=download_timeout,
                sample_id=sample_id,
                page_url=c["page_url"] or None,
            )
            return {
                "idx": c["idx"],
                "title": c["title"],
                "url": c["img_url"],
                "local_path": local_path,
                "page_url": c["page_url"],
            }
        except Exception as e:
            print(f"[ImageSearch] download_image failed url={c['img_url']!r} err={e}", flush=True)
            return None

    max_workers = min(len(candidates), 1)
    success_items: List[Dict[str, Union[int, str]]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_download_one, c) for c in candidates]
        for fut in as_completed(futures):
            item = fut.result()
            if item is not None:
                success_items.append(item)

    # Keep the original "prefer top candidates" behavior.
    success_items.sort(key=lambda x: int(x["idx"]))
    results: List[Dict[str, str]] = []
    for item in success_items[:desired_k]:
        results.append(
            {
                "title": str(item["title"]),
                "url": str(item["url"]),
                "local_path": str(item["local_path"]),
                "page_url": str(item["page_url"]),
            }
        )
    return results


def search_universal_image(
    query: str,
    topk: int = 10,
    max_retry: int = 100,
    save_dir: str = "saved_img",
    sample_id: Optional[Union[int, str]] = None,
) -> List[Dict[str, str]]:
    desired_k = min(max(1, topk), 20)
    items10 = _fetch_universal_image_results(query=query, topk=10, max_retry=max_retry)
    res10 = _download_from_items(items10, 10, desired_k, save_dir, sample_id)
    if len(res10) >= desired_k:
        out_early = res10[:desired_k]
        urls_early = [r.get("url", "") for r in out_early if r.get("url")]
        print(f"[ImageSearch] Final returned {len(out_early)} images (from top-10), urls: {urls_early}", flush=True)
        return out_early[:5]

    items20 = _fetch_universal_image_results(query=query, topk=20, max_retry=max_retry)
    res20 = _download_from_items(items20, 20, desired_k, save_dir, sample_id)
    seen = set()
    out: List[Dict[str, str]] = []
    for r in res10 + res20:
        key = r.get("local_path") or r.get("url") or ""
        if key and key not in seen:
            seen.add(key)
            out.append(r)
            if len(out) >= desired_k:
                break
    final_urls = [r.get("url", "") for r in out if r.get("url")]
    print(f"[ImageSearch] Final returned {len(out)} images, urls: {final_urls}", flush=True)
    return out[:5]

