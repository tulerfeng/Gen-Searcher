import asyncio
import base64
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from vision_deepresearch_async_workflow.tools.shared import (
    DeepResearchTool,
    get_cache_async,
    get_cache_key,
    log_tool_event,
    run_with_retries_async,
    set_cache_async,
)

# Try to import optional dependencies for crop_and_search tool
try:
    from PIL import Image
    import requests
    import oss2

    oss2.defaults.connection_pool_size = 10240
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class CropAndSearchTool(DeepResearchTool):
    """Crop and search tool for visual deep research."""

    MAX_URLS = 3

    def __init__(self):
        if not PIL_AVAILABLE:
            raise ImportError(
                "CropAndSearchTool requires PIL, requests, and oss2 packages"
            )

        super().__init__(
            name="crop_and_search",
            description="Crop regions from an image and perform visual search to gather information. Takes an image_id (path or URL), bbox coordinates (single or multiple), and goal description.",
            parameters={
                "type": "object",
                "properties": {
                    "image_id": {
                        "type": "string",
                        "description": "Path or URL of the image to process",
                    },
                    "bbox": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 4,
                                    "maxItems": 4,
                                },
                                {"type": "number"},
                            ]
                        },
                        "description": "Bounding box coordinates [x1,y1,x2,y2] or array of bboxes",
                    },
                    "goal": {
                        "type": "string",
                        "description": "Description of what to search for in the cropped regions",
                    },
                },
                "required": ["image_id", "bbox"],
            },
        )
        self.oss_access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
        self.oss_access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
        self.oss_endpoint = os.getenv("OSS_ENDPOINT")
        self.oss_bucket_name = os.getenv("OSS_BUCKET_NAME")
        self.zhipu_api_key = os.getenv("ZHIPU_API_KEY")
        self.jina_api_key = os.getenv("JINA_API_KEY")
        self.serp_api_key = os.getenv("SERP_API_KEY")
        self.zhipu_image_search_url = os.getenv(
            "IMAGE_SEARCH_URL",
            "https://search-svip.bigmodel.cn/api/paas/v4/image_search",
        )
        self.serp_image_search_url = os.getenv(
            "IMAGE_SEARCH_URL",
            "https://google.serper.dev/lens",
        )
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
        self.image_crop_cache = os.getenv("IMAGE_CROP_CACHE", None)
        self._oss_bucket = None

    def _get_oss_bucket(self):
        """Get or create OSS bucket instance."""
        if self._oss_bucket is None:
            self._oss_bucket = oss2.Bucket(
                oss2.Auth(self.oss_access_key_id, self.oss_access_key_secret),
                self.oss_endpoint,
                self.oss_bucket_name,
            )
        return self._oss_bucket

    def _select_extract_url(self) -> str | None:
        if not self.extract_urls:
            return None
        selected = random.choice(self.extract_urls)
        if not re.search(r"/v1/chat/completions/?$", selected):
            selected = f"{selected.rstrip('/')}/v1/chat/completions"
        return selected

    def _crop_image_by_bbox(
        self, image_path: str, bbox: List[int], output_dir: str
    ) -> Optional[str]:
        """Crop image by bounding box coordinates."""
        try:
            os.makedirs(output_dir, exist_ok=True)

            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")

                width, height = img.size

                # Convert coordinates (assuming bbox is in 0-1000 range)
                x1 = max(0, min(int(bbox[0] * width / 1000), width - 1))
                y1 = max(0, min(int(bbox[1] * height / 1000), height - 1))
                x2 = max(0, min(int(bbox[2] * width / 1000), width - 1))
                y2 = max(0, min(int(bbox[3] * height / 1000), height - 1))

                if x2 <= x1 or y2 <= y1:
                    log_tool_event(
                        "CropAndSearch", "InvalidBbox", f"bbox={bbox}", level="WARNING"
                    )
                    return None

                # Crop and resize
                cropped_img = img.crop((x1, y1, x2, y2))
                cropped_img = cropped_img.resize(
                    (cropped_img.width * 2, cropped_img.height * 2),
                    Image.Resampling.LANCZOS,
                )

                # Generate deterministic filename based on image path and bbox
                # This ensures same image_id + bbox always produces same filename for caching
                image_basename = os.path.basename(image_path)
                image_name_no_ext = os.path.splitext(image_basename)[0]
                bbox_str = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
                deterministic_name = f"crop_{image_name_no_ext}_{bbox_str}.jpg"

                output_path = os.path.join(output_dir, deterministic_name)
                cropped_img.save(output_path, "JPEG", quality=95)

                return output_path

        except Exception as e:
            log_tool_event("CropAndSearch", "CropError", str(e), level="ERROR")
            return None

    def _upload_to_oss(self, local_path: str) -> Optional[str]:
        """Upload local image to OSS."""
        try:
            filename = os.path.basename(local_path)
            oss_path = filename

            bucket = self._get_oss_bucket()
            with open(local_path, "rb") as f:
                bucket.put_object(oss_path, f)

            endpoint_host = self.oss_endpoint.replace("https://", "").replace(
                "http://", ""
            )
            public_url = f"https://{self.oss_bucket_name}.{endpoint_host}/{oss_path}"
            return public_url

        except Exception as e:
            log_tool_event("CropAndSearch", "UploadError", str(e), level="ERROR")
            return None

    async def _image_search(self, oss_url: str) -> Optional[List[Dict[str, str]]]:
        """Perform image search using Zhipu or Serp API."""
        # Check cache first
        cache_key = get_cache_key(oss_url)
        cached_result = await get_cache_async(
            "image_search", cache_key, executor=self.executor
        )
        if cached_result:
            try:
                return json.loads(cached_result)
            except json.JSONDecodeError:
                pass  # Continue with API call if cache is corrupted

        final_result = None
        if self.zhipu_api_key:
            final_result = await self._image_search_with_zhipu(oss_url)
        else:
            final_result = await self._image_search_with_serp(oss_url)

        # Store result in cache
        if final_result is not None:
            await set_cache_async(
                "image_search",
                cache_key,
                oss_url,
                json.dumps(final_result, ensure_ascii=False),
                executor=self.executor,
            )

        return final_result

    async def _image_search_with_zhipu(
        self, oss_url: str
    ) -> Optional[List[Dict[str, str]]]:
        headers = {
            "Authorization": self.zhipu_api_key,
            "Content-Type": "application/json",
            "Accept": "*/*",
        }
        payload = {"url": oss_url}
        proxies = self._get_requests_proxies()

        def make_search_request():
            response = requests.post(
                self.zhipu_image_search_url,
                headers=headers,
                json=payload,
                timeout=30,
                proxies=proxies,
            )
            response.raise_for_status()
            return response

        try:
            response = await run_with_retries_async(
                func=make_search_request,
                executor=self.executor,
            )

            result_data = response.json()
            search_results = result_data.get("search_result", [])

            formatted_results = []
            for item in search_results[: self.MAX_URLS]:  # Take top 3
                title = item.get("title", "Untitled")
                image_url = item.get("image_url", "")
                link = item.get("link", "")
                source = item.get("source", "")
                thumbnail_url = item.get("thumbnail_url", "")

                if image_url and link:
                    formatted_results.append(
                        {
                            "title": title,
                            "image_url": image_url,
                            "link": link,
                            "bbox_image_url": oss_url,
                            "source": source,
                            "thumbnail_url": thumbnail_url,
                        }
                    )

            return formatted_results if formatted_results else None

        except Exception as e:
            log_tool_event(
                "CropAndSearch",
                "SearchError",
                f"provider=zhipu url={oss_url} error={str(e)}",
                level="ERROR",
            )
            return None

    async def _image_search_with_serp(
        self, oss_url: str
    ) -> Optional[List[Dict[str, str]]]:
        headers = {
            "X-API-KEY": self.serp_api_key,
            "Content-Type": "application/json",
        }
        payload = {"url": oss_url}
        proxies = self._get_requests_proxies()

        def make_search_request():
            response = requests.post(
                self.serp_image_search_url,
                headers=headers,
                json=payload,
                timeout=30,
                proxies=proxies,
            )
            response.raise_for_status()
            return response

        try:
            response = await run_with_retries_async(
                func=make_search_request,
                executor=self.executor,
            )

            result_data = response.json()
            search_results = result_data.get("organic", [])

            formatted_results = []
            for item in search_results[: self.MAX_URLS]:  # Take top 3
                title = item.get("title", "Untitled")
                image_url = item.get("imageUrl", "")
                link = item.get("link", "")
                source = item.get("source", "")
                thumbnail_url = item.get("thumbnailUrl", "")

                if image_url and link:
                    formatted_results.append(
                        {
                            "title": title,
                            "image_url": image_url,
                            "link": link,
                            "bbox_image_url": oss_url,
                            "source": source,
                            "thumbnail_url": thumbnail_url,
                        }
                    )

            return formatted_results if formatted_results else None

        except Exception as e:
            log_tool_event(
                "CropAndSearch",
                "SearchError",
                f"provider=serp url={oss_url} error={str(e)}",
                level="ERROR",
            )
            return None

    # ============================================================================
    # Webpage visiting functions (moved from visit_summary_vl.py)
    # ============================================================================

    def get_num_bytes(base64_str):
        # Ensure base64 padding is correct.
        padding = 4 - len(base64_str) % 4
        if padding < 4:
            base64_str += "=" * padding
        # Decode the base64 string.
        decoded_bytes = base64.b64decode(base64_str)
        # Compute the byte length.
        num_bytes = len(decoded_bytes)

        return num_bytes

    def _validate_base64(self, base64_string: str) -> bool:
        """Validate if a base64 string is valid."""
        try:
            # Check if it contains data URI prefix
            if base64_string.startswith("data:image/"):
                # Extract base64 part
                if (
                    ";base64," in base64_string
                    and self.get_num_bytes(base64_string) > 15000
                ):
                    base64_part = base64_string.split(";base64,", 1)[1]
                else:
                    return False
            else:
                base64_part = base64_string

            # Try to decode
            base64.b64decode(base64_part, validate=True)
            return True
        except Exception:
            return False

    def _encode_local_file_to_base64(self, file_path: str) -> Optional[str]:
        """Encode a local image file to base64 format."""
        try:
            if not os.path.exists(file_path):
                return None

            with open(file_path, "rb") as image_file:
                extension = file_path.split(".")[-1].lower()
                if extension in ["jpg", "jpeg"]:
                    image_format = "jpeg"
                elif extension == "png":
                    image_format = "png"
                elif extension == "gif":
                    image_format = "gif"
                elif extension == "webp":
                    image_format = "webp"
                elif extension == "bmp":
                    image_format = "bmp"
                else:
                    image_format = "jpeg"

                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                result = f"data:image/{image_format};base64,{encoded_string}"

                return result
        except Exception:
            return None

    def _encode_url_to_base64(self, url: str, timeout: int = 30) -> Optional[str]:
        """Encode a network image URL to base64 format."""
        try:
            # Get proxy settings
            proxies = self._get_requests_proxies()

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
            }

            response = requests.get(
                url, timeout=timeout, proxies=proxies, headers=headers, stream=True
            )
            response.raise_for_status()

            content = b""
            max_size = 10 * 1024 * 1024  # 10 MB limit
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > max_size:
                    return None

            if len(content) == 0:
                return None

            # Determine image format
            content_type = response.headers.get("content-type", "")
            image_format = "jpeg"

            if content_type.startswith("image/"):
                image_format = content_type.split("/")[-1].split(";")[0].lower()
            else:
                # Try to infer from content
                if content[:3] == b"\xff\xd8\xff":
                    image_format = "jpeg"
                elif content[:8] == b"\x89PNG\r\n\x1a\n":
                    image_format = "png"
                elif content[:6] in [b"GIF87a", b"GIF89a"]:
                    image_format = "gif"
                elif content[:4] == b"RIFF" and content[8:12] == b"WEBP":
                    image_format = "webp"

            if image_format not in ["jpeg", "png", "gif", "webp", "bmp"]:
                image_format = "jpeg"

            encoded_string = base64.b64encode(content).decode("utf-8")
            result = f"data:image/{image_format};base64,{encoded_string}"

            return result

        except Exception:
            return None

    async def _safe_encode_image_to_base64(
        self, image_path: str, timeout: int = 5
    ) -> Optional[str]:
        """Safely encode an image to base64 with validation."""
        try:
            if image_path.startswith(("http://", "https://")):
                result = await self._run_blocking(
                    lambda: self._encode_url_to_base64(image_path, timeout)
                )
            else:
                result = await self._run_blocking(
                    lambda: self._encode_local_file_to_base64(image_path)
                )

            if result and await self._run_blocking(
                lambda: self._validate_base64(result)
            ):
                return result
            return None
        except Exception:
            return None

    def _extract_images_from_content(self, content: str) -> List[Tuple[str, str]]:
        """Extract all image alt texts and URLs from webpage content."""
        pattern = r"!\[(.*?)\]\((https?://[^\s]+)\)"
        matches = re.findall(pattern, content)
        return matches

    async def _summarize_with_extract_only_text(
        self,
        content: str,
        goal: str,
    ) -> Optional[Dict[str, Any]]:
        """Text-only version of webpage content summarization."""
        TEXT_ONLY_PROMPT = """You are a text analysis assistant. You will receive webpage content (text only) and a user's goal. Your task is to extract information that helps achieve the user's goal.

## Task Guidelines
1. **Content Relevance**: Evaluate how the webpage text relates to the user's goal.
2. **Information Extraction**: Extract key information from the webpage text that supports the user's goal.

## Final Output Requirements
- Output **only** a valid JSON object (no Markdown, code blocks, or any other text).
- The JSON object must contain three keys: `"rational"`, `"evidence"`, and `"summary"`.
- Each key must map to a **string** (use an empty string if no relevant content is available).
- Do not include any additional fields or explanations outside the JSON object.

Example:
{"rational": "Explain why the information is relevant to the goal.", "evidence": "Quote or paraphrase the key supporting content from the webpage.", "summary": "Provide a concise summary that connects the evidence back to the goal."}
"""

        if not content or not content.strip():
            return {
                "rational": "No valid text content extracted from webpage",
                "evidence": "",
                "summary": "Unable to process webpage content, text content is empty",
            }

        max_text_length = 50000
        truncated_content = (
            content[:max_text_length] + "...\n[Content truncated]"
            if len(content) > max_text_length
            else content
        )

        message_content = [
            {"type": "text", "text": f"Webpage content:\n\n{truncated_content}"}
        ]

        if goal:
            message_content.append({"type": "text", "text": f"\nUser's goal: {goal}"})

        messages = [
            {"role": "system", "content": TEXT_ONLY_PROMPT},
            {"role": "user", "content": message_content},
        ]

        # Try to use extract service if available
        extract_url = self._select_extract_url()
        if extract_url:
            try:
                payload = {
                    "model": self.extract_model,
                    "messages": messages,
                    "max_tokens": self.extract_max_tokens,
                }

                headers = {"Content-Type": "application/json"}
                proxies = self._get_requests_proxies()

                response = await run_with_retries_async(
                    lambda: requests.post(
                        extract_url,
                        headers=headers,
                        json=payload,
                        timeout=60,
                        proxies=proxies,
                    ),
                    executor=self.executor,
                )

                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, dict):
                        choices = result.get("choices", [])
                        if choices:
                            content = choices[0].get("message", {}).get("content", "")
                            if content:
                                try:
                                    parsed = json.loads(content.strip())
                                    return parsed
                                except:
                                    pass
            except Exception:
                pass

        return None

    async def _summarize_with_extract(
        self,
        content: str,
        goal: str,
        reader_payload: Dict[str, Any],
        query_image_url: Optional[str] = None,
        title: str = "",
        image_url: str = "",
        thumbnail_url: str = "",
        source: str = "",
        max_images: int = 10,
    ) -> Optional[Dict[str, Any]]:
        """Summarize webpage content using visual language model."""
        EXTRACTOR_PROMPT = """You are a multimodal intelligent assistant capable of analyzing both images and text. You will receive a user's query image, query goal, and relevant web content (including search result previews such as website source, page title, and preview images; as well as the main body text and images retrieved from accessing the web pages). Your task is to extract key information that helps the user achieve their goal.

## Task Guidelines
1. **Image Matching**: Compare the user's query image with the images on the web pages (including search result preview images and images within the page content). Evaluate their visual relevance and determine whether they depict the same entity as the user's query image.
2. **Information Extraction**: When they are determined to be the same entity: Extract from the web content the key information most relevant to the user's goal, to support or fulfill the user's query. When they are determined not to be the same entity: Briefly describe the main visual differences between the query image and the web images, and extract information from the web pages that may still be useful as a reference for the user.

## Final Output Requirements
- Output **only** a valid JSON object (no Markdown, code blocks, comments, or any other text).
- The JSON object must contain three keys: `"rational"`, `"evidence"`, and `"summary"`.
- Each key must map to a **string** (use an empty string if no relevant content is available).
- Do not include any other fields or explanations outside the JSON object.

Example:
{{"rational": "Explain why the information is relevant to the goal.", "evidence": "Quote or paraphrase the key supporting content from the webpage.", "summary": "Provide a concise summary that connects the evidence back to the goal."}}
"""

        message_content: List[Dict[str, Any]] = []

        # 1. User's query image
        if query_image_url:
            query_image_base64 = await self._safe_encode_image_to_base64(
                query_image_url
            )
            if query_image_base64:
                message_content.append(
                    {
                        "type": "text",
                        "text": "User's query image (the image the user is searching for):",
                    }
                )
                message_content.append(
                    {"type": "image_url", "image_url": {"url": query_image_base64}}
                )

        # 2. User's goal
        if goal:
            message_content.append({"type": "text", "text": f"User's goal:\n{goal}"})

        # 3. Search result metadata
        preview_parts = []
        if source:
            preview_parts.append(f"Website source: {source}")
        if title:
            preview_parts.append(f"Page title: {title}")

        if preview_parts:
            message_content.append(
                {
                    "type": "text",
                    "text": "Search result metadata:\n" + "\n".join(preview_parts),
                }
            )

        # Preview images
        preview_items = [
            (label, url)
            for label, url in [("Main image", image_url), ("Thumbnail", thumbnail_url)]
            if url
        ]
        if preview_items:
            preview_tasks = [
                self._safe_encode_image_to_base64(url) for _, url in preview_items
            ]
            preview_results = await asyncio.gather(*preview_tasks)
            for (label, _), img_b64 in zip(preview_items, preview_results):
                if img_b64:
                    message_content.append(
                        {"type": "text", "text": f"Search result {label}:"}
                    )
                    message_content.append(
                        {"type": "image_url", "image_url": {"url": img_b64}}
                    )

        # 4. Webpage content
        if content.strip():
            max_text_length = 50000
            truncated_content = (
                content[:max_text_length] + "...\n[Content truncated]"
                if len(content) > max_text_length
                else content
            )
            message_content.append(
                {"type": "text", "text": "Webpage content:\n\n" + truncated_content}
            )

        # 5. Images from webpage content
        if content:
            image_matches = self._extract_images_from_content(content)
            if image_matches:
                selected_matches = image_matches[:max_images]
                image_tasks = [
                    self._safe_encode_image_to_base64(img_url)
                    for _, img_url in selected_matches
                ]
                image_results = await asyncio.gather(*image_tasks)
                webpage_images = [
                    (alt_text, img_base64)
                    for (alt_text, _), img_base64 in zip(
                        selected_matches, image_results
                    )
                    if img_base64
                ]

                if webpage_images:
                    message_content.append(
                        {"type": "text", "text": "Images from webpage:"}
                    )
                    for alt_text, img_base64 in webpage_images:
                        if alt_text.strip():
                            message_content.append(
                                {"type": "text", "text": f"Image '{alt_text}':"}
                            )
                        message_content.append(
                            {"type": "image_url", "image_url": {"url": img_base64}}
                        )

        # Check if we have content
        has_content = any(
            (item["type"] == "text" and item["text"].strip())
            or item["type"] == "image_url"
            for item in message_content
        )

        if not has_content:
            return {
                "rational": "No valid content extracted from webpage or search results.",
                "evidence": "",
                "summary": "Unable to process webpage and search preview content.",
            }

        # Try extract service first
        extract_url = self._select_extract_url()
        if extract_url:
            try:
                messages = [
                    {"role": "system", "content": EXTRACTOR_PROMPT},
                    {"role": "user", "content": message_content},
                ]

                payload = {
                    "model": self.extract_model,
                    "messages": messages,
                    "max_tokens": self.extract_max_tokens,
                }
                headers = {"Content-Type": "application/json"}
                proxies = self._get_requests_proxies()

                response = await run_with_retries_async(
                    lambda: requests.post(
                        extract_url,
                        headers=headers,
                        json=payload,
                        timeout=60,
                        proxies=proxies,
                    ),
                    executor=self.executor,
                )

                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, dict):
                        choices = result.get("choices", [])
                        if choices:
                            content = choices[0].get("message", {}).get("content", "")
                            if content:
                                try:
                                    cleaned = content.strip()
                                    if cleaned.startswith("```json"):
                                        cleaned = cleaned[7:]
                                    elif cleaned.startswith("```"):
                                        cleaned = cleaned[3:]
                                    if cleaned.endswith("```"):
                                        cleaned = cleaned[:-3]

                                    parsed = json.loads(cleaned)
                                    if isinstance(parsed, dict):
                                        for key in ["rational", "evidence", "summary"]:
                                            if key not in parsed:
                                                parsed[key] = ""
                                        return parsed
                                except:
                                    pass
            except Exception:
                pass

        # Fallback to text-only
        return await self._summarize_with_extract_only_text(
            content, goal, reader_payload
        )

    async def _fetch_reader_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch webpage content using Reader API."""
        # Check cache first
        cache_key = get_cache_key(url)
        cached_result = await get_cache_async(
            "image_visit", cache_key, executor=self.executor
        )
        if cached_result:
            try:
                return json.loads(cached_result)
            except json.JSONDecodeError:
                pass  # Continue with API call if cache is corrupted

        try:
            proxies = self._get_requests_proxies()

            if self.zhipu_api_key:
                headers = {"Content-Type": "application/json"}
                if self.zhipu_api_key:
                    headers["Authorization"] = self.zhipu_api_key

                optional_headers = {
                    "X-Return-Format": "markdown",
                    "X-No-Cache": "false",
                    "X-Timeout": "30",
                    "X-Retain-Images": "true",
                    "X-With-Images-Summary": "true",
                    "X-With-Links-Summary": "true",
                }
                headers.update(
                    {k: v for k, v in optional_headers.items() if v is not None}
                )

                body = {"url": url}

                def send_request():
                    return requests.post(
                        self.zhipu_reader_url,
                        headers=headers,
                        json=body,
                        timeout=30,
                        proxies=proxies,
                    )

                response = await run_with_retries_async(
                    send_request, executor=self.executor
                )

                if response.status_code != 200:
                    return None

                payload = response.json()
                if payload.get("code") != 200:
                    return None

                data = payload.get("data", {})
                result = {
                    "content": data.get("content") or "",
                    "description": data.get("description") or "",
                    "meta": data,
                }
            else:
                headers = {"Authorization": self.jina_api_key}
                body = {"url": url}

                def send_request():
                    return requests.post(
                        self.jina_reader_url,
                        headers=headers,
                        data=body,
                        timeout=30,
                        proxies=proxies,
                    )

                response = await run_with_retries_async(
                    send_request, executor=self.executor
                )

                if response.status_code != 200:
                    return None

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
                    "image_visit",
                    cache_key,
                    url,
                    json.dumps(result, ensure_ascii=False),
                    executor=self.executor,
                )

            return result

        except Exception:
            return None

    async def _handle_single_url(
        self,
        url: str,
        goal: str,
        query_image_url: Optional[str] = None,
        title: str = "",
        thumbnail_url: str = "",
        image_url: str = "",
        source: str = "",
        max_content_chars: int = 120000,
    ) -> str:
        """Handle visiting a single URL."""
        try:
            reader_payload = await self._fetch_reader_content(url)
            if not reader_payload:
                log_tool_event(
                    "CropAndSearch",
                    "ReaderFetchFailed",
                    f"url={url} title={title}",
                    level="ERROR",
                )
                return f"[Error] Failed to fetch content from [{title}]({url})"

            content = reader_payload.get("content") or ""
            description = reader_payload.get("description") or ""

            if not content:
                content = "Webpage content is empty."

            # Truncate content
            if len(content) > max_content_chars:
                content = content[:max_content_chars] + "\n[Content truncated...]"

            # Try visual summarization
            summary_result = await self._summarize_with_extract(
                content=content,
                goal=goal,
                reader_payload=reader_payload,
                query_image_url=query_image_url,
                title=title,
                image_url=image_url,
                thumbnail_url=thumbnail_url,
                source=source,
            )

            if summary_result:
                rational_text = summary_result.get("rational") or ""
                evidence_text = summary_result.get("evidence") or content[:2000] + (
                    "..." if len(content) > 2000 else ""
                )
                summary_text = summary_result.get("summary") or description or ""
            else:
                log_tool_event(
                    "CropAndSearch",
                    "ExtractSummaryFailed",
                    f"url={url} title={title}",
                    level="ERROR",
                )
                rational_text = ""
                evidence_text = content[:2000] + ("..." if len(content) > 2000 else "")
                summary_text = description or "Summary unavailable."

            result = f"The useful information in [{title}]({url}) are:\n\n"
            result += f"Evidence in page:\n{evidence_text}\n\n"
            result += f"Summary:\n{summary_text}\n\n"

            return result

        except Exception as e:
            return f"[Error] Failed to process {url}: {str(e)}"

    async def _visit_webpages_for_search(
        self, search_results: List[Dict[str, str]], goal: str
    ) -> str:
        """Visit webpages for search results and extract relevant information."""
        try:
            # Create concurrent tasks for all webpage visits
            visit_tasks = [
                self._handle_single_url(
                    url=item["link"],
                    goal=goal,
                    query_image_url=item["bbox_image_url"],
                    title=item["title"],
                    thumbnail_url=item["thumbnail_url"],
                    image_url=item["image_url"],
                    source=item["source"],
                )
                for item in search_results
            ]

            # Execute all webpage visits concurrently
            visit_results = await asyncio.gather(*visit_tasks, return_exceptions=True)

            # Process results
            all_results = []
            for i, result in enumerate(visit_results):
                try:
                    if isinstance(result, Exception):
                        log_tool_event(
                            "CropAndSearch",
                            "VisitTaskException",
                            f"webpage_{i+1} error={str(result)}",
                            level="ERROR",
                        )
                        all_results.append(
                            f"[{i+1}] [Error visiting webpage: {str(result)}]"
                        )
                    elif isinstance(result, str):
                        # _handle_single_url returns a string directly
                        all_results.append(f"[{i+1}] {result}")
                    else:
                        log_tool_event(
                            "CropAndSearch",
                            "InvalidVisitResult",
                            f"webpage_{i+1} unexpected_result_type={type(result)}",
                            level="ERROR",
                        )
                        all_results.append(
                            f"[{i+1}] [Invalid result format: {type(result)}]"
                        )
                except Exception as e:
                    log_tool_event(
                        "CropAndSearch",
                        "VisitResultProcessingError",
                        f"webpage_{i+1} error={str(e)}",
                        level="ERROR",
                    )
                    all_results.append(
                        f"[{i+1}] [Error processing visit result: {str(e)}]"
                    )

            return "\n\n=======\n\n".join(all_results)

        except Exception as e:
            log_tool_event("CropAndSearch", "VisitSetupError", str(e), level="ERROR")
            return f"[Error setting up webpage visits: {str(e)}]"

    async def _process_single_bbox(
        self, bbox: List[int], bbox_index: int, image_id: str, cache_dir: str, goal: str
    ) -> Tuple[int, str, Optional[str]]:
        """Process a single bounding box concurrently."""
        try:
            # 1. Crop image (CPU-intensive, run in thread pool)
            cropped_path = await self._run_blocking(
                lambda: self._crop_image_by_bbox(image_id, bbox, cache_dir)
            )
            if not cropped_path:
                log_tool_event(
                    "CropAndSearch",
                    "CropFailed",
                    f"bbox={bbox} image_id={image_id}",
                    level="ERROR",
                )
                return bbox_index, f"Bbox {bbox}: Image cropping failed", None

            # 2. Upload to OSS (I/O bound, run in thread pool)
            oss_url = await self._run_blocking(
                lambda: self._upload_to_oss(cropped_path)
            )
            if not oss_url:
                log_tool_event(
                    "CropAndSearch",
                    "UploadFailed",
                    f"bbox={bbox} cropped_path={cropped_path}",
                    level="ERROR",
                )
                return bbox_index, f"Bbox {bbox}: OSS upload failed", None

            # 3. Perform image search (network I/O, run in thread pool)
            search_results = await self._image_search(oss_url)
            if not search_results:
                log_tool_event(
                    "CropAndSearch",
                    "ImageSearchFailed",
                    f"bbox={bbox} oss_url={oss_url}",
                    level="ERROR",
                )
                return bbox_index, f"Bbox {bbox}: Image search failed", oss_url

            # 4. Visit webpages and extract information (fully concurrent)
            visit_results = await self._visit_webpages_for_search(search_results, goal)

            result_text = (
                f"The search results for bbox {bbox} are as follows:\n{visit_results}"
            )
            return bbox_index, result_text, oss_url

        except Exception as e:
            log_tool_event(
                "CropAndSearch",
                "BboxError",
                f"bbox_{bbox_index+1} error={str(e)}",
                level="ERROR",
            )
            return bbox_index, f"Bbox {bbox}: Processing failed - {str(e)}", None

    async def call(
        self,
        image_id: str,
        bbox: Union[List[int], List[List[int]]],
        goal: str = "",
        **kwargs,
    ) -> str:
        """Execute crop and search operation with controlled concurrency."""
        # Create temporary directory for processing
        cache_dir = self.image_crop_cache
        if cache_dir is None:
            raise ValueError("IMAGE_CROP_CACHE must be provided.")
        os.makedirs(cache_dir, exist_ok=True)

        try:
            # Normalize bbox format
            if isinstance(bbox, list) and len(bbox) > 0:
                if isinstance(bbox[0], list):
                    bbox_list = bbox
                else:
                    bbox_list = [bbox]
            else:
                return "[CropAndSearch] Invalid bbox format"

            # Create concurrent tasks for all bboxes
            tasks = [
                self._process_single_bbox(single_bbox, i, image_id, cache_dir, goal)
                for i, single_bbox in enumerate(bbox_list)
            ]

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and maintain order
            all_results = []
            oss_urls = []

            # Process results while maintaining order
            # Since we created tasks in order, results should be in the same order
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    log_tool_event(
                        "CropAndSearch",
                        "TaskException",
                        f"bbox_{i+1} error={str(result)}",
                        level="ERROR",
                    )
                    all_results.append(
                        f"Bbox {bbox_list[i]}: Task failed - {str(result)}"
                    )
                elif isinstance(result, tuple) and len(result) == self.MAX_URLS:
                    bbox_index, result_text, oss_url = result
                    all_results.append(result_text)
                    if oss_url:
                        oss_urls.append(oss_url)
                else:
                    log_tool_event(
                        "CropAndSearch",
                        "InvalidResult",
                        f"bbox_{i+1} unexpected_result_type={type(result)}",
                        level="ERROR",
                    )
                    all_results.append(f"Bbox {bbox_list[i]}: Invalid result format")

            return "\n\n=======\n\n".join(all_results)

        except Exception as e:
            log_tool_event("CropAndSearch", "ExecutionError", str(e), level="ERROR")
            return f"[CropAndSearch Error] {str(e)}"
