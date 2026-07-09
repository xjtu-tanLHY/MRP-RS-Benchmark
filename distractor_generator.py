import json
import os
import random
from typing import List, Optional

from openai import OpenAI

from config import API_KEY, BASE_URL, DISTRACTOR_MODEL, NUM_DISTRACTORS, MAX_RETRIES
from data_loader import ImageItem


def _build_distractor_prompt() -> str:
    return (
        "You are a remote sensing image analysis expert. Look at this remote sensing image, "
        "which contains a masked/missing region. List objects or land-cover types that may appear "
        "in the image but are NOT related to the missing region. "
        "Return a JSON array only, with no extra text. Example: [\"road\", \"river\", \"farmland\"]. "
        f"Provide at least {NUM_DISTRACTORS + 2} candidate distractors."
    )


def _get_mime_type(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
    if ext == "jpg":
        ext = "jpeg"
    return f"image/{ext}"


def generate_distractors(
    item: ImageItem,
    all_labels: List[str],
    client: Optional[OpenAI] = None,
) -> List[str]:
    """为单张图片生成干扰项。每次请求都是无状态的，不携带历史对话。"""
    if client is None:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    image_b64 = item.load_base64()
    mime_type = _get_mime_type(item.image_path)

    prompt = _build_distractor_prompt()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=DISTRACTOR_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_b64}",
                                },
                            },
                        ],
                    }
                ],
                temperature=0.8,
            )
            text = response.choices[0].message.content.strip()
            # 兼容模型返回 Markdown 代码块的情况，如 ```json ... ```
            if text.startswith("```"):
                text = text.strip("`").strip()
                if text.lower().startswith("json"):
                    text = text[4:].strip()
            candidates = json.loads(text)
            if not isinstance(candidates, list):
                raise ValueError("返回格式不是数组")

            filtered = [
                candidate for candidate in candidates
                if isinstance(candidate, str) and candidate != item.label and candidate not in all_labels
            ]
            random.shuffle(filtered)
            distractors = filtered[:NUM_DISTRACTORS]

            if len(distractors) < NUM_DISTRACTORS:
                fallback_pool = [label for label in all_labels if label != item.label]
                random.shuffle(fallback_pool)
                for fallback_label in fallback_pool:
                    if len(distractors) >= NUM_DISTRACTORS:
                        break
                    if fallback_label not in distractors:
                        distractors.append(fallback_label)

            return distractors[:NUM_DISTRACTORS]

        except Exception as exc:
            print(f"  干扰项生成失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {exc}")

    fallback_pool = [label for label in all_labels if label != item.label]
    random.shuffle(fallback_pool)
    return fallback_pool[:NUM_DISTRACTORS]
