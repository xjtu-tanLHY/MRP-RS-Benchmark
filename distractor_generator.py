import json
import os
import random
from typing import List, Optional

from openai import OpenAI

from config import API_KEY, BASE_URL, DISTRACTOR_MODEL, NUM_DISTRACTORS, MAX_RETRIES
from data_loader import ImageItem


def _build_distractor_prompt() -> str:
    return (
        "你是一个遥感图像分析专家。请观察这张遥感图像，"
        "图像中有一个被遮挡/缺失的区域。请列出图中可能存在但与缺失区域无关的物品或地物。"
        "请以JSON数组格式返回，只返回数组，不要其他内容。例如：[\"道路\",\"河流\",\"农田\"]。"
        f"请至少列出{NUM_DISTRACTORS + 2}个候选干扰项。"
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
