import os
import random
import string
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from openai import OpenAI

from config import API_KEY, BASE_URL, TEST_MODEL, NUM_OPTIONS, MAX_RETRIES
from data_loader import ImageItem


@dataclass
class Question:
    item: ImageItem
    options: List[str]
    correct_index: int = 0
    model_answer: str = ""
    is_correct: bool = False

    @property
    def correct_label(self) -> str:
        return self.item.label

    def format_options(self) -> str:
        lines = []
        for idx, option in enumerate(self.options):
            letter = string.ascii_uppercase[idx]
            lines.append(f"{letter}. {option}")
        return "\n".join(lines)

    def get_answer_letter(self) -> str:
        if self.model_answer:
            return self.model_answer.strip().upper()[0]
        return ""

    def check_answer(self) -> bool:
        letter = self.get_answer_letter()
        if not letter:
            self.is_correct = False
            return False
        idx = string.ascii_uppercase.index(letter) if letter in string.ascii_uppercase else -1
        self.is_correct = idx == self.correct_index
        return self.is_correct


def build_question(item: ImageItem, distractors: List[str]) -> Question:
    options = [item.label] + distractors
    random.shuffle(options)
    correct_index = options.index(item.label)
    return Question(item=item, options=options, correct_index=correct_index)


def _build_test_prompt(question: Question) -> str:
    return (
        "请观察这张遥感图像，图像中有一个被遮挡/缺失的区域。"
        "请从以下选项中选出缺失区域最可能包含的物品或地物，只需回答选项字母。\n\n"
        f"{question.format_options()}\n\n"
        "请只回答一个字母（A/B/C/D/E），不要解释。"
    )


def _get_mime_type(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
    if ext == "jpg":
        ext = "jpeg"
    return f"image/{ext}"


def ask_model(
    question: Question,
    client: Optional[OpenAI] = None,
) -> str:
    if client is None:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    image_b64 = question.item.load_base64()
    mime_type = _get_mime_type(question.item.image_path)

    prompt = _build_test_prompt(question)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=TEST_MODEL,
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
                temperature=0.0,
            )
            answer = response.choices[0].message.content.strip()
            question.model_answer = answer
            question.check_answer()
            return answer
        except Exception as exc:
            print(f"  模型调用失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {exc}")

    question.model_answer = ""
    question.is_correct = False
    return ""


@dataclass
class EvalResult:
    total: int = 0
    correct: int = 0
    per_label: Dict[str, Dict] = field(default_factory=dict)
    details: List[Dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"总题数: {self.total}",
            f"正确数: {self.correct}",
            f"总准确率: {self.accuracy:.2%}",
            "",
            "各标签准确率:",
        ]
        for label, info in sorted(self.per_label.items()):
            label_acc = info["correct"] / info["total"] if info["total"] > 0 else 0
            lines.append(f"  {label}: {info['correct']}/{info['total']} ({label_acc:.2%})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "per_label": self.per_label,
            "details": self.details,
        }


def evaluate(questions: List[Question], client: Optional[OpenAI] = None) -> EvalResult:
    result = EvalResult()
    for idx, question in enumerate(questions):
        print(f"[{idx + 1}/{len(questions)}] 测试: {question.item.image_path} (正确答案: {question.correct_label})")
        ask_model(question, client)

        detail = {
            "image": question.item.image_path,
            "label": question.correct_label,
            "options": question.options,
            "correct_index": question.correct_index,
            "model_answer": question.model_answer,
            "is_correct": question.is_correct,
        }
        result.details.append(detail)
        result.total += 1
        if question.is_correct:
            result.correct += 1

        label = question.correct_label
        if label not in result.per_label:
            result.per_label[label] = {"total": 0, "correct": 0}
        result.per_label[label]["total"] += 1
        if question.is_correct:
            result.per_label[label]["correct"] += 1

        status = "✓" if question.is_correct else "✗"
        print(f"  模型回答: {question.model_answer} {status}")

    return result
