import os
import base64
from dataclasses import dataclass, field
from typing import List, Optional

from config import DATABASE_DIR, IMAGE_EXTENSIONS


@dataclass
class ImageItem:
    image_path: str
    label: str
    image_base64: str = field(default="", repr=False)

    def load_base64(self) -> str:
        if not self.image_base64:
            with open(self.image_path, "rb") as file:
                self.image_base64 = base64.b64encode(file.read()).decode("utf-8")
        return self.image_base64


def load_database(database_dir: str = DATABASE_DIR) -> List[ImageItem]:
    items: List[ImageItem] = []
    if not os.path.isdir(database_dir):
        raise FileNotFoundError(f"数据库目录不存在: {database_dir}")

    for folder_name in sorted(os.listdir(database_dir)):
        folder_path = os.path.join(database_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for filename in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            image_path = os.path.join(folder_path, filename)
            items.append(ImageItem(
                image_path=image_path,
                label=folder_name,
            ))

    return items


def get_all_labels(items: List[ImageItem]) -> List[str]:
    seen: set = set()
    labels: List[str] = []
    for item in items:
        if item.label not in seen:
            seen.add(item.label)
            labels.append(item.label)
    return labels
