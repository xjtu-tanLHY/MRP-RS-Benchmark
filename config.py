import os

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

TEST_MODEL = "qwen-vl-max"
DISTRACTOR_MODEL = "qwen-vl-plus"

DATABASE_DIR = os.path.join(os.path.dirname(__file__), "database")

NUM_DISTRACTORS = 4
NUM_OPTIONS = NUM_DISTRACTORS + 1

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

MAX_RETRIES = 3
