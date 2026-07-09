import os
from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件（如果存在）
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# 模型 API 配置
# 该密钥实际对应阿里云 DashScope（百炼）平台，因此默认使用 DashScope 兼容接口
API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("MOONSHOT_API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 评测使用的视觉语言模型
# DashScope 可用视觉模型示例：qwen-vl-max, qwen-vl-plus, qwen3-vl-plus, qvq-max 等
TEST_MODEL = os.getenv("TEST_MODEL", "qwen-vl-max")
DISTRACTOR_MODEL = os.getenv("DISTRACTOR_MODEL", "qwen-vl-plus")

# 数据库目录
DATABASE_DIR = os.path.join(os.path.dirname(__file__), "database")

# 题目选项配置
NUM_DISTRACTORS = 4
NUM_OPTIONS = NUM_DISTRACTORS + 1

# 支持的图片格式
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# API 调用最大重试次数
MAX_RETRIES = 3
