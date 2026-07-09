import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

# 尝试加载数据集
dataset = load_dataset("zlyzlyzly/CVSBench")

print(dataset)