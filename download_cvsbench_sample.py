import os
import json
import ssl
import urllib3
import requests
from pathlib import Path
from tqdm import tqdm

# 禁用 SSL 验证并屏蔽警告
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

BASE_URL = "https://hf-mirror.com"
REPO_ID = "zlyzlyzly/CVSBench"
REPO_TYPE = "dataset"
BRANCH = "main"

# 下载到 database/CVSBench_sample
ROOT_DIR = Path(__file__).resolve().parent
LOCAL_DIR = ROOT_DIR / "database" / "CVSBench_sample"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)


def hf_raw_url(path: str) -> str:
    return f"{BASE_URL}/datasets/{REPO_ID}/raw/{BRANCH}/{path}"


def hf_resolve_url(path: str) -> str:
    """获取实际文件内容（适用于 Git LFS 大文件）。"""
    return f"{BASE_URL}/datasets/{REPO_ID}/resolve/{BRANCH}/{path}"


def _is_lfs_pointer(path: Path) -> bool:
    """检查本地文件是否只是 Git LFS 指针文件。"""
    if not path.exists() or path.stat().st_size > 512:
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline()
        return first_line.startswith("version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


def download_file(remote_path: str, local_path: Path, resolve_lfs: bool = False):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    # 如果已存在且不是 LFS 指针，则跳过
    if local_path.exists() and not _is_lfs_pointer(local_path):
        return
    url = hf_resolve_url(remote_path) if resolve_lfs else hf_raw_url(remote_path)
    r = requests.get(url, verify=False, timeout=120, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(local_path, "wb") as f, tqdm(
        desc=local_path.name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def list_tree(path: str, recursive: bool = False):
    url = f"{BASE_URL}/api/datasets/{REPO_ID}/tree/{BRANCH}/{path}?recursive={'true' if recursive else 'false'}"
    r = requests.get(url, verify=False, timeout=60)
    r.raise_for_status()
    return r.json()


def sample_image_pairs(num_pairs: int = 10):
    """从 CVUSA 中随机抽取若干卫星-街景图像对。"""
    # 读取一个 annotation 文件，获取图像 ID
    ann_path = LOCAL_DIR / "cvusa" / "gs_grounding" / "Ground2Sat_5level_test.jsonl"
    if not ann_path.exists():
        print(f"Annotation file not found: {ann_path}, skip image sampling.")
        return

    ids = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(ids) >= num_pairs * 2:
                break
            item = json.loads(line)
            img_id = item.get("img_id", "").split("_")[0]
            if img_id and img_id not in ids:
                ids.append(img_id)
            if len(ids) >= num_pairs:
                break

    print(f"Selected {len(ids)} image pairs: {ids[:5]}{'...' if len(ids) > 5 else ''}")

    for img_id in ids:
        sat_path = f"cvusa/data/bingmap/input{img_id}.png"
        street_path = f"cvusa/data/streetview/{img_id}.jpg"
        for rp in [sat_path, street_path]:
            lp = LOCAL_DIR / rp
            if lp.exists() and not _is_lfs_pointer(lp):
                print(f"  Exists: {lp.name}")
                continue
            try:
                download_file(rp, lp, resolve_lfs=True)
            except Exception as e:
                print(f"  Failed to download {rp}: {e}")


def main():
    files_to_download = [
        "README.md",
        "cvusa/g2s/Ground2Sat_VQA_test.jsonl",
        "cvusa/s2g/Sat2Ground_VQA_test.jsonl",
        "cvusa/gs_grounding/Ground2Sat_5level_test.jsonl",
        "cvusa/gs_grounding/Sat2Ground_5level_test.jsonl",
        "fov/g2s/Ground2Sat_VQA_test.jsonl",
        "fov/s2g/Sat2Ground_VQA_test.jsonl",
    ]

    print(f"Downloading CVSBench sample to: {LOCAL_DIR}\n")

    for rp in files_to_download:
        lp = LOCAL_DIR / rp
        if lp.exists():
            print(f"Exists: {rp}")
            continue
        print(f"Downloading: {rp}")
        try:
            download_file(rp, lp)
        except Exception as e:
            print(f"Failed to download {rp}: {e}")

    print("\nDownloading sample image pairs from cvusa/data ...")
    sample_image_pairs(num_pairs=10)

    print("\nDone.")


if __name__ == "__main__":
    main()
