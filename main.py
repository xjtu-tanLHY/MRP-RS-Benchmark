import argparse
import json
import os
import random
import sys
from datetime import datetime

# 在 Windows 等默认编码非 UTF-8 的终端上强制使用 UTF-8 输出中文
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI

from config import API_KEY, BASE_URL
from data_loader import load_database, get_all_labels
from distractor_generator import generate_distractors
from evaluator import build_question, evaluate


def write_list_md(result, list_md_path: str):
    """将每个类别的准确率写入 Markdown 文件。"""
    lines = [
        "# 评测结果：各标签准确率",
        "",
        f"- 总题数: {result.total}",
        f"- 正确数: {result.correct}",
        f"- 总准确率: {result.accuracy:.2%}",
        "",
        "| 标签 | 总题数 | 正确数 | 准确率 |",
        "|------|--------|--------|--------|",
    ]

    # 按标签名字母顺序排列，也可改为按准确率排序
    for label in sorted(result.per_label.keys()):
        info = result.per_label[label]
        label_acc = info["correct"] / info["total"] if info["total"] > 0 else 0
        lines.append(
            f"| {label} | {info['total']} | {info['correct']} | {label_acc:.2%} |"
        )

    lines.append("")
    with open(list_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="多模态模型遥感图像缺失预测测试")
    parser.add_argument("--database", type=str, default=None, help="数据库目录路径")
    parser.add_argument("--output", type=str, default=None, help="结果输出JSON文件路径")
    parser.add_argument("--list-md", type=str, default="List.md", help="类别准确率 Markdown 文件路径（默认 List.md）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--limit", type=int, default=None, help="限制测试图片数量")
    args = parser.parse_args()

    random.seed(args.seed)

    if not API_KEY:
        print("错误: 请设置环境变量 DASHSCOPE_API_KEY")
        print("  Windows: set DASHSCOPE_API_KEY=your-key")
        print("  Linux/Mac: export DASHSCOPE_API_KEY=your-key")
        return

    database_dir = args.database or os.path.join(os.path.dirname(__file__), "database")

    print(f"加载数据库: {database_dir}")
    items = load_database(database_dir)
    if not items:
        print("错误: 数据库为空，请确保 database 目录下有子文件夹且包含图片")
        return

    if args.limit:
        items = items[:args.limit]

    all_labels = get_all_labels(items)
    print(f"共加载 {len(items)} 张图片，{len(all_labels)} 个标签: {all_labels}")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print("\n生成干扰项并构建题目...")
    questions = []
    for idx, item in enumerate(items):
        print(f"  [{idx + 1}/{len(items)}] {item.image_path} -> 标签: {item.label}")
        distractors = generate_distractors(item, all_labels, client)
        question = build_question(item, distractors)
        questions.append(question)
        print(f"    选项: {question.options}")

    print(f"\n开始评测，共 {len(questions)} 题...")
    result = evaluate(questions, client)

    print("\n" + "=" * 50)
    print("评测结果:")
    print("=" * 50)
    print(result.summary())

    output_path = args.output or os.path.join(
        os.path.dirname(__file__),
        f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {output_path}")

    write_list_md(result, args.list_md)
    print(f"类别准确率已保存到: {args.list_md}")


if __name__ == "__main__":
    main()
