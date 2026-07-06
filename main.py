import argparse
import json
import os
import random
from datetime import datetime

from openai import OpenAI

from config import API_KEY, BASE_URL
from data_loader import load_database, get_all_labels
from distractor_generator import generate_distractors
from evaluator import build_question, evaluate


def main():
    parser = argparse.ArgumentParser(description="多模态模型遥感图像缺失预测测试")
    parser.add_argument("--database", type=str, default=None, help="数据库目录路径")
    parser.add_argument("--output", type=str, default=None, help="结果输出JSON文件路径")
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


if __name__ == "__main__":
    main()
