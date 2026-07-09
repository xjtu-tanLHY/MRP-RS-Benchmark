# MRP Test for Remote Sensing Images

多模态模型遥感图像缺失区域预测（Missing Region Prediction）评测工具。通过遮挡遥感图像中的部分区域，测试多模态大模型对缺失内容的推理能力。

## 工作原理

1. **加载数据** — 从 `database/` 目录按子文件夹名作为标签加载遥感图像
2. **生成干扰项** — 调用视觉语言模型分析图像，生成与缺失区域无关的干扰选项
3. **构建题目** — 将正确标签与干扰项随机排列，组成多选题
4. **模型评测** — 让待测模型观察图像并选择缺失区域最可能包含的地物
5. **输出结果** — 计算总准确率及各标签准确率，保存为 JSON 文件

## 项目结构

```
├── main.py                 # 入口程序
├── config.py               # 配置项（模型、API、参数）
├── data_loader.py          # 数据库加载与图像编码
├── distractor_generator.py # 干扰项生成
├── evaluator.py            # 题目构建、模型调用与评测
├── prepare_dior.py         # DIOR 数据集标注转遮挡样本工具
├── requirements.txt        # 依赖
└── database/               # 遥感图像数据库（按类别建子目录）
    ├── 类别A/
    │   ├── img1.jpg
    │   └── img2.png
    └── 类别B/
        └── img3.jpg
```

## 数据库目录规范

在 `database/` 下为每个地物类别创建一个子文件夹，文件夹名即为标签，图片放入对应文件夹：

```
database/
├── 机场/
│   ├── 001.jpg
│   └── 002.png
├── 港口/
│   └── 003.jpg
└── 农田/
    └── 004.jpg
```

支持的图片格式：`.jpg`、`.jpeg`、`.png`、`.bmp`、`.tiff`、`.tif`、`.webp`

## 安装

```bash
pip install -r requirements.txt
```

## 配置

设置阿里云 DashScope API Key 环境变量：

```bash
# Windows
set DASHSCOPE_API_KEY=your-key

# Linux / macOS
export DASHSCOPE_API_KEY=your-key
```

可在 `config.py` 中修改以下配置：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `TEST_MODEL` | `qwen-vl-max` | 评测使用的模型 |
| `DISTRACTOR_MODEL` | `qwen-vl-plus` | 生成干扰项的模型 |
| `NUM_DISTRACTORS` | `4` | 每题干扰项数量 |
| `MAX_RETRIES` | `3` | API 调用最大重试次数 |

## 数据准备（DIOR 数据集）

项目提供了 `prepare_dior.py`，用于把 DIOR 检测数据集转换成本项目可用的“遮挡区域预测”格式。

支持两种模式：

### 1. 自动模式（快速初始化，非手动标注）

直接使用 DIOR 原始标注框生成遮挡样本，无需手动画框。该模式会保存所有样本，**不属于手动标注**：

```bash
python prepare_dior.py --dior-root ./DIOR --output ./database --auto
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--dior-root PATH` | DIOR 数据集根目录，需包含 `Annotations/` 和图片目录 |
| `--output PATH` | 输出目录（默认 `./database`） |
| `--split {all,train,val,test,trainval}` | 只处理指定 split，默认 `all` |
| `--limit INT` | 自动模式下最多处理多少个对象 |
| `--auto` | 启用自动模式 |

### 2. GUI 手动模式

如果需要自己画框（推荐，确保 database 只含你认可的样本）：

```bash
# 首次使用或想清空旧数据，确保 database 只含本次手动标注
python prepare_dior.py --dior-root ./DIOR --output ./database --clear-output

# 追加标注（保留已有数据）
python prepare_dior.py --dior-root ./DIOR --output ./database
```

操作说明：

1. 窗口会显示 DIOR 图片和当前标签（绿色虚线框为原始标注提示）。
2. 用鼠标左键拖拽，在目标物品上画一个红色框。
3. 按 `s` 保存：框内区域会被填充为白色，整张图写入 `./database/<标签>/<图片名>_<序号>.jpg`（文件夹自动创建）。
4. 按 `r` 重画当前框；`n` 跳过当前样本（**不会进入 database**）；`q` 退出。

手动模式保障：

- **只有按 `s` 保存的样本才会写入 database**，按 `n` 跳过的样本不会生成任何图片。
- 支持**断点续标**：退出后会生成 `progress.json`，下次启动时自动从上次位置继续。
  - 如需从头开始，加 `--no-resume`。
  - 如需强制清空旧数据，加 `--clear-output`。
- 退出时会弹出统计：已保存数、已跳过数、总数。
- 生成 `manual_saved.log` 和 `skipped_samples.log` 记录保存与跳过详情。

> 提示：工具会自动识别 `JPEGImages/`、`JPEGImages-trainval/` 等常见图片目录结构。若只下载了 trainval 子集，使用 `--split all` 即可处理所有已下载图片。

## 使用

```bash
python main.py [选项]
```

### 命令行参数

| 参数 | 说明 |
|------|------|
| `--database PATH` | 数据库目录路径（默认 `./database`） |
| `--output PATH` | 结果输出 JSON 文件路径（默认自动生成） |
| `--list-md PATH` | 类别准确率 Markdown 文件路径（默认 `List.md`） |
| `--seed INT` | 随机种子（默认 `42`） |
| `--limit INT` | 限制测试图片数量 |

### 示例

```bash
# 使用默认配置运行
python main.py

# 指定数据库目录和输出文件
python main.py --database ./my_data --output result.json

# 只测试前 10 张图片
python main.py --limit 10
```

## 输出

评测结果保存为 JSON 文件，包含：

- `total` — 总题数
- `correct` — 正确数
- `accuracy` — 总准确率
- `per_label` — 各标签准确率
- `details` — 每题详细信息（图像路径、选项、模型回答、是否正确）
