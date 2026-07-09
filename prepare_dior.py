import argparse
import json
import os
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageTk

# 在 Windows 等默认编码非 UTF-8 的终端上强制使用 UTF-8 输出中文
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# DIOR 数据集的 20 个类别
DIOR_CLASSES = {
    "airplane", "airport", "baseballfield", "basketballcourt", "bridge",
    "chimney", "dam", "expressway-service-area", "expressway-toll-station",
    "golffield", "groundtrackfield", "harbor", "overpass", "ship",
    "stadium", "storagetank", "tenniscourt", "trainstation", "vehicle",
    "windmill",
}


def parse_dior_xml(xml_path: Path):
    """解析 DIOR 的 PASCAL VOC 格式 XML，返回所有 object 的列表。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename", default="")
    size_elem = root.find("size")
    width = int(size_elem.findtext("width", default="0")) if size_elem is not None else 0
    height = int(size_elem.findtext("height", default="0")) if size_elem is not None else 0

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="")
        difficult = obj.findtext("difficult", default="0")
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            xmin = int(float(bndbox.findtext("xmin", default="0")))
            ymin = int(float(bndbox.findtext("ymin", default="0")))
            xmax = int(float(bndbox.findtext("xmax", default="0")))
            ymax = int(float(bndbox.findtext("ymax", default="0")))
        except ValueError:
            continue
        objects.append({
            "name": name,
            "difficult": int(difficult),
            "bbox": (xmin, ymin, xmax, ymax),
        })

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects,
    }


def find_image_dir(dior_root: Path) -> Path | None:
    """自动发现 DIOR 图片目录。支持 JPEGImages、JPEGImages-trainval 等常见结构。"""
    candidates = [
        dior_root / "JPEGImages",
        dior_root / "JPEGImages-trainval",
        dior_root / "JPEGImages-trainval" / "JPEGImages-trainval",
        dior_root / "JPEGImages-test",
        dior_root / "JPEGImages-test" / "JPEGImages-test",
    ]
    for cand in candidates:
        if cand.exists() and any(cand.glob("*.jpg")):
            return cand
    # 兜底：找任意包含 .jpg 的子目录
    for subdir in dior_root.iterdir():
        if subdir.is_dir() and any(subdir.glob("*.jpg")):
            return subdir
    return None


def load_split_set(image_sets_dir: Path, split: str) -> set:
    """加载 ImageSets/Main/<split>.txt 中的图片名集合。"""
    split_file = image_sets_dir / f"{split}.txt"
    if not split_file.exists():
        return set()
    names = set()
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.add(name)
    return names


def build_sample_list(
    dior_root: Path,
    split: str | None = None,
    skip_difficult: bool = True,
) -> list:
    """遍历 DIOR 标注，生成样本列表。"""
    dior_root = Path(dior_root)
    anno_dir = dior_root / "Annotations"
    image_dir = find_image_dir(dior_root)
    image_sets_dir = dior_root / "ImageSets" / "Main"

    if not anno_dir.exists():
        print(f"错误：标注目录不存在 {anno_dir}")
        return []
    if image_dir is None:
        print(f"错误：在 {dior_root} 下未找到图片目录")
        return []

    print(f"图片目录: {image_dir}")
    print(f"标注目录: {anno_dir}")

    allowed_names = None
    if split and split != "all":
        allowed_names = load_split_set(image_sets_dir, split)
        if not allowed_names:
            print(f"警告：未找到 split 文件 {image_sets_dir / f'{split}.txt'}，将使用全部数据")
            allowed_names = None
        else:
            print(f"使用 split '{split}'，共 {len(allowed_names)} 张图片")

    samples = []
    skipped_no_image = 0
    for xml_path in sorted(anno_dir.glob("*.xml")):
        name = xml_path.stem
        if allowed_names is not None and name not in allowed_names:
            continue

        anno = parse_dior_xml(xml_path)
        if not anno["objects"]:
            continue

        # 尝试多种图片文件名
        image_path = None
        for candidate_name in [anno["filename"], name + ".jpg", name + ".png"]:
            cand = image_dir / candidate_name
            if cand.exists():
                image_path = cand
                break
        if image_path is None:
            skipped_no_image += 1
            continue

        for obj in anno["objects"]:
            if skip_difficult and obj.get("difficult"):
                continue
            label = obj["name"]
            samples.append({
                "image_path": image_path,
                "label": label,
                "anno_bbox": obj["bbox"],
                "xml_path": xml_path,
            })

    print(f"共加载 {len(samples)} 个标注对象（来自 {len(list(anno_dir.glob('*.xml')))} 个 XML，跳过 {skipped_no_image} 个无图片的标注）")
    return samples


def save_masked_image(
    image_path: Path,
    label: str,
    bbox: tuple,
    output_dir: Path,
    suffix: str = "",
    quality: int = 95,
) -> Path:
    """将 bbox 区域涂白并保存到 output_dir/<label>/。"""
    safe_label = "".join(c for c in label if c.isalnum() or c in "-_ ").strip()
    save_dir = output_dir / safe_label
    save_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(list(bbox), fill=(255, 255, 255))

    out_name = f"{image_path.stem}{suffix}.jpg"
    out_path = save_dir / out_name
    image.save(out_path, "JPEG", quality=quality)
    return out_path


class DIORPreparationTool:
    def __init__(
        self,
        dior_root: Path,
        output_dir: Path,
        split: str | None = None,
        max_display_size: tuple = (1280, 800),
        clear_output: bool = False,
        resume: bool = True,
    ):
        self.dior_root = Path(dior_root)
        self.output_dir = Path(output_dir)
        self.split = split
        self.max_display_size = max_display_size
        self.clear_output = clear_output
        self.resume = resume

        self.samples = build_sample_list(self.dior_root, split=self.split)
        self.index = 0
        self.saved_count = 0
        self.skipped_count = 0
        self.saved_log_path = self.output_dir / "manual_saved.log"
        self.skipped_log_path = self.output_dir / "skipped_samples.log"
        self.progress_path = self.output_dir / "progress.json"

        # 当前状态
        self.original_image: Image.Image | None = None
        self.display_image: Image.Image | None = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.current_bbox = None  # 显示坐标系下的 bbox
        self.anno_bbox = None     # 原始坐标系下的标注 bbox（提示）

        if self.resume:
            self._load_progress()
        self._init_ui()

    def _prepare_output_dir(self):
        """手动模式下清空输出目录（如用户确认），确保 database 只含本次手动标注。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.clear_output and any(self.output_dir.iterdir()):
            if messagebox.askyesno(
                "清空输出目录",
                f"输出目录 {self.output_dir} 已存在内容，是否清空？\n"
                "（选择“否”将保留已有文件，但手动标注结果会混入其中）",
            ):
                for item in self.output_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                print(f"已清空输出目录: {self.output_dir}")

    def _load_progress(self):
        """读取进度文件，从上次标注的位置继续。"""
        if not self.progress_path.exists():
            return
        try:
            with open(self.progress_path, "r", encoding="utf-8") as f:
                progress = json.load(f)
            last_index = progress.get("last_index", -1)
            if 0 <= last_index < len(self.samples) - 1:
                self.index = last_index + 1
                print(f"检测到上次标注进度，将从第 {self.index + 1}/{len(self.samples)} 个样本继续")
            else:
                print("进度文件显示已全部完成，将从头开始")
                self.index = 0
        except Exception as e:
            print(f"读取进度文件失败: {e}，将从头开始")
            self.index = 0

    def _save_progress(self):
        """保存当前进度到文件。"""
        try:
            progress = {
                "last_index": self.index,
                "total": len(self.samples),
                "saved_count": self.saved_count,
                "skipped_count": self.skipped_count,
            }
            with open(self.progress_path, "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存进度失败: {e}")

    def _init_ui(self):
        self.root = tk.Tk()
        self.root.title("DIOR 数据准备工具")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 信息栏
        self.info_label = tk.Label(
            self.root,
            text="",
            font=("Microsoft YaHei", 12),
            anchor="w",
            justify="left",
        )
        self.info_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # 画布
        self.canvas = tk.Canvas(self.root, bg="#333333")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 提示栏
        hint = (
            "操作说明：鼠标左键拖拽画框 → s 保存（框内变白并写入 database/<标签>/，"
            "未画框时按 s 会自动使用 DIOR 绿色虚线标注框） → "
            "r 重画 → n 跳过 → q 退出"
        )
        self.hint_label = tk.Label(
            self.root,
            text=hint,
            font=("Microsoft YaHei", 10),
            fg="gray",
        )
        self.hint_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        # 绑定事件
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.root.bind("<Key>", self._on_key)

        # 窗口大小
        self.root.geometry(f"{self.max_display_size[0]}x{self.max_display_size[1]}")
        self.root.update_idletasks()

        # 准备输出目录
        self._prepare_output_dir()

    def _load_sample(self):
        if self.index >= len(self.samples):
            messagebox.showinfo("完成", "所有样本已处理完毕")
            self.root.destroy()
            return

        sample = self.samples[self.index]
        self.current_sample = sample
        self.original_image = Image.open(sample["image_path"]).convert("RGB")
        self.anno_bbox = sample["anno_bbox"]
        self.current_bbox = None
        self.start_x = None
        self.start_y = None
        self.rect_id = None

        # 缩放图像以适应窗口
        self._fit_image()
        self._render()

    def _fit_image(self):
        img_w, img_h = self.original_image.size
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = self.max_display_size

        scale_w = canvas_w / img_w
        scale_h = canvas_h / img_h
        self.scale = min(scale_w, scale_h, 1.0)

        new_w = int(img_w * self.scale)
        new_h = int(img_h * self.scale)
        self.offset_x = (canvas_w - new_w) // 2
        self.offset_y = (canvas_h - new_h) // 2

        self.display_image = self.original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def _to_display_coords(self, bbox):
        """将原始图像坐标转换为显示坐标。"""
        xmin, ymin, xmax, ymax = bbox
        return (
            self.offset_x + int(xmin * self.scale),
            self.offset_y + int(ymin * self.scale),
            self.offset_x + int(xmax * self.scale),
            self.offset_y + int(ymax * self.scale),
        )

    def _to_original_coords(self, display_bbox):
        """将显示坐标转换回原始图像坐标。"""
        x1, y1, x2, y2 = display_bbox
        x1 = min(max(0, int((x1 - self.offset_x) / self.scale)), self.original_image.width)
        y1 = min(max(0, int((y1 - self.offset_y) / self.scale)), self.original_image.height)
        x2 = min(max(0, int((x2 - self.offset_x) / self.scale)), self.original_image.width)
        y2 = min(max(0, int((y2 - self.offset_y) / self.scale)), self.original_image.height)
        # 确保 x1 < x2, y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        return (x1, y1, x2, y2)

    def _render(self):
        self.canvas.delete("all")
        if self.display_image is None:
            return

        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image)

        # 绘制标注框提示（绿色虚线）
        if self.anno_bbox:
            dx1, dy1, dx2, dy2 = self._to_display_coords(self.anno_bbox)
            self._draw_dashed_rect(dx1, dy1, dx2, dy2, "#00FF00", "anno_hint")

        # 绘制用户框（红色实线）
        if self.current_bbox:
            dx1, dy1, dx2, dy2 = self.current_bbox
            self.canvas.create_rectangle(dx1, dy1, dx2, dy2, outline="red", width=2, tags="user_box")

        # 更新信息文本
        sample = self.current_sample
        progress = f"{self.index + 1}/{len(self.samples)}"
        self.info_label.config(
            text=(
                f"进度: {progress}  |  "
                f"图片: {sample['image_path'].name}  |  "
                f"标签: {sample['label']}  |  "
                f"已保存: {self.saved_count}  |  已跳过: {self.skipped_count}"
            )
        )

    def _draw_dashed_rect(self, x1, y1, x2, y2, color, tag):
        """用短线段模拟虚线矩形。"""
        dash_len = 6
        # 上边
        for x in range(x1, x2, dash_len * 2):
            self.canvas.create_line(x, y1, min(x + dash_len, x2), y1, fill=color, width=1, tags=tag)
        # 下边
        for x in range(x1, x2, dash_len * 2):
            self.canvas.create_line(x, y2, min(x + dash_len, x2), y2, fill=color, width=1, tags=tag)
        # 左边
        for y in range(y1, y2, dash_len * 2):
            self.canvas.create_line(x1, y, x1, min(y + dash_len, y2), fill=color, width=1, tags=tag)
        # 右边
        for y in range(y1, y2, dash_len * 2):
            self.canvas.create_line(x2, y, x2, min(y + dash_len, y2), fill=color, width=1, tags=tag)

    def _on_mouse_down(self, event):
        self.canvas.delete("user_box")
        self.start_x = event.x
        self.start_y = event.y
        self.current_bbox = None
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="red", width=2, tags="user_box",
        )

    def _on_mouse_drag(self, event):
        if self.rect_id is None:
            return
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def _on_mouse_up(self, event):
        if self.rect_id is None:
            return
        x1, y1, x2, y2 = self.canvas.coords(self.rect_id)
        self.current_bbox = (int(x1), int(y1), int(x2), int(y2))
        self._render()

    def _on_key(self, event):
        key = event.char.lower()
        if key == "q":
            self._on_close()
        elif key == "r":
            self._reset_box()
        elif key == "n":
            self._skip_sample()
        elif key == "s":
            self._save_sample()

    def _reset_box(self):
        self.canvas.delete("user_box")
        self.current_bbox = None
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self._render()

    def _save_sample(self):
        if self.current_bbox is not None:
            # 使用用户手动画的框
            orig_bbox = self._to_original_coords(self.current_bbox)
            source = "manual"
        elif self.anno_bbox is not None:
            # 未手动画框时，使用 DIOR 原始标注框自动遮蔽
            orig_bbox = self.anno_bbox
            source = "auto"
        else:
            messagebox.showwarning("未画框", "请先使用鼠标左键拖拽画出目标区域")
            return

        x1, y1, x2, y2 = orig_bbox
        if x2 - x1 < 5 or y2 - y1 < 5:
            messagebox.showwarning("框太小", "框选区域太小，请重新画框")
            return

        label = self.current_sample["label"]
        out_path = save_masked_image(
            self.current_sample["image_path"],
            label,
            orig_bbox,
            self.output_dir,
            suffix=f"_{self.index:06d}",
        )
        self.saved_count += 1
        self._log_saved(self.current_sample, orig_bbox, out_path, source=source)
        self._save_progress()
        print(f"已保存: {out_path}  标签: {label}  来源: {source}  遮挡区域: {orig_bbox}")
        self._next_sample()

    def _log_saved(self, sample, bbox, out_path, source: str = "manual"):
        with open(self.saved_log_path, "a", encoding="utf-8") as f:
            f.write(f"{out_path}\t{sample['image_path']}\t{sample['label']}\t{bbox}\t{source}\n")

    def _log_skipped(self, sample):
        with open(self.skipped_log_path, "a", encoding="utf-8") as f:
            f.write(f"{sample['image_path']}\t{sample['label']}\t{sample['anno_bbox']}\n")

    def _skip_sample(self):
        """跳过当前样本，不保存任何图片。"""
        self.skipped_count += 1
        self._log_skipped(self.current_sample)
        self._save_progress()
        print(f"已跳过: {self.current_sample['image_path']}  标签: {self.current_sample['label']}")
        self._next_sample()

    def _next_sample(self):
        self.index += 1
        self._load_sample()

    def _on_close(self):
        summary = (
            f"已保存: {self.saved_count}\n"
            f"已跳过: {self.skipped_count}\n"
            f"总计: {self.saved_count + self.skipped_count}/{len(self.samples)}"
        )
        if messagebox.askyesno("确认退出", f"确定要退出吗？\n\n{summary}"):
            print("\n" + "=" * 50)
            print("手动标注统计:")
            print("=" * 50)
            print(summary)
            print(f"保存日志: {self.saved_log_path}")
            print(f"跳过日志: {self.skipped_log_path}")
            self.root.destroy()

    def run(self):
        if not self.samples:
            messagebox.showerror("无数据", "未找到有效的 DIOR 标注，请检查 --dior-root 路径")
            return
        self._load_sample()
        self.root.mainloop()


def run_auto_mode(dior_root: Path, output_dir: Path, split: str | None = None, limit: int | None = None):
    """自动模式：使用 DIOR 原始标注框生成遮挡样本，无需 GUI。"""
    samples = build_sample_list(dior_root, split=split)
    if not samples:
        print("没有可处理的样本")
        return

    if limit:
        samples = samples[:limit]
        print(f"限制处理前 {limit} 个对象")

    output_dir = Path(output_dir)
    if output_dir.exists():
        # 为了避免污染，自动模式默认会清空输出目录；如需保留请提前备份
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for idx, sample in enumerate(samples):
        out_path = save_masked_image(
            sample["image_path"],
            sample["label"],
            sample["anno_bbox"],
            output_dir,
            suffix=f"_{idx:06d}",
        )
        saved += 1
        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{len(samples)} ...")

    print(f"\n完成，共保存 {saved} 张遮挡样本到 {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DIOR 数据集准备工具：画框 -> 遮挡 -> 保存到 database")
    parser.add_argument("--dior-root", type=str, required=True, help="DIOR 数据集根目录（包含 JPEGImages 和 Annotations）")
    parser.add_argument("--output", type=str, default="./database", help="输出目录，默认 ./database")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val", "test", "trainval"], help="使用 ImageSets/Main 中的指定 split，默认 all")
    parser.add_argument("--auto", action="store_true", help="自动模式：使用 DIOR 原始标注框生成遮挡样本，无需 GUI")
    parser.add_argument("--clear-output", action="store_true", help="手动模式：启动前清空输出目录，确保 database 只含本次手动标注")
    parser.add_argument("--resume", action="store_true", default=True, help="手动模式：检测到 progress.json 时从上次位置继续（默认开启）")
    parser.add_argument("--no-resume", action="store_true", help="手动模式：忽略 progress.json，从头开始")
    parser.add_argument("--limit", type=int, default=None, help="自动模式下最多处理多少个对象")
    parser.add_argument("--max-width", type=int, default=1280, help="显示窗口最大宽度")
    parser.add_argument("--max-height", type=int, default=800, help="显示窗口最大高度")
    args = parser.parse_args()

    if args.auto:
        run_auto_mode(
            dior_root=Path(args.dior_root),
            output_dir=Path(args.output),
            split=args.split if args.split != "all" else None,
            limit=args.limit,
        )
    else:
        tool = DIORPreparationTool(
            dior_root=Path(args.dior_root),
            output_dir=Path(args.output),
            split=args.split if args.split != "all" else None,
            max_display_size=(args.max_width, args.max_height),
            clear_output=args.clear_output,
            resume=args.resume and not args.no_resume,
        )
        tool.run()


if __name__ == "__main__":
    main()
