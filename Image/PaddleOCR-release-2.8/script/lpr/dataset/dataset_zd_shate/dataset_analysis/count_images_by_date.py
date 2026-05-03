#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统计每个日期目录下 JPEGImages 文件夹中的 .jpg 图片数量。

用法示例：
1) 自动发现所有日期目录：
   python count_images_by_date.py \
     --base_dir /yuanhuan/data/image/RM_ANPR/original/zd/Shate/Shate

2) 指定日期目录列表：
   python count_images_by_date.py \
     --base_dir /yuanhuan/data/image/RM_ANPR/original/zd/Shate/Shate \
     --date_names shate_20250115 shate_20250116
"""

import argparse
import csv
import datetime as dt
import logging
import os
import sys
from typing import List, Tuple


# ----------------------------- 日志配置 -----------------------------
# 使用 Python logging 作为日志记录工具，细粒度记录关键流程
logger = logging.getLogger("count_images_by_date")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setLevel(logging.INFO)
_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_handler.setFormatter(_formatter)
# 避免重复添加 handler（在交互式环境中多次运行时）
if not logger.handlers:
    logger.addHandler(_handler)


def is_valid_date_dir(name: str) -> bool:
    """判断目录名是否符合 shate_YYYYMMDD 形式。
    - 约束：前缀为 'shate_' 且后缀为 8 位数字
    """
    if not name.startswith("shate_"):
        return False
    suffix = name[len("shate_") :]
    return len(suffix) == 8 and suffix.isdigit()


def discover_date_dirs(base_dir: str) -> List[str]:
    """自动发现 base_dir 下的日期目录列表。
    - 仅返回一级子目录且满足 is_valid_date_dir 条件
    """
    logger.info("开始自动发现日期目录：%s", base_dir)
    if not os.path.isdir(base_dir):
        logger.error("基路径不存在或不是目录：%s", base_dir)
        return []

    try:
        sub_entries = os.listdir(base_dir)
    except Exception as exc:
        logger.exception("列举目录失败：%s", exc)
        return []

    date_dirs = [d for d in sub_entries if os.path.isdir(os.path.join(base_dir, d)) and is_valid_date_dir(d)]
    date_dirs.sort()
    logger.info("发现日期目录数量：%d", len(date_dirs))
    return date_dirs


def count_jpg_in_jpegimages(date_dir_path: str) -> int:
    """统计给定日期目录下 JPEGImages 目录中的 .jpg 文件数量。
    - 仅统计后缀为 .jpg 或 .JPG 的文件
    - 不做递归统计
    """
    jpegimages_dir = os.path.join(date_dir_path, "JPEGImages")
    if not os.path.isdir(jpegimages_dir):
        logger.warning("未找到 JPEGImages 目录：%s", jpegimages_dir)
        return 0

    try:
        files = os.listdir(jpegimages_dir)
    except Exception as exc:
        logger.exception("列举 JPEGImages 目录失败：%s", exc)
        return 0

    count = sum(1 for f in files if f.endswith(".jpg") or f.endswith(".JPG"))
    return count


def build_results(base_dir: str, date_names: List[str]) -> List[Tuple[str, int]]:
    """根据日期列表统计每个日期的 .jpg 数量并返回结果表。
    - 返回列表元素为 (date_name, count)
    """
    results: List[Tuple[str, int]] = []

    for date_name in date_names:
        date_dir_path = os.path.join(base_dir, date_name)
        if not os.path.isdir(date_dir_path):
            logger.error("日期目录不存在：%s", date_dir_path)
            results.append((date_name, 0))
            continue

        count = count_jpg_in_jpegimages(date_dir_path)
        logger.info("日期 %s | .jpg 数量：%d", date_name, count)
        results.append((date_name, count))

    return results


def write_csv(results: List[Tuple[str, int]], output_csv: str) -> None:
    """将统计结果写入 CSV 文件，并在末尾写入总计行。"""
    # 计算总量；在写行前计算以便一起输出
    total_count = sum(count for _, count in results)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date_name", "jpg_count"])  # 表头
        for date_name, count in results:
            writer.writerow([date_name, count])
        # 追加总计行，便于后续汇总分析
        writer.writerow(["TOTAL", total_count])
    logger.info("结果已写入：%s（含 TOTAL=%d）", output_csv, total_count)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="统计每个日期目录下 JPEGImages 中 .jpg 文件数量")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/yuanhuan/data/image/RM_ANPR/original/zd/Shate/Shate",
        help="数据集根目录，包含多个 shate_YYYYMMDD 子目录",
    )
    parser.add_argument(
        "--date_names",
        type=str,
        nargs="*",
        default=None,
        help="指定日期目录名列表（空则自动发现）",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="输出 CSV 路径（默认写入 base_dir 下 counts_YYYYmmdd_HHMMSS.csv）",
    )
    return parser.parse_args()


def main() -> None:
    """入口函数：解析参数、发现目录、统计并输出。"""
    args = parse_args()

    # 决定日期目录列表
    if args.date_names is None or len(args.date_names) == 0:
        date_names = discover_date_dirs(args.base_dir)
    else:
        date_names = list(args.date_names)
        logger.info("使用指定的日期目录列表，共 %d 个", len(date_names))

    if len(date_names) == 0:
        logger.error("未找到任何日期目录，程序结束。")
        return

    # 统计
    results = build_results(args.base_dir, date_names)

    # 打印汇总（含总量）
    print("\n统计结果：")
    for date_name, count in results:
        print(f"{date_name}: {count}")
    total_count = sum(count for _, count in results)  # 计算总量
    print(f"TOTAL: {total_count}")  # 控制台输出总量，便于快速查看
    logger.info("全部日期总计 .jpg 数量：%d", total_count)

    # 输出 CSV
    if args.output_csv is None or args.output_csv == "":
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = os.path.join(args.base_dir, f"counts_{ts}.csv")
    else:
        output_csv = args.output_csv

    write_csv(results, output_csv)


if __name__ == "__main__":
    main() 