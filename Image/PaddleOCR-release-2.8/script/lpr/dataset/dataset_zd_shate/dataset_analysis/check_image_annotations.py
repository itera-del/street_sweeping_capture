#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像标签匹配检查工具

该脚本用于检查图像目录中的每张图片是否都有对应的XML标签文件。
如果发现没有对应标签的图片，会将其复制到review目录中供人工检查。

作者: AI Assistant
日期: 2024
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Tuple

# 配置日志记录
def setup_logging():
    """配置Winston风格的日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('image_review.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_image_files(image_dir: str) -> List[str]:
    """
    获取图像目录中所有.jpg文件
    
    Args:
        image_dir: 图像目录路径
        
    Returns:
        包含所有.jpg文件名的列表
    """
    logger = logging.getLogger(__name__)
    try:
        image_path = Path(image_dir)
        if not image_path.exists():
            logger.error(f"图像目录不存在: {image_dir}")
            return []
        
        # 获取所有.jpg文件
        image_files = [f.name for f in image_path.glob("*.jpg")]
        logger.info(f"在图像目录中找到 {len(image_files)} 个.jpg文件")
        return image_files
    except Exception as e:
        logger.error(f"获取图像文件时出错: {e}")
        return []

def get_xml_files(annotation_dir: str) -> List[str]:
    """
    获取标签目录中所有.xml文件
    
    Args:
        annotation_dir: 标签目录路径
        
    Returns:
        包含所有.xml文件名的列表
    """
    logger = logging.getLogger(__name__)
    try:
        annotation_path = Path(annotation_dir)
        if not annotation_path.exists():
            logger.error(f"标签目录不存在: {annotation_dir}")
            return []
        
        # 获取所有.xml文件
        xml_files = [f.name for f in annotation_path.glob("*.xml")]
        logger.info(f"在标签目录中找到 {len(xml_files)} 个.xml文件")
        return xml_files
    except Exception as e:
        logger.error(f"获取XML文件时出错: {e}")
        return []

def find_missing_annotations(image_files: List[str], xml_files: List[str]) -> List[str]:
    """
    查找没有对应XML标签的图像文件
    
    Args:
        image_files: 图像文件名列表
        xml_files: XML文件名列表
        
    Returns:
        没有对应标签的图像文件名列表
    """
    logger = logging.getLogger(__name__)
    
    # 将XML文件名转换为对应的图像文件名（去掉.xml后缀，添加.jpg后缀）
    xml_image_names = {xml_file.replace('.xml', '.jpg') for xml_file in xml_files}
    
    # 找出没有对应XML标签的图像
    missing_annotations = [img_file for img_file in image_files if img_file not in xml_image_names]
    
    logger.info(f"发现 {len(missing_annotations)} 个图像没有对应的XML标签")
    return missing_annotations

def copy_images_to_review(missing_images: List[str], 
                         source_dir: str, 
                         target_dir: str) -> int:
    """
    将没有标签的图像复制到review目录
    
    Args:
        missing_images: 没有标签的图像文件名列表
        source_dir: 源图像目录
        target_dir: 目标review目录
        
    Returns:
        成功复制的图像数量
    """
    logger = logging.getLogger(__name__)
    
    # 创建目标目录（如果不存在）
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"目标目录已创建/确认: {target_dir}")
    
    copied_count = 0
    
    for image_name in missing_images:
        try:
            source_path = Path(source_dir) / image_name
            target_path = Path(target_dir) / image_name
            
            if source_path.exists():
                shutil.copy2(source_path, target_path)
                copied_count += 1
                logger.info(f"已复制: {image_name}")
            else:
                logger.warning(f"源文件不存在: {source_path}")
        except Exception as e:
            logger.error(f"复制文件 {image_name} 时出错: {e}")
    
    logger.info(f"成功复制 {copied_count} 个图像到review目录")
    return copied_count

def main():
    """主函数：执行图像标签匹配检查流程"""
    logger = setup_logging()
    
    # 定义路径
    IMAGE_DIR = "/yuanhuan/data/image/RM_ANPR/original/zd/Shate/Shate_crop_scale_padding/DIFFSTE/original_2094_Shate/JPEGImages"
    ANNOTATION_DIR = "/yuanhuan/data/image/RM_ANPR/original/zd/Shate/Shate_crop_scale_padding/DIFFSTE/original_2094_Shate/Annotations"
    REVIEW_DIR = "/yuanhuan/data/image/RM_ANPR/original/zd/Shate/Shate_crop_scale_padding/DIFFSTE/original_2094_Shate/JPEGImages_review"
    
    logger.info("开始图像标签匹配检查")
    logger.info(f"图像目录: {IMAGE_DIR}")
    logger.info(f"标签目录: {ANNOTATION_DIR}")
    logger.info(f"Review目录: {REVIEW_DIR}")
    
    # 获取图像和标签文件列表
    image_files = get_image_files(IMAGE_DIR)
    xml_files = get_xml_files(ANNOTATION_DIR)
    
    if not image_files:
        logger.error("未找到图像文件，程序退出")
        return
    
    if not xml_files:
        logger.warning("未找到XML标签文件，所有图像都将被复制到review目录")
        missing_images = image_files
    else:
        # 查找没有标签的图像
        missing_images = find_missing_annotations(image_files, xml_files)
    
    if missing_images:
        # 复制没有标签的图像到review目录
        copied_count = copy_images_to_review(missing_images, IMAGE_DIR, REVIEW_DIR)
        logger.info(f"检查完成！共发现 {len(missing_images)} 个没有标签的图像，成功复制 {copied_count} 个到review目录")
    else:
        logger.info("检查完成！所有图像都有对应的XML标签文件")
    
    # 输出统计信息
    logger.info("=" * 50)
    logger.info("统计信息:")
    logger.info(f"总图像数量: {len(image_files)}")
    logger.info(f"总标签数量: {len(xml_files)}")
    logger.info(f"缺失标签的图像数量: {len(missing_images)}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()