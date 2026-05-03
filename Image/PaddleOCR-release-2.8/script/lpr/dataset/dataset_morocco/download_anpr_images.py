import pandas as pd
import requests
import os
from urllib.parse import unquote
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_log.txt'),
        logging.StreamHandler()
    ]
)

def create_directory(path):
    """创建目录（如果不存在）"""
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"创建目录: {path}")

def download_image(url, save_path):
    """下载图片并保存到指定路径"""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"成功下载图片: {save_path}")
            return True
        else:
            logging.error(f"下载失败，状态码: {response.status_code}, URL: {url}")
            return False
    except Exception as e:
        logging.error(f"下载出错: {str(e)}, URL: {url}")
        return False

def main():
    # Excel文件路径
    excel_file = "/yuanhuan/data/image/RM_ANPR/original/Morocco/Morocco/ANPR vehicle snap record-Untreated-20250301000000-20250422235959.xlsx"
    
    # 创建基础保存目录
    base_save_dir = "/yuanhuan/data/image/RM_ANPR/original/Morocco/Morocco/morocco_20250301_20250422/JPEGImages"
    create_directory(base_save_dir)
    
    try:
        # 读取Excel文件
        logging.info(f"开始读取Excel文件: {excel_file}")
        df = pd.read_excel(excel_file, engine='openpyxl')
        
        # 确保'ANPR car pic（全图）'列存在
        if 'ANPR car pic（全图）' not in df.columns:
            logging.error("Excel文件中未找到'ANPR car pic（全图）'列")
            return
        
        total_images = len(df)
        successful_downloads = 0
        
        # 遍历每一行并下载图片
        for index, row in df.iterrows():
            image_url = row['ANPR car pic（全图）']
            if pd.isna(image_url):
                logging.warning(f"第 {index + 1} 行的URL为空")
                continue
                
            # 从URL中提取文件名
            file_name = os.path.basename(unquote(image_url)).split('fileId=')[-1] + '.jpg'
            
            # 完整的保存路径
            save_path = os.path.join(base_save_dir, file_name)
            
            # 检查文件是否已存在
            if os.path.exists(save_path):
                logging.info(f"文件已存在，跳过下载: {save_path}")
                successful_downloads += 1
                continue
            
            # 下载图片
            if download_image(image_url, save_path):
                successful_downloads += 1
            
            # 打印进度
            if (index + 1) % 10 == 0:
                logging.info(f"进度: {index + 1}/{total_images}")
        
        logging.info(f"下载完成！成功下载 {successful_downloads}/{total_images} 张图片")
        
    except Exception as e:
        logging.error(f"处理过程中出错: {str(e)}")

if __name__ == "__main__":
    main()