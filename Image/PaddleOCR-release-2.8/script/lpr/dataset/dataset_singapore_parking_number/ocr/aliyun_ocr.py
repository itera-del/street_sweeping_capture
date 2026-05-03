# -*- coding: utf-8 -*-
import os
import sys
import csv
import glob
import re
from typing import List, Tuple

from alibabacloud_ocr_api20210707.client import Client as ocr_api20210707Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_ocr_api20210707 import models as ocr_api_20210707_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient

class Sample:
    def __init__(self):
        pass

    @staticmethod
    def create_client() -> ocr_api20210707Client:
        config = open_api_models.Config(
            # 必填，请确保代码运行环境设置了环境变量 ALIBABA_CLOUD_ACCESS_KEY_ID。,
            access_key_id=os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID'],
            # 必填，请确保代码运行环境设置了环境变量 ALIBABA_CLOUD_ACCESS_KEY_SECRET。,
            access_key_secret=os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET']
        )
        # 默认公网接入地址为"ocr-api.cn-hangzhou.aliyuncs.com"，若您需要使用vpc域名访问，请确保您的ecs建立在杭州region，vpc接入地址为"ocr-api-vpc.cn-hangzhou.aliyuncs.com"
        config.endpoint = f'ocr-api.cn-hangzhou.aliyuncs.com'
        return ocr_api20210707Client(config)

    @staticmethod
    def extract_label(filename: str) -> str:
        """从文件名中提取标签（最后一个数字）"""
        match = re.search(r'_(\d+)\.jpg$', filename)
        if match:
            return match.group(1)
        return ""

    @staticmethod
    def calculate_accuracy(ocr_result: str, label: str) -> float:
        """计算OCR结果与标签的准确率（完全匹配）"""
        if not ocr_result or not label:
            return 0.0
        
        # 清理OCR结果，只保留数字
        ocr_digits = ''.join(filter(str.isdigit, ocr_result))
        
        # 如果OCR结果为空，返回0
        if not ocr_digits:
            return 0.0
            
        # 完全匹配则返回1.0，否则返回0.0
        return 1.0 if ocr_digits == label else 0.0

    @staticmethod
    def process_image(client: ocr_api20210707Client, image_path: str) -> str:
        """处理单张图片并返回OCR结果"""
        try:
            with open(image_path, 'rb') as img:
                recognize_all_text_request = ocr_api_20210707_models.RecognizeAllTextRequest(
                    body=img,
                    # url='https://img.alicdn.com/tfs/TB1q5IeXAvoK1RjSZFNXXcxMVXa-483-307.jpg',
                    type='Advanced'
                    # type='General'
                )
                runtime = util_models.RuntimeOptions()
                resp = client.recognize_all_text_with_options(recognize_all_text_request, runtime)
                return resp.body.data.content
        except Exception as error:
            print(f"处理图片 {image_path} 时出错: {error.message}")
            if hasattr(error, 'data') and error.data:
                print(f"诊断地址: {error.data.get('Recommend')}")
            return ""

    @staticmethod
    def main(
        args: List[str],
    ) -> None:
        client = Sample.create_client()
        image_dir = '/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_crop/test_data/Images_clear/'
        output_csv = '/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_crop/test_data/aliyun_ocr_images_clear_results.csv'
        
        # 获取所有jpg文件
        image_files = glob.glob(os.path.join(image_dir, '*.jpg'))
        image_files.sort()
        total_files = len(image_files)
        
        # 创建CSV文件并写入结果
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_path', 'label', 'ocr_result', 'accuracy'])
            
            for idx, image_path in enumerate(image_files, 1):
                print(f"处理图片 [{idx}/{total_files}]: {image_path}")
                label = Sample.extract_label(image_path)
                ocr_result = Sample.process_image(client, image_path)
                accuracy = Sample.calculate_accuracy(ocr_result, label)
                writer.writerow([image_path, label, ocr_result, f"{accuracy:.2f}"])
                print(f"标签: {label}, OCR结果: {ocr_result}, 准确率: {accuracy:.2f}")

if __name__ == '__main__':
    Sample.main(sys.argv[1:])