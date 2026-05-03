# -*- coding: utf-8 -*-

import base64
import json
import types
import os
import glob
import csv
import re
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.ocr.v20181119 import ocr_client, models

def extract_label(filename: str) -> str:
    """从文件名中提取标签（最后一个数字）"""
    match = re.search(r'_(\d+)\.jpg$', filename)
    if match:
        return match.group(1)
    return ""

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

def process_image(client: ocr_client.OcrClient, image_path: str) -> str:
    """处理单张图片并返回OCR结果"""
    try:
        req = models.GeneralAccurateOCRRequest()
        
        with open(image_path, 'rb') as f:
            img = base64.b64encode(f.read())
        
        params = {
            "ImageBase64": "data:image/jpeg;base64," + img.decode('utf-8')
        }
        req.from_json_string(json.dumps(params))
        
        resp = client.GeneralAccurateOCR(req)
        result = json.loads(resp.to_json_string())
        
        if 'TextDetections' in result and len(result['TextDetections']) > 0:
            return result['TextDetections'][0]['DetectedText']
        return ""
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")
        return ""

def main():
    try:
        # 实例化认证对象（从环境变量读取，勿将密钥写入仓库）
        secret_id = os.environ.get("TENCENTCLOUD_SECRET_ID")
        secret_key = os.environ.get("TENCENTCLOUD_SECRET_KEY")
        if not secret_id or not secret_key:
            raise RuntimeError("请设置环境变量 TENCENTCLOUD_SECRET_ID 与 TENCENTCLOUD_SECRET_KEY")
        cred = credential.Credential(secret_id, secret_key)
        
        # 实例化http选项
        httpProfile = HttpProfile()
        httpProfile.endpoint = "ocr.tencentcloudapi.com"
        
        # 实例化client选项
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        
        # 实例化client对象
        client = ocr_client.OcrClient(cred, "", clientProfile)
        
        # 设置图片目录和输出CSV文件路径
        # image_dir = '/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_crop/test_data/Images_clear/'
        # output_csv = '/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_crop/test_data/tengcent_ocr_images_clear_results.csv'
        image_dir = '/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_crop/test_data/Images_blurred/'
        output_csv = '/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_crop/test_data/tengcent_ocr_images_blurred_results.csv'
        
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
                label = extract_label(image_path)
                ocr_result = process_image(client, image_path)
                accuracy = calculate_accuracy(ocr_result, label)
                writer.writerow([image_path, label, ocr_result, f"{accuracy:.2f}"])
                print(f"标签: {label}, OCR结果: {ocr_result}, 准确率: {accuracy:.2f}")

    except TencentCloudSDKException as err:
        print(err)

if __name__ == '__main__':
    main()