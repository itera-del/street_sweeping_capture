import base64
import urllib
import requests
import os
import glob
import csv
import re

API_KEY = "QZdrf6tqWRjuyJjlJx1qdDSB"
SECRET_KEY = "S3MgDACxM73E7sNPftALCFPhLYraWwxH"

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

def process_image(image_path: str) -> str:
    """处理单张图片并返回OCR结果"""
    try:
        url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic?access_token=" + get_access_token()

        with open(image_path, 'rb') as f:
            img = base64.b64encode(f.read())
        
        params = {"image": img}
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=params)
        result = response.json()
        
        if 'words_result' in result and len(result['words_result']) > 0:
            return result['words_result'][0]['words']
        return ""
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")
        return ""

def main():
    # image_dir = '/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_crop/test_data/Images_clear/'
    # output_csv = '/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_crop/test_data/baidu_ocr_images_clear_results.csv'
    image_dir = '/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_crop/test_data/Images_blurred/'
    output_csv = '/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_crop/test_data/baidu_ocr_Images_blurred_results.csv'
    
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
            ocr_result = process_image(image_path)
            accuracy = calculate_accuracy(ocr_result, label)
            writer.writerow([image_path, label, ocr_result, f"{accuracy:.2f}"])
            print(f"标签: {label}, OCR结果: {ocr_result}, 准确率: {accuracy:.2f}")

def get_file_content_as_base64(path, urlencoded=False):
    """
    获取文件base64编码
    :param path: 文件路径
    :param urlencoded: 是否对结果进行urlencoded 
    :return: base64编码信息
    """
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

if __name__ == '__main__':
    main()
