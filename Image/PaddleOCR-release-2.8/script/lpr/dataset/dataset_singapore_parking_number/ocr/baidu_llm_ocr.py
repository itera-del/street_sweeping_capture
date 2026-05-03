import requests
import json
import os
import glob
import csv
import re
import base64
import time

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
        url = "https://qianfan.baidubce.com/v2/chat/completions"
        
        # 读取图片并转换为base64
        with open(image_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # 使用base64数据作为URL
        image_url = f"data:image/jpeg;base64,{image_base64}"
        
        payload = json.dumps({
            # "model": "internvl2.5-38b-mpo",
            "model": "ernie-4.5-turbo-vl-32k",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "识别图像中的车位号，返回json数据{'parking_number':'车位号'}。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        },
                    ]
                }
            ]
        })
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer bce-v3/ALTAK-pJDhSiUcmYk7hIDQpP6Ip/a32eaf30f8be794ef5a935a4c104200adcbd4000'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        result = json.loads(response.text)
        
        # 从响应中提取OCR结果
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        return ""
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")
        return ""

def main():
    # 设置图片目录和输出CSV文件路径
    # image_dir = '/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_crop/test_data/Images_clear/'
    # output_csv = '/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_crop/test_data/baidu_llm_ernie-4.5-turbo-vl-32k_ocr_images_clear_results.csv'
    image_dir = '/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_crop/test_data/Images_blurred/'
    output_csv = '/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_crop/test_data/baidu_llm_ernie-4.5-turbo-vl-32k_ocr_images_blurred_results.csv'
    
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
            # 添加延时以避免请求过快
            time.sleep(1)

if __name__ == '__main__':
    main()