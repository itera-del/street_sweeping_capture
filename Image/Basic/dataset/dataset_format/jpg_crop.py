import argparse
import cv2
import numpy as np
import os
import sys 
from tqdm import tqdm
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/demo/Image')
from Basic.script.json.platform_json_write import PlatformJsonWriter


def crop_objects_from_images(args):
    # 创建输出目录
    os.makedirs(args.crop_jpg_dir, exist_ok=True)

    # 目标类别列表
    target_classes = ['car', 'bus', 'truck', 'tricyclist']
    
    xml_list = np.array(os.listdir(args.xml_dir))
    xml_list = xml_list[[xml.endswith('.xml') for xml in xml_list]]
    xml_list.sort()

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    
    for idx in tqdm(range(len(xml_list))):
        # xml
        xml_name = xml_list[idx]
        xml_path = os.path.join(args.xml_dir, xml_name)

        # jpg
        jpg_name = str(xml_name)[:-4] + '.jpg'
        jpg_path = os.path.join(args.jpg_dir, jpg_name)
        if jpg_name not in jpg_list:
            continue

        # 读取图像
        img = cv2.imread(jpg_path)
        if img is None:
            print("无法读取图像: {}".format(jpg_path))
            continue

        # 读取 XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 处理每个目标
        obj_count = 0
        for object in root.findall('object'):
            # 获取类别名称
            classname = str(object.find('name').text)

            # 检查是否是目标类别
            if classname not in target_classes:
                continue

            # 获取边界框
            bbox = object.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))

            # 检查目标大小
            width = xmax - xmin
            height = ymax - ymin
            if width <= 20 or height <= 20:
                continue

            # 裁剪图像
            crop_img = img[ymin:ymax, xmin:xmax]
            
            # 保存裁剪的图像
            crop_name = "{}_{}_{}".format(jpg_name[:-4], classname, obj_count) + '.jpg'
            crop_path = os.path.join(args.crop_jpg_dir, crop_name)
            cv2.imwrite(crop_path, crop_img)
            obj_count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.jpg_dir = "/defaultShare/0nfs_defaultShare/AEBS_BYD_ACC_DATA/AEB_taxt_topicOptimization/accident_202504/find_img/h264_analyze/h264_data/find_img"
    args.xml_dir = "/defaultShare/0nfs_defaultShare/AEBS_BYD_ACC_DATA/AEB_taxt_topicOptimization/accident_202504/find_img/h264_analyze_gdino_v1.1.4_pretrain_v1.0/h264_xml/xml_adas_det_v1.1.4_pretrain_v1.0_nms/find_img"
    args.crop_jpg_dir = "//defaultShare/0nfs_defaultShare/AEBS_BYD_ACC_DATA/AEB_taxt_topicOptimization/accident_202504/find_img/crop_img"

    crop_objects_from_images(args)