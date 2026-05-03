import argparse
import cv2
import numpy as np
import os
import sys 
from tqdm import tqdm
import xml.etree.ElementTree as ET

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/')
sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.utils.folder_tools import *
from Basic.script.json.platform_json_write import PlatformJsonWriter


def xml_2_platform_json(args):
    
    # mkdir 
    os.makedirs(args.platform_json_dir, exist_ok=True)

    xml_list = np.array(os.listdir(args.xml_dir))
    # xml_list = xml_list[[xml.endswith('.xml') for xml in xml_list]]
    xml_list.sort()

    jpg_list = np.array(os.listdir(args.jpg_dir))
    # jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    
    platform_json_writer = PlatformJsonWriter()

    for idx in tqdm(range(len(xml_list))):
        # xml
        xml_name = xml_list[idx]
        xml_path = os.path.join(args.xml_dir, xml_name)

        # jpg
        jpg_name = str(xml_name)[:-4] + '.jpg'
        jpg_path = os.path.join(args.jpg_dir, jpg_name)
        if jpg_name not in jpg_list:
            continue

        # read xml
        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点

        # img_width & img_height
        img = cv2.imread(jpg_path)
        img_width = img.shape[1]
        img_height = img.shape[0]
        # img_width = int(root.find('size').find('width').text)
        # img_height = int(root.find('size').find('height').text)
        # print(img_width, img_height)

        # 标签检测和标签转换
        rect_list = []
        for object in root.findall('object'):
            # name
            classname = str(object.find('name').text)

            # change name
            if classname not in args.label_list:
                classname = args.label_list[0]
            if classname not in args.label_list:
                print("Ignore: {}".format(classname))
                continue
            assert classname in args.label_list 

            # bbox
            bbox = object.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text))
                bndbox.append(cur_pt)
            
            bndbox.extend([classname])
            rect_list.append(bndbox)
        
        json_name = str(xml_name)[:-4] + '.json'
        out_json_path = os.path.join(args.platform_json_dir, json_name)
        platform_json_writer.write_json(img_width, img_height, jpg_name, out_json_path, frame_num=idx, rect_list=rect_list)

    out_meta_json_path = os.path.join(args.platform_json_dir, 'meta.json')
    platform_json_writer.write_meta_json(out_meta_json_path, task_name="{}_{}".format(args.task_name, len(xml_list)), label_list=args.label_list)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Capture
    args.input_dir = "/yuanhuan/data/image/RM_ANPR/original/RM_Character"
    args.jpg_dir = os.path.join(args.input_dir, "JPEGImages/")
    args.xml_dir = os.path.join(args.input_dir, "Annotations/")
    args.platform_json_dir = os.path.join(args.input_dir, "Jsons/")
    args.task_name = "Capture"
    args.label_list = ["char"]

    xml_2_platform_json(args)