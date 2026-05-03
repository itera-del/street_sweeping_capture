import argparse
import cv2
import numpy as np
import os
import sys 
from tqdm import tqdm
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.script.json.platform_json_write import PlatformJsonWriter
from Image.recognition2d.script.lpr.dataset.dataset_morocco.dataset_dict.dataset_morocco_dict_normal import *


def xml_2_platform_json(args):
    
    # mkdir
    os.makedirs(args.platform_json_dir, exist_ok=True)

    xml_list = np.array(os.listdir(args.xml_dir))
    xml_list.sort()

    jpg_list = np.array(os.listdir(args.jpg_dir))
    
    platform_json_writer = PlatformJsonWriter()

    for idx in tqdm(range(len(xml_list))):
        # xml
        xml_name = xml_list[idx]
        xml_path = os.path.join(args.xml_dir, xml_name)

        # jpg
        jpg_name = str(xml_name)[:-4] + '.jpg'
        jpg_path = os.path.join(args.jpg_dir, jpg_name)
        assert jpg_name in jpg_list

        # read xml
        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点

        # img_width & img_height
        img = cv2.imread(jpg_path)
        img_width = img.shape[1]
        img_height = img.shape[0]

        # 标签检测和标签转换
        rect_list = []
        for object in root.findall('object'):
            # name
            classname = str(object.find('name').text)

            # bbox
            bbox = object.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text))
                bndbox.append(cur_pt)            

            # city
            license_city = "unknown"

            # city
            # city_f
            if classname in city_name_list[1:]:
                bndbox.extend(["license_city"])
                bndbox.extend([[{"name": "license_city", "value": license_city}]])

            # kind
            elif classname in kind_name_list[1:]:
                bndbox.extend(["license_kind"])
                bndbox.extend([[{"name": "license_kind", "value": "null"}]])
            
            # num
            elif classname in num_name_list[1:]:
                bndbox.extend(["license_num"])
                bndbox.extend([[{"name": "license_num", "value": "null"}]])

            # 标签检测
            rect_list.append(bndbox)
        
        json_name = str(xml_name)[:-4] +  '.json'
        out_json_path = os.path.join(args.platform_json_dir, json_name)
        platform_json_writer.write_json(img_width, img_height, jpg_name, out_json_path, frame_num=idx, rect_list=rect_list)

    out_meta_json_path = os.path.join(args.platform_json_dir, 'meta.json')
    platform_json_writer.write_meta_json(
        args.meta_template_path,
        out_meta_json_path,
        task_name=args.task_name,
        data_size=len(jpg_list),
    )


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_crop/test",
    )
    parser.add_argument(
        "--meta_template_path",
        type=str,
        default="/yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_morocco/dataset_xml_json/meta_plat_capture.json",
    )    
    parser.add_argument("--jpg_name", type=str, default="Images/")
    parser.add_argument("--xml_name", type=str, default="Xmls_Prelabel/")
    parser.add_argument("--json_name", type=str, default="Jsons_Prelabel_Platform/")
    args = parser.parse_args()

    args.jpg_dir = os.path.join(args.input_dir, args.jpg_name)
    args.xml_dir = os.path.join(args.input_dir, args.xml_name)
    args.platform_json_dir = os.path.join(args.input_dir, args.json_name)

    args.task_name = "_".join(
        [
            os.path.basename(os.path.dirname(os.path.dirname(args.input_dir))),
            os.path.basename(os.path.dirname(args.input_dir)),
        ]
    )

    xml_2_platform_json(args)