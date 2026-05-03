import argparse
import cv2
import numpy as np
import os
import sys 
from tqdm import tqdm
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo/Image')
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
        assert jpg_name in jpg_list

        # read xml
        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点

        # img_width & img_height
        img = cv2.imread(jpg_path)
        img_width = img.shape[1]
        img_height = img.shape[0]
        # img_width = int(root.find('size').find('width').text)
        # img_height = int(root.find('size').find('height').text)

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
        
            if classname in args.label_list:
                bndbox.extend([classname])
            elif classname == "license" or "license" in classname:      # "license_plate" & "license_plate_车牌号"
                bndbox.extend(["license"])
            else:
                print(classname)
                raise Exception

            # 车牌，增加车牌属性
            if args.add_license_attribute:
                if classname == "license":
                    bndbox.extend([[{"name": "number", "value": ""}, {"name": "status", "value": "n"}]])
                elif "license" in classname:
                    license_plate_ocr = classname.split('license_')[1]
                    if license_plate_ocr == "123456":
                        bndbox.extend([[{"name": "number", "value": ""}, {"name": "status", "value": "n"}]])
                    else:
                        bndbox.extend([[{"name": "number", "value": license_plate_ocr}, {"name": "status", "value": "n"}]])
            rect_list.append(bndbox)
        
        json_name = str(xml_name)[:-4] + '.json'
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
        default="/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_new_style/uae_collect_20241113",
    )
    parser.add_argument(
        "--meta_template_path",
        type=str,
        default="/yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/meta_plat_capture.json",
    )    
    parser.add_argument("--jpg_name", type=str, default="JPEGImages/")
    parser.add_argument("--xml_name", type=str, default="Annotations/")
    parser.add_argument("--json_name", type=str, default="Jsons/")
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
    args.label_list =  ['garbage', 'license']
    args.add_license_attribute = True

    xml_2_platform_json(args)