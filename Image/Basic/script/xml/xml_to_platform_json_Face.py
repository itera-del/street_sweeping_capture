import argparse
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
    create_folder(args.platform_json_dir)

    xml_list = np.array(os.listdir(args.xml_dir))
    xml_list = xml_list[[xml.endswith('.xml') for xml in xml_list]]
    xml_list.sort()

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    
    platform_json_writer = PlatformJsonWriter()

    for idx in tqdm(range(len(xml_list))):
        # xml
        xml_name = xml_list[idx]
        xml_path = os.path.join(args.xml_dir, xml_name)

        # jpg
        jpg_name = str(xml_name)[:-4] + '.jpg'
        assert jpg_name in jpg_list

        # read xml
        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点

        # img_width & img_height
        img_width = int(root.find('size').find('width').text)
        img_height = int(root.find('size').find('height').text)

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
            elif "face" in classname:      # 'front_face','side_face','face_occlusion'
                bndbox.extend(["face"])
            else:
                print(classname)
                raise Exception

            # 增加属性
            if args.add_face_attribute:
                if "face" in classname:  
                    if classname == "front_face":
                        bndbox.extend([[{"name": "zhedang", "value": "False"}, {"name": "dajiaodu", "value": "False"}]])
                    elif classname == "side_face":
                        bndbox.extend([[{"name": "zhedang", "value": "False"}, {"name": "dajiaodu", "value": "True"}]])
                    elif classname == "face_occlusion":
                        bndbox.extend([[{"name": "zhedang", "value": "True"}, {"name": "dajiaodu", "value": "False"}]])
                    else:
                        print(classname)
                        raise Exception
            rect_list.append(bndbox)
        
        json_name = str(xml_name)[:-4] +  '.json'
        out_json_path = os.path.join(args.platform_json_dir, json_name)
        platform_json_writer.write_json(img_width, img_height, jpg_name, out_json_path, frame_num=idx, rect_list=rect_list)

    out_meta_json_path = os.path.join(args.platform_json_dir, 'meta.json')
    platform_json_writer.write_meta_json(out_meta_json_path, task_name="{}_{}".format(args.task_name, len(xml_list)), label_list=args.label_list)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--jpg_dir', type=str)  
    parser.add_argument('--xml_dir', type=str )  
    parser.add_argument('--platform_json_dir',type=str ) 
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/data/image/RM_Face/original/temp/face_occlusion/005"
    args.jpg_dir = os.path.join(args.input_dir, "JPEGImages/")
    args.xml_dir = os.path.join(args.input_dir, "Annotations_Face_wZhedang_MMGroundingDINO_NMS/")
    args.platform_json_dir = os.path.join(args.input_dir, "Json/")

    args.task_name = "Capture_Face"
    args.label_list =  ['face', 'person','bicyclist','motorcyclist',]
    args.add_face_attribute = True

    xml_2_platform_json(args)