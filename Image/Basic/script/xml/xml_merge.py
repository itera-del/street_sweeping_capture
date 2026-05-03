import argparse
import cv2
import numpy as np
import os
import sys 
from tqdm import tqdm
import xml.etree.ElementTree as ET

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.script.xml.xml_write import write_xml


def xml_merge(args):

    # mkdir 
    create_folder(args.output_xml_dir)

    # image init 
    img_list = np.array(os.listdir(args.img_dir))
    img_list = img_list[[jpg.endswith(args.suffix) for jpg in img_list]]
    img_list.sort()

    for idx in tqdm(range(len(img_list))):
    
        img_name = img_list[idx]
        xml_name = img_name.replace(args.suffix, ".xml")
        img_path = os.path.join(args.img_dir, img_name)
        xml_path = os.path.join(args.xml_dir, xml_name)
        xml_2_path = os.path.join(args.xml_2_dir, xml_name)
        output_xml_path = os.path.join(args.output_xml_dir, xml_name)

        # check
        if not (os.path.exists(xml_path) and os.path.exists(xml_2_path)):
            print(img_path)
            print(xml_path)
            print(xml_2_path)

            # if os.path.exists(img_path):
            #     os.remove(img_path)
            # if os.path.exists(xml_path):
            #     os.remove(xml_path)
            # if os.path.exists(xml_2_path):
            #     os.remove(xml_2_path)
            continue

        # img
        img = cv2.imread(img_path)     

        # show_bboxes
        show_bboxes = {}

        # read xml
        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点
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
            
            # bndbox.append(float(bbox.find('score').text))

            if classname not in show_bboxes:
                show_bboxes[classname] = []
            
            show_bboxes[classname].append(bndbox)

        # read xml
        tree = ET.parse(xml_2_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点
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

            # bndbox.append(float(bbox.find('score').text))
            
            # # plane_name
            # if classname == "license_plate":
            #     if object.find('vehicleic') != None:
            #         vehicleic = str(object.find('vehicleic').text).replace('-', '')
            #         classname += "_" + vehicleic
            #         print(classname)
            #     else:
            #         print(classname)
            # else:
            #     print(classname)
            #     continue
            
            if classname not in show_bboxes:
                show_bboxes[classname] = []
            
            show_bboxes[classname].append(bndbox)

        # save xml
        write_xml(output_xml_path, img_path, show_bboxes, img.shape)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_crop/test",
    )
    parser.add_argument("--jpg_name", type=str, default="Images/")
    parser.add_argument("--xml_name", type=str, default="Xmls_Prelabel_City/")
    parser.add_argument("--xml_2_name", type=str, default="Xmls_Prelabel_Color/")
    parser.add_argument("--xml_out_name", type=str, default="Xmls_Prelabel/")
    args = parser.parse_args()

    args.img_dir = os.path.join(args.input_dir, args.jpg_name)
    args.xml_dir = os.path.join(args.input_dir, args.xml_name)
    args.xml_2_dir = os.path.join(args.input_dir, args.xml_2_name)
    args.output_xml_dir = os.path.join(args.input_dir, args.xml_out_name)
    args.suffix = '.jpg'

    xml_merge(args)