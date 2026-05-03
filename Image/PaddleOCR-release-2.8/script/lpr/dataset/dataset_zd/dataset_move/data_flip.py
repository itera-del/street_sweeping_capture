import argparse
import cv2
import numpy as np
import os
import shutil
import sys
from tqdm import tqdm
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.script.xml.xml_write import write_xml

def data_flip(args):

    # jpg_list
    jpg_list = np.array(os.listdir(args.jpg_dir))

    for idy in tqdm(range(len(jpg_list))):
        
        jpg_name = jpg_list[idy]
        jpg_path = os.path.join(args.jpg_dir, jpg_name)
        xml_name = jpg_name[:-4] + '.xml'
        xml_path = os.path.join(args.xml_dir, xml_name)

        # flip_bool
        flip_bool = False
        for flip_id in args.flip_img_id_list:
            if flip_id in jpg_name:
                flip_bool = True
                
        if flip_bool:
            # read image
            img = cv2.imread(jpg_path)
            img = cv2.flip(img, 1)
            img_shape = img.shape
            cv2.imwrite(jpg_path, img)

            # read xml
            tree = ET.parse(
                xml_path
            )  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
            root = tree.getroot()  # 获取根节点

            # 标签检测和标签转换
            xml_bboxes = {}
            for object in root.findall("object"):
                # name
                classname = str(object.find("name").text)

                # bbox
                bbox = object.find("bndbox")
                pts = ["xmin", "ymin", "xmax", "ymax"]
                bndbox = []
                for i, pt in enumerate(pts):
                    cur_pt = int(float(bbox.find(pt).text))
                    bndbox.append(cur_pt)
                
                bndbox_flip = bndbox.copy()
                bndbox_flip[0] = img_shape[1] - bndbox[2]
                bndbox_flip[2] = img_shape[1] - bndbox[0]
                # print(bndbox_flip)

                if classname not in xml_bboxes:
                    xml_bboxes[classname] = []              
                xml_bboxes[classname].append(bndbox_flip)
            
            write_xml(xml_path, jpg_path, xml_bboxes, img_shape)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_new_style/uae_20241108/") 
    parser.add_argument("--jpg_subdir", type=str, default="JPEGImages")
    parser.add_argument("--xml_subdir", type=str, default="Annotations")
    # parser.add_argument("--flip_img_id_list", type=str, nargs="+", default=["00000d", "00000f", "00000g"])
    parser.add_argument("--flip_img_id_list", type=str, nargs="+", default=["00000c", "00000d", "00000e", "00000f", "00000g"])
    args = parser.parse_args()

    args.jpg_dir = os.path.join(args.input_dir, args.jpg_subdir)
    args.xml_dir = os.path.join(args.input_dir, args.xml_subdir)

    args.jpg_suffix = ".jpg"
    args.xml_suffix = ".xml"

    data_flip(args)
