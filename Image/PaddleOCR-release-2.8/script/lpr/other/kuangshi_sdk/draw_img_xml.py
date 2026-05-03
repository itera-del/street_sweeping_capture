import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import xml.etree.ElementTree as ET


def draw_img_xml(args, jpg_path, xml_path, wist_score_bool=False):

    img = cv2.imread(jpg_path)

    # read xml
    tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
    root = tree.getroot()   # 获取根节点

    # 标签检测和标签转换
    for object in root.findall('object'):
        # name
        classname = str(object.find('name').text)

        # bbox
        if wist_score_bool:
            bbox = object.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax', 'score']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = float(bbox.find(pt).text)
                bndbox.append(cur_pt)

            if bndbox[-1] < args.score_thr:
                continue
        else:
            bbox = object.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text))
                bndbox.append(cur_pt)

        # draw
        cv2.rectangle(img, (int(bndbox[0]), int(bndbox[1])), (int(bndbox[2]), int(bndbox[3])), args.label_color[0], 4)

        if wist_score_bool:
            img = cv2.putText(img, "{}: {:.2f}".format( classname, bndbox[4] ), (int(bndbox[0]), int(bndbox[1]) - 10), cv2.FONT_HERSHEY_COMPLEX, 1, args.label_color[args.label_list.index(classname)], 2)
        else:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            font_style = ImageFont.truetype("font/simsun.ttc", 50, encoding="utf-8")
            draw = ImageDraw.Draw(img)
            draw.text((int(bndbox[0]), int(bndbox[1]) - 50), "{}".format( classname ), args.label_color[0], font=font_style)
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            # img = cv2.putText(img, "{}".format( classname ), (int(bndbox[0]), int(bndbox[1]) - 10), cv2.FONT_HERSHEY_COMPLEX, 1, args.label_color[0], 2)

    return img


def draw_img(args):

    os.makedirs(args.output_img_dir, exist_ok=True)

    # jpg list 
    jpg_list = np.array(os.listdir(args.input_img_dir))
    jpg_list = jpg_list[[jpg.endswith(args.img_suffix) for jpg in jpg_list]]
    jpg_list.sort()

    for idx in tqdm(range(len(jpg_list))):

        # jpg
        jpg_name = jpg_list[idx]
        jpg_path = os.path.join(args.input_img_dir, jpg_name)

        # xml
        xml_name = str(jpg_name)[:-len(args.img_suffix)] + args.xml_suffix
        xml_path = os.path.join(args.input_xml_dir, xml_name)

        if not os.path.exists(xml_path):
            continue

        output_img_path = os.path.join(args.output_img_dir, jpg_name)
        output_img = draw_img_xml(args, jpg_path, xml_path)

        cv2.imwrite(output_img_path, output_img)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_img_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/测试视频/0619车牌数据/jpg/0000000000000000-240619-091710-091910-000005000000"
    args.input_xml_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/测试视频/0619车牌数据/xml/0000000000000000-240619-091710-091910-000005000000"
    args.output_img_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/测试视频/0619车牌数据/jpg_res/0000000000000000-240619-091710-091910-000005000000"
    args.label_list = ["car", "bus", "truck", "motorcyclist", "license", 'front_face','side_face','person','bicyclist']
    args.label_color = [(235, 211, 70), (106, 90, 205), (160, 32, 240), (176, 23, 31), (142, 0, 0), (230, 0, 0), (106, 0, 228), (60, 100, 0), (80, 100, 0),]
    args.img_suffix = ".jpg"
    args.xml_suffix = ".xml"
    args.img_size = (2592, 1520)

    draw_img(args)