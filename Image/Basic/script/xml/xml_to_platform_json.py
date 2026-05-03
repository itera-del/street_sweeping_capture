import argparse
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
import xml.etree.ElementTree as ET

sys.path.insert(0, "/yuanhuan/code/demo")
from Image.Basic.utils.folder_tools import *
from Image.Basic.script.json.platform_json_write import PlatformJsonWriter


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
        jpg_name = str(xml_name)[:-4] + ".jpg"
        jpg_path = os.path.join(args.jpg_dir, jpg_name)
        if jpg_name not in jpg_list:
            continue

        # read xml
        tree = ET.parse(
            xml_path
        )  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()  # 获取根节点

        # img_width & img_height
        img = cv2.imread(jpg_path)
        img_width = img.shape[1]
        img_height = img.shape[0]
        # img_width = int(root.find('size').find('width').text)
        # img_height = int(root.find('size').find('height').text)
        # print(img_width, img_height)

        # 标签检测和标签转换
        rect_list = []
        for object in root.findall("object"):
            # name
            classname = str(object.find("name").text)

            # change name
            if classname in args.change_label_dict.keys():
                classname = args.change_label_dict[classname]
            # if classname not in args.label_list:
            #     print("Ignore: {}".format(classname))
            #     continue
            # assert classname in args.label_list

            # bbox
            bbox = object.find("bndbox")
            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text))
                bndbox.append(cur_pt)

            bndbox.extend([classname])
            rect_list.append(bndbox)

        json_name = str(xml_name)[:-4] + ".json"
        out_json_path = os.path.join(args.platform_json_dir, json_name)
        platform_json_writer.write_json(
            img_width,
            img_height,
            jpg_name,
            out_json_path,
            frame_num=idx,
            rect_list=rect_list,
        )

    out_meta_json_path = os.path.join(args.platform_json_dir, "meta.json")
    platform_json_writer.write_meta_json(
        args.meta_template_path,
        out_meta_json_path,
        task_name=args.task_name,
        data_size=len(jpg_list),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/yuanhuan/data/image/RM_Capture/training/Capture_BSD_R151_bicyclist_motorcyclist",
    )
    parser.add_argument(
        "--meta_template_path",
        type=str,
        default="/yuanhuan/code/demo/Image/Basic/script/xml/meta.json",
    )
    # parser.add_argument(
    #     "--label_list",
    #     type=str,
    #     nargs="+",
    #     default=[""],
    # )
    parser.add_argument("--jpg_name", type=str, default="JPEGImages_cyclist/")
    parser.add_argument("--xml_name", type=str, default="Annotations_cyclist/")
    parser.add_argument("--json_name", type=str, default="Jsons_cyclist/")
    parser.add_argument("--task_name", type=str, default="normal")

    args = parser.parse_args()

    print(args.input_dir)
    args.jpg_dir = os.path.join(args.input_dir, args.jpg_name)
    args.xml_dir = os.path.join(args.input_dir, args.xml_name)
    args.platform_json_dir = os.path.join(args.input_dir, args.json_name)
    args.change_label_dict = {}

    xml_2_platform_json(args)
