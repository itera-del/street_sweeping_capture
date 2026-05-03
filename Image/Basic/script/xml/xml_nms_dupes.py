import argparse
import numpy as np
import multiprocessing as mp
import os
import sys 
from tqdm import tqdm
import xml.etree.ElementTree as ET
from typing import Dict

sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.script.xml.xml_write import write_xml
from detection2d.ssd_rfb_crossdatatraining.utils.nms import nms

def xml_nms_mp(info: Dict):

    args = info.pop("args")
    xml_name = info.pop("xml_name")
    input_xml_dir = info.pop("input_xml_dir")
    output_xml_dir = info.pop("output_xml_dir")

    gid = info.pop("gid")
    total_groups = info.pop("total_groups")
    if gid % 1000 == 0:
        print(f"processing group={gid}/{total_groups}")

    # xml
    xml_path = os.path.join(input_xml_dir, xml_name)
    out_xml_path = os.path.join(output_xml_dir, xml_name)

    # read xml
    tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
    root = tree.getroot()   # 获取根节点

    # filename
    filename = root.find('filename').text

    # img_width & img_height
    img_width = int(root.find('size').find('width').text)
    img_height = int(root.find('size').find('height').text)
    img_shape = np.array([img_height, img_width, 3])

    # 标签检测和标签转换
    nms_group_rects = {i: [] for i in range(len(args.nms_groups))}
    no_nms_rect_list = []
    
    for object in root.findall('object'):
        # name
        classname = str(object.find('name').text)

        # change name
        if classname in args.change_label_dict.keys():
            classname = args.change_label_dict[classname]   
        assert classname in args.label_list 
        classid = args.label_list.index(classname)

        # bbox
        bbox = object.find('bndbox')
        pts = ['xmin', 'ymin', 'xmax', 'ymax', 'score']
        bndbox = []
        for i, pt in enumerate(pts):
            if i != len(pts) - 1:
                cur_pt = int(float(bbox.find(pt).text))
            else:
                cur_pt = float(bbox.find(pt).text)
            bndbox.append(cur_pt)
        
        bndbox.extend([classid])
        
        # 检查对象属于哪个NMS组
        found_group = False
        for group_idx, group_labels in enumerate(args.nms_groups):
            if classname in group_labels:
                nms_group_rects[group_idx].append(bndbox)
                found_group = True
                break
        
        if not found_group:
            no_nms_rect_list.append(bndbox)

    # 对每个组分别进行NMS
    xml_bboxes = {}
    for group_idx, rect_list in nms_group_rects.items():
        if len(rect_list):
            rect_list = np.array(rect_list)
            keep = nms(rect_list, 0.45)
            rect_list = rect_list[keep, :]
            
            # 将NMS后的结果添加到xml_bboxes
            for rect_idx in range(len(rect_list)):
                rect_id = rect_list[rect_idx]
                label = args.label_list[int(rect_id[-1])]
                x1 = int(rect_id[0])
                y1 = int(rect_id[1])
                x2 = int(rect_id[2])
                y2 = int(rect_id[3])
                score = float(rect_id[4])

                if label not in xml_bboxes:
                    xml_bboxes[label] = []    
                xml_bboxes[label].append([x1, y1, x2, y2, score])

    # 添加不需要NMS的框
    for rect_idx in range(len(no_nms_rect_list)):
        rect_id = no_nms_rect_list[rect_idx]
        label = args.label_list[int(rect_id[-1])]
        x1 = int(rect_id[0])
        y1 = int(rect_id[1])
        x2 = int(rect_id[2])
        y2 = int(rect_id[3])
        score = float(rect_id[4])

        if label not in xml_bboxes:
            xml_bboxes[label] = []    
        xml_bboxes[label].append([x1, y1, x2, y2, score])
    
    write_xml(out_xml_path, filename, xml_bboxes, img_shape)   


def xml_nms(args):

    # mkdir 
    os.makedirs(args.output_xml_dir, exist_ok=True)

    xml_list = np.array(os.listdir(args.input_xml_dir))
    xml_list = xml_list[[xml.endswith('.xml') for xml in xml_list]]
    xml_list.sort()

    # 组织一下, 变成多进程的输入
    grouped_items = []

    for xml_idx in tqdm(range(len(xml_list))):

        xml_name = xml_list[xml_idx]

        # 增加一些的信息
        item_info = {}
        item_info["args"] = args
        item_info["xml_name"] = xml_name
        item_info["input_xml_dir"] = args.input_xml_dir
        item_info["output_xml_dir"] = args.output_xml_dir

        item_info["gid"] = xml_idx
        item_info["total_groups"] = len(xml_list)
        grouped_items.append(item_info)

    # for idx in range(len(grouped_items)):
    #     xml_nms_mp(grouped_items[idx])

    # 进程池
    nproc = 8
    MAX_PROCESS = min(len(grouped_items), nproc)
    pool = mp.Pool(MAX_PROCESS)

    # 多进程处理
    pool.map(xml_nms_mp, grouped_items)

    
if __name__ == '__main__':
    
    # RM_Capture
    parser = argparse.ArgumentParser()
    parser.add_argument('--jpg_dir', type=str, default="/yuanhuan/data/image/RM_C28_detection/original/america/JPEGImages_dupes_0_95/") 
    parser.add_argument('--input_xml_dir', type=str, default="/yuanhuan/data/image/RM_C28_detection/original/america/Annotations_dupes_0_95_ADAS_MMGroundingDINO/") 
    parser.add_argument('--output_xml_dir', type=str, default="/yuanhuan/data/image/RM_C28_detection/original/america/Annotations_dupes_0_95_ADAS_MMGroundingDINO_NMS/") 
    args = parser.parse_args()

    args.change_label_dict = {'car reg': 'car_reg', 'car big reg': 'car_big_reg', 'car front': 'car_front', 'car big front': 'car_big_front', 'tricycle reg': 'tricycle_reg', \
                                'front face' : 'front_face', 'side face' : 'side_face', 'face occlusion' : 'face_occlusion', }
    args.label_list = ['car', 'bus', 'truck', 'tricycle', 'bicycle', 'motorcycle', 
                 'car_reg', 'car_big_reg', 'car_front', 'car_big_front', 'tricycle_reg', 
                 'upspeed', 'unspeed', 'downspeed', 'height', 'other_sign', 
                 'person', 'bicyclist', 'motorcyclist', 'tricyclist', 'cyclist', 
                 'zebra_crossing', 'bridge', 'trafficlight', 'license',
                 'front_face', 'side_face', 'face_occlusion', 
                 'garbage', 'char','head']
    
    # 定义多个NMS组
    args.nms_groups = [
        ['car', 'bus', 'truck', 'tricyclist'],  # 组1
        ['car_reg', 'car_big_reg', 'car_front', 'car_big_front', 'tricycle_reg'],  # 组2
        ['person', 'bicyclist', 'motorcyclist', 'cyclist'],  # 组3
        ['front_face', 'side_face', 'face_occlusion'],  # 组4
        ['char'],  # 组5
        ['garbage']  # 组6
    ]

    xml_nms(args)