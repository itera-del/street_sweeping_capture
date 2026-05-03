import argparse
import numpy as np
import os
import sys 
from tqdm import tqdm
import xml.etree.ElementTree as ET
import cv2

sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.script.xml.xml_write import write_xml
from detection2d.ssd_rfb_crossdatatraining.utils.nms import nms

def xml_nms(args):

    # mkdir 
    os.makedirs(args.out_xml_dir, exist_ok=True)

    xml_list = np.array(os.listdir(args.xml_dir))
    xml_list = xml_list[[xml.endswith('.xml') for xml in xml_list]]
    xml_list.sort()

    jpg_list = np.array(os.listdir(args.jpg_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]

    for idx in tqdm(range(len(xml_list))):
        # xml
        xml_name = xml_list[idx]
        xml_path = os.path.join(args.xml_dir, xml_name)
        out_xml_path = os.path.join(args.out_xml_dir, xml_name)

        # jpg
        jpg_name = str(xml_name)[:-4] + '.jpg'
        jpg_path = os.path.join(args.jpg_dir, jpg_name)
        assert jpg_name in jpg_list

        # read xml
        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点

        # img_width & img_height
        # img = cv2.imread(jpg_path)
        # img_width = img.shape[1]
        # img_height = img.shape[0]
        img_width = int(root.find('size').find('width').text)
        img_height = int(root.find('size').find('height').text)
        img_shape = np.array([img_height, img_width, 3])

        # 标签检测和标签转换
        rect_list = []
        for object in root.findall('object'):
            # name
            classname = str(object.find('name').text)

            # change name
            if classname in args.change_label_dict.keys():
                classname = args.change_label_dict[classname]   
            assert classname in args.label_list 
            classname = args.label_list.index(classname)

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
            
            bndbox.extend([classname])
            rect_list.append(bndbox)

        # nms
        if len(rect_list):
            rect_list = np.array(rect_list)
            keep = nms(rect_list, 0.45)
            rect_list = rect_list[keep, :]

        xml_bboxes = {}
        for rect_idx in range(len(rect_list)):
            rect_id = rect_list[rect_idx]
            label = args.label_list[int(rect_id[-1])]
            x1 = int(rect_id[0])
            y1 = int(rect_id[1])
            x2 = int(rect_id[2])
            y2 = int(rect_id[3])

            if label not in xml_bboxes:
                xml_bboxes[label] = []    
            xml_bboxes[label].append([x1, y1, x2, y2])
        
        write_xml(out_xml_path, jpg_path, xml_bboxes, img_shape)   


if __name__ == '__main__':
    
    # RM_Capture
    parser = argparse.ArgumentParser()
    parser.add_argument('--jpg_dir', type=str, default="/yuanhuan/data/image/RM_Face/original/aebs/ADAS_AllInOne/JPEGImages/") 
    parser.add_argument('--xml_dir', type=str, default="/yuanhuan/data/image/RM_Face/original/aebs/ADAS_AllInOne/Annotations_Face_wZhedang_MMGroundingDINO/") 
    parser.add_argument('--out_xml_dir', type=str, default="/yuanhuan/data/image/RM_Face/original/aebs/ADAS_AllInOne/Annotations_Face_wZhedang_MMGroundingDINO_NMS/") 
    args = parser.parse_args()

    # Capture
    args.task_name = "Capture"
    args.label_list =  ['car', 'bus', 'truck', 'tricycle', 'bicycle', 'motorcycle', 'license', 'person', 'bicyclist', 'motorcyclist', 'tricyclist', 'front_face', 'side_face', 'face_occlusion']
    args.change_label_dict = {'front face' : 'front_face', 'side face' : 'side_face', 'face occlusion' : 'face_occlusion'}

    xml_nms(args)