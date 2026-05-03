import numpy as np
import io
import json
import os
import xml.etree.ElementTree as ET
import sys

sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
from script.lpr.dataset.dataset_normal.dataset_dict.dataset_dict_normal import *

num_labels = [ '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
kind_labels = ['A', 'B', 'D', 'H', 'E', 'T', 'Y', 'K', 'L', 'M', 'N', 'C', 'F', 'R', 'S', '^']
kind_lowercase_labels = ['a', 'b', 'd', 'h', 'e', 't', 'y', 'k', 'l', 'm', 'n', 'c', 'f', 'r', 's', '^']
status_name_list = ['none', 'n', 'f', 'o']              # n 表示车牌能看清, f 表示车牌因反光或太小等因素模糊而不能识别文字, o 表示车牌遮挡车牌不全的情况
column_name_list = ['none', 'double', 'single']
color_name_list = ['none', 'red', 'green', 'yellow', 'blue', 'white', 'infrared', 'black', 'orange', 'brown']
city_name_list = ['none', 'city']
kind_name_list = ['none', 'kind']
num_name_list = ['none', 'num']

# 无用标签修正
replace_name_list = ['.', ' ', '　', '"', '”']
# 错误标注修正
replace_name_dict = {}

# 跳过位置标签
ignore_unknown_label = []

analysis_label_columns = {
                            'num': kind_num_labels,
                            'column': column_name_list,
                            'color': color_name_list,
                        }

# method 
def get_color(idx):
    idx = idx * 5
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def json_load_object_plate_points(cell):
    
    # plate points
    if 'type' in cell and cell["type"] == "rectangle":
        pts = np.array(cell["points"], np.int32)
        pts = pts.reshape((-1, 1, 2))

        x1 = np.min((pts[0][0][0], pts[1][0][0]))
        x2 = np.max((pts[0][0][0], pts[1][0][0]))
        y1 = np.min((pts[0][0][1], pts[1][0][1]))
        y2 = np.max((pts[0][0][1], pts[1][0][1]))
        w = x2 - x1
        h = y2 - y1
    
    else:
        pts = np.array(cell["points"], np.int32)
        pts = pts.reshape((-1, 1, 2))

        x1 = np.min((pts[0][0][0], pts[1][0][0], pts[2][0][0], pts[3][0][0]))
        x2 = np.max((pts[0][0][0], pts[1][0][0], pts[2][0][0], pts[3][0][0]))
        y1 = np.min((pts[0][0][1], pts[1][0][1], pts[2][0][1], pts[3][0][1]))
        y2 = np.max((pts[0][0][1], pts[1][0][1], pts[2][0][1], pts[3][0][1]))
        w = x2 - x1
        h = y2 - y1

    return x1, x2, y1, y2, w, h


def json_load_object_plate_status(cell):
    
    # plate status
    plate_status = status_name_list[0]

    if 'attributes' in cell and len(cell["attributes"]):
        for json_attributes in cell["attributes"]:
            if json_attributes["name"] == "status":
                plate_status = json_attributes["value"]

    return plate_status


def json_load_object_plate_color(cell):

    # plate color
    plate_color = color_name_list[0]
    load_plate_color = color_name_list[0]

    if 'vidcolor' in cell:
        load_plate_color = cell['vidcolor']

        if load_plate_color == 'red_card':
            plate_color = color_name_list[1]
        elif load_plate_color == 'green_card':
            plate_color = color_name_list[2]
        elif load_plate_color == 'yellow_card':
            plate_color = color_name_list[3]
        elif load_plate_color == 'blue_card':
            plate_color = color_name_list[4]
        elif load_plate_color == 'white_card':
            plate_color = color_name_list[5]
        elif load_plate_color == 'infrared_card':
            plate_color = color_name_list[6]
        elif load_plate_color == 'black_card':
            plate_color = color_name_list[7]
        elif load_plate_color == 'orange_card':
            plate_color = color_name_list[8]
        elif load_plate_color == 'brown_card':
            plate_color = color_name_list[9]

    elif 'attributes' in cell and len(cell["attributes"]):
        for json_attributes in cell["attributes"]:
            if json_attributes["name"] == "color":
                load_plate_color = json_attributes["value"]
        
        if load_plate_color == 'red':
            plate_color = color_name_list[1]
        elif load_plate_color == 'green':
            plate_color = color_name_list[2]
        elif load_plate_color == 'yellow':
            plate_color = color_name_list[3]
        elif load_plate_color == 'blue':
            plate_color = color_name_list[4]
        elif load_plate_color == 'white':
            plate_color = color_name_list[5]
        elif load_plate_color == 'infrared':
            plate_color = color_name_list[6]
        elif load_plate_color == 'black':
            plate_color = color_name_list[7]
        elif load_plate_color == 'orange':
            plate_color = color_name_list[8]
        elif load_plate_color == 'brown':
            plate_color = color_name_list[9]
        # plate_color == 'unknown'：默认为白色
        elif load_plate_color == 'unknown':
            plate_color = color_name_list[5]
        else:
            print()
    
    return plate_color, load_plate_color


def json_load_object_plate_column(cell, w, h):

    # plate column
    plate_column = column_name_list[0]
    load_plate_column = column_name_list[0]
    
    if 'vidlan' in cell:
        load_plate_column = cell['vidlan']

        if load_plate_column == 'Double_column':
            plate_column = column_name_list[1]
        elif load_plate_column == 'Single_column':
            plate_column = column_name_list[2]

    elif 'attributes' in cell and len(cell["attributes"]):
        for json_attributes in cell["attributes"]:
            if json_attributes["name"] == "column":
                plate_column = json_attributes["value"]

    if plate_column == column_name_list[0]:
        if( w / h > 2.5 ):
            plate_column = column_name_list[2]
        else:
            plate_column = column_name_list[1]
    
    return plate_column, load_plate_column


def json_load_object_plate_num(cell):

    # plate num
    plate_num = 'none'
    if 'vehicleic' in cell:
        plate_num = cell["vehicleic"]
    elif 'attributes' in cell and len(cell["attributes"]):
        for json_attributes in cell["attributes"]:
            if json_attributes["name"] == "number":
                plate_num = json_attributes["value"]
            elif json_attributes["name"] == "id":
                plate_num = json_attributes["value"]
    else:
        pass
    
    if plate_num.lower() != 'none':
        plate_num = plate_num.upper()
        for replace_name in replace_name_list:
            plate_num = plate_num.replace(replace_name, '')
        
        for replace_name in replace_name_dict.keys():
            plate_num = plate_num.replace(replace_name, replace_name_dict[replace_name])

    return plate_num


def load_ori_object_roi(xml_path, json_path, new_style):
    
    object_roi_list = []

    if not new_style:
        if not os.path.exists(xml_path):
            return object_roi_list
        
        # xml
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

            classname = classname.lower()

            if classname in replace_name_dict:
                classname = replace_name_dict[classname]        

            if classname in ignore_unknown_label:
                continue
            
            object_roi_list.append({"classname": classname, "bndbox":bndbox})
    
    else:
        if not os.path.exists(json_path):
            return object_roi_list
        
        # json
        with open(json_path, 'r', encoding='UTF-8') as fr:
            data_json = json.load(fr)
        
        for cell in data_json['shapes']:
            
            if cell["label"] == "license_kind" or \
                cell["label"] ==  "license_num":

                # name
                classname = str(cell["label"]).replace("license_", "")

            elif cell["label"] == "license_country" or \
                cell["label"] == "license_country_f" or \
                cell["label"] == "license_city" or \
                cell["label"] == "license_city_f" or \
                cell["label"] == "license_car_type" or \
                cell["label"] == "license_car_type_f" or \
                cell["label"] == "license_color" :

                # name
                classname = str(cell["attributes"][0]["value"])

            # bbox
            pts = np.array(cell["points"], np.int32)
            pts = pts.reshape((-1, 1, 2))

            x1 = np.min((pts[0][0][0], pts[1][0][0]))
            x2 = np.max((pts[0][0][0], pts[1][0][0]))
            y1 = np.min((pts[0][0][1], pts[1][0][1]))
            y2 = np.max((pts[0][0][1], pts[1][0][1]))

            # bbox
            bndbox = [x1, y1, x2, y2]

            if classname in ignore_unknown_label:
                continue

            object_roi_list.append({"classname": classname, "bndbox":bndbox})
            
    return object_roi_list