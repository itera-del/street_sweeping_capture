import io
import json
import os
import sys
import xml.etree.ElementTree as ET

# sys.path.insert(0, '/home/huanyuan/code/demo/')
sys.path.insert(0, '/yuanhuan/code/demo/')
from Image.recognition2d.script.lpr.dataset.dataset_morocco.dataset_dict.dataset_morocco_dict_normal import *

##############################################

name_2_mask_name_dict = {   
                # background
                'none': 'none',

                # city
                'city': 'city',

                # kind
                'kind': 'kind',

                # num
                'num': 'num',
                }


name_2_id_dict = {   
                # background
                'none': 0,

                # city
                'city': 1,

                # kind
                'kind': 2,

                # num
                'num': 3,
                }


name_2_mask_id_dict = {
                # background [0, 0, 0]

                # city
                'city': [1, 1, 1],

                # kind
                'kind': [2, 2, 2],

                # num
                'num': [3, 3, 3],
                }


name_2_mask_color_dict = {
                # background [0, 0, 0]

                # city
                'city': get_color(1),

                # kind
                'kind': get_color(2),

                # num
                'num': get_color(3),
                }


id_2_mask_name_dict = {

                # city
                1: 'city',
            
                # kind
                2: 'kind',

                # num
                3: 'num',
                }


id_2_mask_id_dict = {

                # city
                1: [1, 1, 1],

                # kind
                2: [2, 2, 2],

                # num
                3: [3, 3, 3],
                }


id_2_mask_color_dict = {
                # city
                1: get_color(1),

                # kind
                2: get_color(2),

                # num
                3: get_color(3),
                }


class_seg_label = ['bkg','city', 'kind', 'num']
class_seg_weight = [0.1, 1, 1, 1]
add_mask_order = ['num','kind','city']

city_mask_id_list = [1]
kind_mask_id_list = [2]
num_mask_id_list = [3]

city_mask_name_list = ['city']
kind_mask_name_list = ['kind']
num_mask_name_list = ['num']

class_seg_label_group = ['city', 'kind', 'num']
class_seg_label_group_threh_map = {
                                'city': 80, 
                                'kind': 80,
                                'num': 820,
                            }
class_seg_label_group_2_id_map = {
                                'city': city_mask_id_list, 
                                'kind': kind_mask_id_list,
                                'num': num_mask_id_list,
                            }

class_seg_label_group_2_name_map = {
                                'city': city_mask_name_list, 
                                'kind': kind_mask_name_list,
                                'num': num_mask_name_list,
                            }


# method
def load_object_roi(xml_path, json_path, new_style):
    
    object_roi_list = []

    if not os.path.exists(json_path):
        return object_roi_list
        
    # json
    with open(json_path, 'r', encoding='UTF-8') as fr:
        data_json = json.load(fr)

    for cell in data_json['shapes']:
        
        if cell["label"] == "license_kind" or \
            cell["label"] ==  "license_num" or \
            cell["label"] ==  "license_city":

            # name
            classname = str(cell["label"]).replace("license_", "")

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

        if classname not in name_2_mask_id_dict or \
            classname not in name_2_mask_color_dict:
            print(classname)
            continue
        
        object_roi_list.append({"classname": classname, "bndbox":bndbox})

    return object_roi_list