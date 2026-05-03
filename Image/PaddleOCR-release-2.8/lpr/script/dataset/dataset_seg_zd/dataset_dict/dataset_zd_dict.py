import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
from script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict_nomal import *

##############################################
# TODO（临时方案，city_f_2_city）
change_country_f_2_country_dict = {
                        'Oman_f': 'Oman', 
                    }
change_city_f_2_city_dict = {
                        'UMMALQAIWAIN_f': 'UMMALQAIWAIN', 
                        'FUJAIRAH_f': 'FUJAIRAH'
                    }
change_car_type_f_2_car_type_dict = {
                        'TRADE_f': 'TRADE', 
                    }

name_2_mask_name_dict = {   
                # background
                'none': 'none',

                # country
                'UAE': 'UAE',
                'Oman': 'Oman',

                # city
                'AD': 'ABUDHABI',
                'ABUDHABI': 'ABUDHABI',
                'DUBAI': 'DUBAI',
                'AJMAN': 'AJMAN',
                'SHARJAH': 'SHARJAH',
                'SHJ': 'SHARJAH',
                'RAK': 'RAK',
                'UMMALQAIWAIN': 'UMMALQAIWAIN',
                'FUJAIRAH': 'FUJAIRAH',

                # car_type
                'TAXI': 'TAXI',
                'POLICE': 'POLICE',
                'TRP': 'TRP',
                'PTR': 'PTR',
                'TRADE': 'TRADE',

                # kind
                'kind': 'kind',

                # num
                'num': 'num',
                }


name_2_id_dict = {   
                # background
                'none': 0,
                'None': 0,

                # country
                'UAE': 1,
                'Oman': 2,

                # city
                'AD': 3,
                'ABUDHABI': 3,
                'DUBAI': 4,
                'AJMAN': 5,
                'SHARJAH': 6,
                'SHJ': 6,
                'RAK': 7,
                'UMMALQAIWAIN': 8,
                'FUJAIRAH': 9,

                # car_type
                'TAXI': 10,
                'POLICE': 11,
                'TRP': 12,
                'PTR': 13,
                'TRADE': 14,

                # kind
                'kind': 15,

                # num
                'num': 16,
                }


name_2_mask_id_dict = {
                # background [0, 0, 0]

                # country
                'UAE': [1, 1, 1],               
                'Oman': [2, 2, 2],             

                # city
                'AD': [3, 3, 3],
                'ABUDHABI': [3, 3, 3],
                'DUBAI': [4, 4, 4],
                'AJMAN': [5, 5, 5],
                'SHARJAH': [6, 6, 6],
                'SHJ': [6, 6, 6],
                'RAK': [7, 7, 7],
                'UMMALQAIWAIN': [8, 8, 8],
                'FUJAIRAH': [9, 9, 9],
                
                # car_type
                'TAXI': [10, 10, 10],
                'POLICE': [11, 11, 11],
                'TRP': [12, 12, 12],
                'PTR': [13, 13, 13],
                'TRADE': [14, 14, 14],

                # kind
                'kind': [15, 15, 15],

                # num
                'num': [16, 16, 16],
                }


name_2_mask_color_dict = {
                # background [0, 0, 0]

                # country
                'UAE': get_color(1),
                'Oman': get_color(2),

                # city
                'AD': get_color(3),
                'ABUDHABI': get_color(3),
                'DUBAI': get_color(4),
                'AJMAN': get_color(5),
                'SHARJAH': get_color(6),
                'SHJ':  get_color(6),
                'RAK': get_color(7),
                'UMMALQAIWAIN': get_color(8),
                'FUJAIRAH': get_color(9),

                # car_type
                'TAXI': get_color(10),
                'POLICE': get_color(11),
                'TRP': get_color(12),
                'PTR': get_color(13),
                'TRADE': get_color(14),

                # kind
                'kind': get_color(15),

                # num
                'num': get_color(16),
                }


id_2_mask_name_dict = {
                # country
                1: 'UAE',
                2: 'Oman',

                # city
                3: 'ABUDHABI',
                4: 'DUBAI',
                5: 'AJMAN',
                6: 'SHARJAH',
                7: 'RAK',
                8: 'UMMALQAIWAIN',
                9: 'FUJAIRAH',
                
                # car_type
                10: 'TAXI',
                11: 'POLICE',
                12: 'TRP',
                13: 'PTR',
                14: 'TRADE',

                # kind
                15: 'kind',

                # num
                16: 'num',
                }


id_2_mask_id_dict = {
                # country
                1: [1, 1, 1],
                2: [2, 2, 2],

                # city                
                3: [3, 3, 3],
                4: [4, 4, 4],
                5: [5, 5, 5],
                6: [6, 6, 6],
                7: [7, 7, 7],
                8: [8, 8, 8],
                9: [9, 9, 9],
                
                # car_type
                10: [10, 10, 10],
                11: [11, 11, 11],
                12: [12, 12, 12],
                13: [13, 13, 13],
                14: [14, 14, 14],

                # kind
                15: [15, 15, 15],

                # num
                16: [16, 16, 16],
                }


id_2_mask_color_dict = {
                # country
                1: get_color(1),
                2: get_color(2),

                # city
                3: get_color(3),
                4: get_color(4),
                5: get_color(5),
                6: get_color(6),
                7: get_color(7),
                8: get_color(8),
                9: get_color(9),

                # car_type
                10: get_color(10),
                11: get_color(11),
                12: get_color(12),
                13: get_color(13),
                14: get_color(14),
                
                # kind
                15: get_color(15),

                # num
                16: get_color(16),
                }


class_seg_label = ['bkg','UAE','Oman','ABUDHABI','DUBAI','AJMAN','SHARJAH','RAK','UMMALQAIWAIN','FUJAIRAH','TAXI','POLICE','TRP','PTR','TRADE', 'kind','num']
# class_seg_weight = [0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5]
class_seg_weight = [0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.5, 0.5]
add_mask_order = ['num','kind','UAE','Oman','AD','ABUDHABI','DUBAI','AJMAN','SHARJAH','SHJ','RAK','UMMALQAIWAIN','FUJAIRAH','TAXI','POLICE','TRP','PTR','TRADE']

country_mask_id_list = [1, 2]
city_mask_id_list = [3, 4, 5, 6, 7, 8, 9]
car_type_mask_id_list = [10, 11, 12, 13, 14]
kind_mask_id_list = [15]
num_mask_id_list = [16]

country_mask_name_list = ['UAE','Oman']
city_mask_name_list = ['ABUDHABI','DUBAI','AJMAN','SHARJAH','RAK','UMMALQAIWAIN','FUJAIRAH']
car_type_mask_name_list = ['TAXI','POLICE','TRP','PTR','TRADE']   
kind_mask_name_list = ['kind']
num_mask_name_list = ['num']

class_seg_label_group = ['country', 'city', 'car_type', 'kind', 'num']
class_seg_label_group_threh_map = {
                                'country': 80, 
                                'city': 10, 
                                'car_type': 80,
                                'kind': 80,
                                'num': 820,
                            }
class_seg_label_group_2_id_map = {
                                'country': country_mask_id_list, 
                                'city': city_mask_id_list, 
                                'car_type': car_type_mask_id_list,
                                'kind': kind_mask_id_list,
                                'num': num_mask_id_list,
                            }

class_seg_label_group_2_name_map = {
                                'country': country_mask_name_list, 
                                'city': city_mask_name_list, 
                                'car_type': car_type_mask_name_list,
                                'kind': kind_mask_name_list,
                                'num': num_mask_name_list,
                            }


# method
def load_object_roi(xml_path):
    
    object_roi_list = []

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
    
        if classname in ignore_unknown:
            continue
        
        if classname in change_name_dict:
            classname = change_name_dict[classname]

        #################################
        # （TODO）以下类别，暂时不参与训练
        #################################
        if classname in change_country_f_2_country_dict:
            classname = change_country_f_2_country_dict[classname]
        if classname in change_city_f_2_city_dict:
            classname = change_city_f_2_city_dict[classname]
        if classname in change_car_type_f_2_car_type_dict:
            classname = change_car_type_f_2_car_type_dict[classname]
        if classname in country_f_name_list:
            continue
        if classname in city_f_name_list:
            continue
        if classname in car_type_f_name_list:
            continue
        if classname in color_name_list:
            continue
        if classname in kind_num_labels:
            continue

        if classname not in name_2_mask_id_dict or \
            classname not in name_2_mask_color_dict:
            print(classname)
            continue
        
        object_roi_list.append({"classname": classname, "bndbox":bndbox})
        
    return object_roi_list