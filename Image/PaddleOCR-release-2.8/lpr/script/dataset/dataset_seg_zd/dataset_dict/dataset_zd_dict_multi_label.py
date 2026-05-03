import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
from script.dataset.dataset_seg_zd.dataset_dict.dataset_zd_dict_nomal import *

##############################################
# TODO（临时方案，city_f_2_city）
change_city_f_2_city_dict = {
                        'UMMALQAIWAIN_f': 'UMMALQAIWAIN', 
                        'FUJAIRAH_f': 'FUJAIRAH'
                    }
change_country_f_2_country_dict = {
                        'Oman_f': 'Oman', 
                    }

name_2_sigmoid_id_dict = {   
                # country
                'UAE': 0,
                'Oman': 0,

                # city
                'AD': 0,
                'ABUDHABI': 0,
                'DUBAI': 0,
                'AJMAN': 0,
                'SHARJAH': 0,
                'SHJ': 0,
                'RAK': 0,
                'UMMALQAIWAIN': 0,
                'FUJAIRAH': 0,
                
                # car_type
                'TAXI': 0,
                'POLICE': 0,
                'PUBLIC': 0,
                'TRP': 0,
                'PROTOCOL': 0,
                'PTR': 0,
                'EXPORT': 0,
                'EXP': 0,
                'LEARNING': 0,
                'CLASSIC': 0,

                # color
                'RED': 1,
                'GREEN': 1,
                'YELLOW': 1,
                'BULE': 1,
                'ORANGE': 1,
                'BROWN': 1,

                # kind
                'kind': 2,

                # num
                'num': 3,
                }


name_2_sigmoid_mask_color_dict = {
                # country
                'UAE': get_color(1),
                'Oman': get_color(1),

                # city
                'AD': get_color(1),
                'ABUDHABI': get_color(1),
                'DUBAI': get_color(1),
                'AJMAN': get_color(1),
                'SHARJAH': get_color(1),
                'SHJ': get_color(1),
                'RAK': get_color(1),
                'UMMALQAIWAIN': get_color(1),
                'FUJAIRAH': get_color(1),
                
                # car_type
                'TAXI': get_color(1),
                'POLICE': get_color(1),
                'PUBLIC': get_color(1),
                'TRP': get_color(1),
                'PROTOCOL': get_color(1),
                'PTR': get_color(1),
                'EXPORT': get_color(1),
                'EXP': get_color(1),
                'LEARNING': get_color(1),
                'CLASSIC': get_color(1),

                # color
                'RED': get_color(2),
                'GREEN': get_color(2),
                'YELLOW': get_color(2),
                'BULE': get_color(2),
                'ORANGE': get_color(2),
                'BROWN': get_color(2),

                # kind
                'kind': get_color(3),

                # num
                'num': get_color(4),
                }

id_2_sigmoid_mask_color_dict = {
                # country & city & & car_type
                0: get_color(1),
                # color
                1: get_color(2),
                # kind
                2: get_color(3),
                # num
                3: get_color(4),
                }

sigmoid_name_id_2_sigmoid_mask_color_dict = {
                # country & city & & car_type
                'country_city_car_type': get_color(1),
                # color
                'color': get_color(2),
                # kind
                'kind': get_color(3),
                # num
                'num': get_color(4),
                }

class_seg_sigmoid_label = ['country_city_car_type', 'color', 'kind', 'num']
class_seg_sigmoid_weight = [1, 1, 1, 1]

class_seg_label_sigmoid_threh_map = {
                                'country_city_car_type': 10,
                                'color': 80,
                                'kind': 80,
                                'num': 820,
                            }

country_city_car_type_name_list = []
country_city_car_type_name_list.extend(country_name_list)
country_city_car_type_name_list.extend(city_name_list)
country_city_car_type_name_list.extend(car_type_name_list)

class_seg_label_sigmoid_2_name_map = {
                                'country_city_car_type': country_city_car_type_name_list,
                                'color': color_name_list,
                                'kind': kind_name_list,
                                'num': num_name_list,
                            }

def load_object_roi(xml_path):
    
    object_roi_list = []

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
        if classname in change_city_f_2_city_dict:
            classname = change_city_f_2_city_dict[classname]
        if classname in change_country_f_2_country_dict:
            classname = change_country_f_2_country_dict[classname]

        if classname in country_f_name_list:
            continue
        if classname in city_f_name_list:
            continue
        if classname in car_type_f_name_list:
            continue

        if classname not in name_2_sigmoid_id_dict or \
            classname not in name_2_sigmoid_mask_color_dict:
            print(classname)
            continue
        
        object_roi_list.append({"classname": classname, "bndbox":bndbox})
        
    return object_roi_list