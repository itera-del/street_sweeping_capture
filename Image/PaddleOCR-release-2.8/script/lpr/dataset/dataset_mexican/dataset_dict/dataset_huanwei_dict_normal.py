import numpy as np
import xml.etree.ElementTree as ET

# 加上了 A-Z 26个大写英文字母
kind_num_labels = [
    ' ', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '^', '-', '#',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]
status_name_list = ['n', 'f', 'o']              # n 表示车牌能看清, f 表示车牌因反光或太小等因素模糊而不能识别文字, o 表示车牌遮挡车牌不全的情况

# 无用标签修正
# '-' '@' 实际出现的特殊符号
replace_name_list = ['.', '#', ' ', '　', '"', '”', '@']
# 错误标注修正
replace_name_dict = {
                        'Ａ': 'A',
                        'Ｂ': 'B',
                        'Ｃ': 'C',
                        'Ｄ': 'D',
                        'Ｅ': 'E',
                        'Ｆ': 'F',
                        'Ｇ': 'G',
                        'Ｈ': 'H',
                        'Ｉ': 'I',
                        'Ｊ': 'J',
                        'Ｋ': 'K',
                        'Ｌ': 'L',
                        'Ｍ': 'M',
                        'Ｎ': 'N',
                        '０': 'O',
                        '１': '1',
                        '２': '2',
                        '３': '3',
                        '４': '4',
                        '５': '5',
                        '６': '6',
                        '７': '7',
                        '８': '8',
                        '９': '9',
                        'Ｑ': 'Q',
                        'Ｒ': 'R',
                        'Ｓ': 'S',
                        'Ｔ': 'T',
                        'Ｕ': 'U',
                        'Ｖ': 'V',
                        'Ｗ': 'W',
                        'Ｘ': 'X',
                        'Ｚ': 'Z',
                        'Ｙ': 'Y',
                        '-': '',
                        '_': '',
                    }

analysis_label_columns = {
                            'num': kind_num_labels,
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

    bool_find_status = False
    if 'attributes' in cell and len(cell["attributes"]):
        for json_attributes in cell["attributes"]:
            if json_attributes["name"] == "status":
                bool_find_status = True
                plate_status = json_attributes["value"]

    return plate_status


def json_load_object_plate_num(cell):
    
    # plate num
    plate_num = 'none'
    if 'vehicleic' in cell:
        plate_num = cell["vehicleic"]
    elif 'attributes' in cell and len(cell["attributes"]):
        for json_attributes in cell["attributes"]:
            if json_attributes["name"] == "id":
                plate_num = json_attributes["value"]
            if json_attributes["name"] == "number":
                plate_num = json_attributes["value"]
    else:
        pass
    
    if plate_num != 'none':
        plate_num = plate_num.upper()
        for replace_name in replace_name_list:
            plate_num = plate_num.replace(replace_name, '')
        
        for replace_name in replace_name_dict.keys():
            plate_num = plate_num.replace(replace_name, replace_name_dict[replace_name])

    return plate_num