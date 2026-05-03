import xml.etree.ElementTree as ET

ignore_unknown = ['2020', '5 - 5', 'RTP', 'license_plate']

num_labels = [ '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
kind_num_labels = ['-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                    'J', 'K', 'L', 'M', 'N', 'P', 'O', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','#']
column_name_list = ['none', 'Double', 'Single']
country_name_list = ['none', 'UAE', 'KSA', 'Oman', 'Qatar', 'BAHRAIN', 'Kuwait']
country_f_name_list = ['none', 'UAE_f', 'KSA_f', 'Oman_f', 'Qatar_f', 'BAHRAIN_F', 'Kuwait_f']
city_name_list = ['none', 'AD', 'ABUDHABI', 'DUBAI', 'AJMAN', 'SHARJAH', 'SHJ', 'RAK', 'UMMALQAIWAIN', 'FUJAIRAH']   
city_f_name_list = ['none', 'AD_f', 'ABUDHABI_f', 'DUBAI_f', 'AJMAN_f', 'SHARJAH_f', 'SHJ_f', 'RAK_f', 'UMMALQAIWAIN_f', 'FUJAIRAH_f']   
car_type_name_list = ['none', 'TAXI', 'POLICE', 'PUBLIC', 'TRP', 'PROTOCOL', 'PTR', 'TRADE', 'TRAILER', 'CONSULATE', 'LEARNING', "DIPLOMAT", "CLASSIC", "EXPORT", "Military", 'Commercial']     # EXP LEARNING
car_type_f_name_list = ['none', 'TAXI_f', 'POLICE_f', 'PUBLIC_f', 'TRP_f', 'PROTOCOL_f', 'PTR_f', 'TRADE_f', 'TRAILER_f', 'CONSULATE_f', 'LEARNING_f', "DIPLOMAT_f", "CLASSIC_f", "EXPORT_f", "Military_f", 'Commercial_f']
color_name_list = ['RED', 'GREEN', 'YELLOW', 'BLUE', 'ORANGE', 'BROWN']
kind_name_list = ['kind']
num_name_list = ['num']

# 无用标签修正
replace_name_list = ['TAXI#', 'DUBAI#', 'POLICE#', 'POICE#', 'POLCE#', 'TXAI#', 'TAXU#', 
                     '#DUBAI', '#POLICE', '#POICE',  '#POLCE', '#TXAI', '#TAXU', 
                     'TAXI', 'DUBAI', 'POLICE', 'POICE', ' POLCE', 'TXAI', 'TAXU',
                     ' ', '"', '”']

# 标注错误，进行切换
change_name_dict = {
                        'k': 'K', 
                        'UAR': 'UAE',
                        'OMAN': 'Oman',
                        'AbuDhabi': 'ABUDHABI',
                        'DUbAI': 'DUBAI',
                        '3\tDUBAI': 'DUBAI',
                        'UMMAlQAIWAIN': 'UMMALQAIWAIN',
                        'POPLICE': 'POLICE',
                        'OMAN_f': 'Oman_f',
                        'Kuwait-f': 'Kuwait_f',
                        'BULE': 'BLUE',
                        'learning': 'LEARNING',
                        'learning_f': 'LEARNING_f',
                        'commercial': 'Commercial',
                        'commercial_f': 'Commercial_f',
                    }

analysis_label_columns = {
                            'country': country_name_list,
                            'country_f': country_f_name_list,
                            'city': city_name_list, 
                            'city_f': city_f_name_list, 
                            'car_type': car_type_name_list,
                            'car_type_f': car_type_f_name_list,
                            'color': color_name_list,
                        }


def get_color(idx):
    idx = idx * 5
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def load_ori_object_roi(xml_path):
    
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
        
        object_roi_list.append({"classname": classname, "bndbox":bndbox})
        
    return object_roi_list