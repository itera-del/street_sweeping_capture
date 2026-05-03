import argparse
import csv  
import os  
from typing import Dict  
import pandas as pd


AREA_MAPPING: Dict[str, str] = {  
    '京': '京',  
    '沪': '沪',  
    '津': '津',  
    '渝': '渝',  
    '冀': '冀',  
    '豫': '豫',  
    '云': '云',  
    '辽': '辽',  
    '黑': '黑',  
    '湘': '湘',  
    '皖': '皖',  
    '鲁': '鲁',  
    '新': '新',  
    '苏': '苏',  
    '浙': '浙',  
    '赣': '赣',  
    '鄂': '鄂',  
    '桂': '桂',  
    '甘': '甘',  
    '晋': '晋',  
    '蒙': '蒙',  
    '陕': '陕',  
    '吉': '吉',  
    '闽': '闽',  
    '贵': '贵',  
    '粤': '粤',  
    '青': '青',  
    '藏': '藏',  
    '川': '川',  
    '宁': '宁',  
    '琼': '琼',  
    '港': '港',  
    '澳': '澳',  
    '台': '台',  
}
OTHER_MAPPING: Dict[str, str] = {  
    '警': '警',  
    '挂': '挂',  
}

#颜色统计
def ananlysis_region_num_color(output_region_csv_dir, output_analysis_csv_path):  
    all_regions_data = []  
    for filename in os.listdir(output_region_csv_dir):   
        if filename.endswith('.csv'):  
            i = len(all_regions_data) + 1 
            color_dict = {'blue': 0, 'green': 0, 'yellow': 0, 'white': 0, 'black': 0}  
            full_path = os.path.join(output_region_csv_dir, filename)  
            df = pd.read_csv(full_path)  
            for index, row in df.iterrows():  
                color = row['color']  
                if color in color_dict:  
                    color_dict[color] += 1  
            area=get_area_char(filename)
            if area == "":
                area=get_other_char(filename)
            region_data = {  
                '地区编号': i,  
                '地区名称': area,  
                'blue': str(color_dict['blue']) ,
                'green': str(color_dict['green']) ,
                'yellow': str(color_dict['yellow']) ,
                'white': str(color_dict['white']) ,
                'black': str(color_dict['black']) 
            }  
            all_regions_data.append(region_data)  
        print(i,".地区:",filename,"  颜色分布:",color_dict)
    all_regions_df = pd.DataFrame(all_regions_data) 
    all_regions_df.to_csv(output_analysis_csv_path, index=False, encoding='utf-8-sig')
  

def get_area_char(text: str) -> str:  
    for char in text:  
        if '\u4e00' <= char <= '\u9fff':  
            if char in AREA_MAPPING:
                return char  
    return ''  


def get_other_char(text: str) -> str:  
    for char in text:  
        if '\u4e00' <= char <= '\u9fff':  
            if char in OTHER_MAPPING:
                return char  
    return ''  


area_set = set() 
other_set = set()


def get_region_csv(input_csv_path: str, output_region_csv_dir: str):  
    print("程序执行中...")  
    os.makedirs(output_region_csv_dir, exist_ok=True)
    with open(input_csv_path, newline='', encoding='utf-8') as csvfile:  
        reader = csv.DictReader(csvfile)  
        for row in reader:  
            name = row['name'] 

            area_char = get_area_char(row['num'])
            if area_char in AREA_MAPPING:  
                area = AREA_MAPPING[area_char]  
                output_path = os.path.join(output_region_csv_dir, f'{area}.csv')  
                with open(output_path, 'a', newline='', encoding='utf-8') as outputfile:  
                    if os.path.getsize(output_path) == 0:  
                        writer = csv.DictWriter(outputfile, fieldnames=reader.fieldnames)  
                        writer.writeheader()  
                    else:  
                        writer = csv.DictWriter(outputfile, fieldnames=reader.fieldnames) 
                    writer.writerow(row)  
                    area_set.add(name)

            other_char = get_other_char(row['num'])
            if other_char in OTHER_MAPPING:  
                other = OTHER_MAPPING[other_char]  
                output_path = os.path.join(output_region_csv_dir, f'{other}.csv')  
                with open(output_path, 'a', newline='', encoding='utf-8') as outputfile:  
                    if os.path.getsize(output_path) == 0:  
                        writer = csv.DictWriter(outputfile, fieldnames=reader.fieldnames)  
                        writer.writeheader()  
                    else:  
                        writer = csv.DictWriter(outputfile, fieldnames=reader.fieldnames) 
                    writer.writerow(row)  
                    other_set.add(name) 


def main(args):
    input_csv_dir = args.input_csv_dir
    output_region_csv_dir = args.output_region_csv_dir 
    output_analysis_csv_path=args.output_analysis_csv_path

    for filename in os.listdir(input_csv_dir):  
        input_csv_path = os.path.join(input_csv_dir, filename) 
        print(input_csv_path)
        get_region_csv(input_csv_path, output_region_csv_dir)
    ananlysis_region_num_color(output_region_csv_dir, output_analysis_csv_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_dir', type=str, default='/yuanhuan/data/image/RM_ANPR/original/cn_202406/china_csv')  # 输入的需要划分的原始csv
    parser.add_argument('--output_region_csv_dir', type=str, default='/yuanhuan/data/image/RM_ANPR/original/cn_202406/region_csv')    # 输出按照省份汉字划分的csv
    parser.add_argument('--output_analysis_csv_path', type=str, default='/yuanhuan/data/image/RM_ANPR/original/cn_202406/china_analysis/ImageSetsLabel/region_color.csv')   # 划分后的颜色统计 
    args = parser.parse_args()
    main(args)
    
