import argparse
import cv2
import numpy as np
import numpy.random as npr
import json
import os
from tqdm import tqdm
import shutil


ocr_labels = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
              "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏",
              "陕", "甘", "青", "宁", "新", "警", "学", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '挂', '-']  # 33+24+10+1


def decode_json(args):
    json_list = np.array(os.listdir(args.input_dir))
    json_list = json_list[[json_file.endswith('.json') for json_file in json_list]]

    # init 
    lp_num = 0

    # mkdir
    train_str = "train"
    if not os.path.exists(os.path.join(args.output_dir, train_str, 'N')):
        os.makedirs(os.path.join(args.output_dir, train_str, 'N'))
        os.makedirs(os.path.join(args.output_dir, train_str, 'Y1'))
    train_str = "test"
    if not os.path.exists(os.path.join(args.output_dir, train_str, 'N')):
        os.makedirs(os.path.join(args.output_dir, train_str, 'N'))
        os.makedirs(os.path.join(args.output_dir, train_str, 'Y1'))
    if not os.path.exists(args.input_done_dir):
        os.mkdir(args.input_done_dir)
    if not os.path.exists(args.input_error_dir):
        os.mkdir(args.input_error_dir)

    for idx in tqdm(range(len(json_list))):
        json_file = os.path.join(args.input_dir, json_list[idx])
        jpg_file = os.path.join(args.input_dir, json_list[idx][:-4] + "jpg")
        done_bool = True
        
        # 加载 json
        file = open(json_file,'r', encoding="UTF-8") 
        try:
            json_data = json.load(file)
        except:
            print("Load Json Error: ", json_file)
            continue
                    
        # 加载 img
        img_data = cv2.imread(jpg_file, 1)
        if not isinstance(img_data, np.ndarray):
            print("Load Img Error: ", jpg_file)
            continue
            
        # 解析 json
        for json_cell in json_data['shapes']:
            try:
                # type_1
                lp_color = ""
                lp_number = ""
                lp_loc = []

                if json_cell["label"] != "plate":
                    print("Unknow Json Lable: ", json_cell["label"])
                    print("Json path: ", json_file)
                    done_bool = False
                    continue
                
                for json_attributes in json_cell["attributes"]:
                    if json_attributes["name"] == "number":
                        lp_number = json_attributes["value"]
                    elif json_attributes["name"] == "color":
                        lp_color = json_attributes["value"]
                    else:
                        print("Unknow Json Attributes: ", json_attributes["name"])
                        print("Json path: ", json_file)
                        raise Exception()   
                
                lp_loc = json_cell["points"]
            except:
                print("Load LP Error: ", json_file)
                raise Exception()   
            
            # number color check
            lp_number = lp_number.replace('O', '0')
            lp_number = lp_number.replace('I', '1')
            lp_number = lp_number.replace('泸', '沪')
            lp_number = lp_number.replace('翼', '冀')
            lp_number = lp_number.replace('闵', '闽')
            lp_number = lp_number.replace('峡', '陕')
            lp_number = lp_number.replace('Ａ', 'A')
            lp_number = lp_number.replace('Ｂ', 'B')
            lp_number = lp_number.replace('Ｃ', 'C')
            lp_number = lp_number.replace('Ｄ', 'D')
            lp_number = lp_number.replace('Ｅ', 'E')
            lp_number = lp_number.replace('Ｆ', 'F')
            lp_number = lp_number.replace('Ｇ', 'G')
            lp_number = lp_number.replace('Ｈ', 'H')
            lp_number = lp_number.replace('Ｊ', 'J')
            lp_number = lp_number.replace('Ｋ', 'K')
            lp_number = lp_number.replace('Ｌ', 'L')
            lp_number = lp_number.replace('Ｍ', 'M')
            lp_number = lp_number.replace('Ｎ', 'N')
            lp_number = lp_number.replace('０', '0')
            lp_number = lp_number.replace('Ｑ', 'Q')
            lp_number = lp_number.replace('Ｓ', 'S')
            lp_number = lp_number.replace('Ｔ', 'T')
            lp_number = lp_number.replace('Ｕ', 'U')
            lp_number = lp_number.replace('Ｗ', 'W')
            lp_number = lp_number.replace('Ｙ', 'Y')
            lp_number = lp_number.replace('８', '8')
            lp_number = lp_number.replace('.', '')
            lp_number = lp_number.replace('#', '')
            lp_number = lp_number.replace(' ', '')
            lp_number = lp_number.replace('　', '')
            if not ((len(lp_number) == 8 and lp_color == "green") or \
                (len(lp_number) == 7 and lp_color == "yellow")):
                print("Unknow LP Number: ", lp_number)
                print("Unknow LP Color: ", lp_color)
                print("Json path: ", json_file)
                done_bool = False
                continue

            try:
                lp_num += 1
                lp_number = [str(ocr_labels.index(str.upper(lp_number[idx]))) for idx in range(len(lp_number))]
                lp_number = "{}{:0>5d}".format(str(args.data_name), lp_num) + '-' + "_".join(lp_number) + ".jpg"
            except:
                print("Unknow LP Number: ", lp_number)
                print("Json path: ", json_file)
                raise Exception()   

            # random x1\x2\y1\y2
            # TODO(yuanhuan)：做到模型里面
            x1 = lp_loc[0]
            x2 = lp_loc[2]
            y1 = lp_loc[1]
            y2 = lp_loc[3]
            h = y2 - y1
            w = x2 - x1
            
            try:
                x1 = max(0, x1 + npr.randint(-0.08 * w, 0.04 * w))
                x2 = min(img_data.shape[1], x2 + npr.randint(-0.04 * w, 0.08 * w))
                y1 = max(0, y1 + npr.randint(-0.05 * h, 0.05 * h))
                y2 = min(img_data.shape[0], y2 + npr.randint(-0.05 * h, 0.05 * h))
            except:
                # print(x1, x2, y1, y2, w, h)
                pass
            
            # crop
            img_lp = img_data[y1: y2, x1: x2]

            # train or test
            train_bool = True if lp_num%9!=0 else False
            train_str = "train" if train_bool else "test"

            if lp_color == "green":
                cv2.imwrite(os.path.join(args.output_dir, train_str, 'N', lp_number), img_lp)
            elif lp_color == "yellow":
                cv2.imwrite(os.path.join(args.output_dir, train_str, 'Y1', lp_number), img_lp)
        
        # DONE
        if not done_bool:
            shutil.move(json_file, args.input_error_dir)
            shutil.move(jpg_file, args.input_error_dir)
        else:
            try:
                shutil.move(json_file, args.input_done_dir)
                shutil.move(jpg_file, args.input_done_dir)        
            except:
                os.remove(json_file)
                os.remove(jpg_file)
                print("Already Done: ", jpg_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = "/yuanhuan/LicensePlateOCR/原始数据/ZG/"
    args.input_done_dir = "/yuanhuan/LicensePlateOCR/原始数据/ZG_DONE/"
    args.input_error_dir = "/yuanhuan/LicensePlateOCR/原始数据/ZG_ERROR/"
    args.output_dir = "/yuanhuan/LicensePlateOCR/裁剪数据/ZG/"
    args.data_name = 'zg'

    decode_json(args)