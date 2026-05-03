import argparse
import cv2
import io
import json
import os
import pandas as pd
import shutil
import sys
from tqdm import tqdm

sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *

sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/')
from Image.recognition2d.script.lpr.dataset.dataset_singapore_parking_number.dataset_dict.dataset_singapore_dict_normal import *

def dataset_csv(args):

    # mkdir
    create_folder(args.output_csv_dir)
    create_folder(args.output_error_data_dir)

    # img list 
    img_list = get_sub_filepaths_suffix(args.img_dir, ".jpg")
    img_list.sort()
    print(len(img_list))

    # init 
    csv_list = []           # [{"img_path": "", "json_path": "", "id": "", "name": "", "roi": "", "country": "", "city": "", "color": "", "column": "", "num": ""}]
    error_list = []         # [{"img_path": "", "json_path": "", "type": "", "value": ""}]

    for idx in tqdm(range(len(img_list))):

        plate_id = 0
        img_name = os.path.basename(img_list[idx]).replace(".jpg", "")
        img_path = img_list[idx]
        json_name = img_name + ".json"
        json_path = os.path.join(args.json_dir, json_name)

        if args.bool_check_img:
            # img
            try:
                img = cv2.imread(img_path)
            except:
                error_list.append({"img_path": img_path, "json_path": json_path, "type": "img", "value": ""})
                print('"img_path": {}, "json_path": {},"type": "img"'.format(img_path, json_path))
                # raise Exception()
                continue

        # json 
        try:
            with open(json_path, 'r', encoding='UTF-8') as fr:
                data_json = json.load(fr)
        except:
            error_list.append({"img_path": img_path, "json_path": json_path, "type": "json", "value": ""})
            print('"img_path": {}, "json_path": {},"type": "json"'.format(img_path, json_path))
            # raise Exception()
            continue
        
        for cell in data_json['shapes']:
            
            # plate roi
            try:
                x1, x2, y1, y2, w, h = json_load_object_plate_points(cell)
                plate_roi = str("{},{},{},{}".format(x1, x2, y1, y2))
                if args.bool_check_img:
                    plate_img = img[y1:y2, x1:x2]
            except:
                error_list.append({"img_path": img_path, "json_path": json_path, "type": "plate_roi", "value": ""})
                print('"img_path": {}, "json_path": {},"type": "plate_roi"'.format(img_path, json_path))
                # raise Exception()
                continue
            
            # plate status
            plate_status = json_load_object_plate_status(cell)
            if args.new_style == True and plate_status == status_name_list[0]:
                continue

            # plate num
            plate_num = json_load_object_plate_num(cell)
            if plate_num == "":
                plate_num = "none"
            
            # "{}-{:0>2d}_{}"          
            # name-id_车牌号
            plate_name = args.data_format.format(img_name, plate_id, plate_num)
            plate_id += 1

            csv_list.append({"img_path": img_path, "json_path": json_path, "id": plate_id, "name": plate_name, "roi": plate_roi, "num": plate_num})

    # out csv
    csv_pd = pd.DataFrame(csv_list)
    csv_pd.to_csv(args.output_csv_path, index=False, encoding="utf_8_sig")

    error_data_csv_path = os.path.join(args.output_error_data_dir, 'error.csv')
    error_pd = pd.DataFrame(error_list)
    error_pd.to_csv(error_data_csv_path, index=False, encoding="utf_8_sig")


def write_crop_data(args):

    if not args.bool_write_crop_data:
        return

    # mkdir 
    create_folder(args.output_crop_data_img_dir)
    create_folder(args.output_crop_data_nolabel_img_dir)

    # pd
    data_pd = pd.read_csv(args.output_csv_path)

    for idx, row in tqdm(data_pd.iterrows(), total=len(data_pd)):
        
        # info
        img_path = row['img_path']
        plate_name = row['name']
        plate_roi = row['roi'] 
        plate_num = row['num'] 

        # img
        img = cv2.imread(img_path)
        
        # plate_img
        plate_roi_list = plate_roi.split(',')
        x1 = int(plate_roi_list[0])
        x2 = int(plate_roi_list[1])
        y1 = int(plate_roi_list[2])
        y2 = int(plate_roi_list[3])

        # 外扩0.5倍尺寸
        h, w = img.shape[:2]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        bw = x2 - x1
        bh = y2 - y1
        new_bw = int(bw * 2)
        new_bh = int(bh * 2)
        new_x1 = max(0, cx - new_bw // 2)
        new_x2 = min(w, cx + new_bw // 2)
        new_y1 = max(0, cy - new_bh // 2)
        new_y2 = min(h, cy + new_bh // 2)

        try:
            plate_img = img[new_y1:new_y2, new_x1:new_x2]
            plate_img = cv2.flip(plate_img, 1)
            if plate_num != "none":
                cv2.imwrite(os.path.join(args.output_crop_data_img_dir, plate_name + ".jpg"), plate_img)
            else:
                cv2.imwrite(os.path.join(args.output_crop_data_nolabel_img_dir, plate_name + ".jpg"), plate_img)
        except:
            continue


def write_error_data(args):

    if not args.bool_write_error_data:
        return

    # mkdir
    create_folder(args.output_error_data_img_dir)
    create_folder(args.output_error_data_json_dir)
    
    # pd
    error_data_csv_path = os.path.join(args.output_error_data_dir, 'error.csv')

    try:
        data_pd = pd.read_csv(error_data_csv_path)
    except:
        return

    for idx, row in tqdm(data_pd.iterrows(), total=len(data_pd)):
            
        # init 
        rect_list = []

        # info
        img_path = row['img_path']
        json_path = row['json_path']
        img_name = os.path.basename(img_path)
        json_name = os.path.basename(json_path)

        # img
        try:
            to_img_path = os.path.join(args.output_error_data_img_dir, img_name)
            shutil.copy(img_path, to_img_path)
        except:
            continue
        
        # json
        try:
            to_json_path = os.path.join(args.output_error_data_json_dir, json_name)
            shutil.copy(json_path, to_json_path)
        except:
            continue


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="Singapore_202504_1080p") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore") 
    parser.add_argument('--output_csv_dir', type=str, default="/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapor_csv/") 
    parser.add_argument('--output_crop_data_dir', type=str, default="/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapor_crop/") 
    parser.add_argument('--output_error_data_dir', type=str, default="/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapor_error_data/") 
    parser.add_argument('--new_style', action='store_true', default=False) 
    parser.add_argument('--bool_write_crop_data', action='store_true', default=False) 
    parser.add_argument('--bool_write_error_data', action='store_true', default=False) 
    parser.add_argument('--bool_check_img', action='store_true', default=False) 
    parser.add_argument('--img_folder', type=str, default="JPEGImages") 
    parser.add_argument('--json_folder', type=str, default="Json_manual") 

    args = parser.parse_args()
    
    args.data_format = "{}-{:0>2d}_{}"          # name-id_车牌号
    args.img_dir = os.path.join(args.input_dir, args.date_name, args.img_folder)
    args.json_dir = os.path.join(args.input_dir, args.date_name, args.json_folder)
    args.output_csv_path = os.path.join(args.output_csv_dir, args.date_name + '.csv')
    args.output_crop_data_dir = os.path.join(args.output_crop_data_dir, args.date_name)
    args.output_error_data_dir = os.path.join(args.output_error_data_dir, args.date_name)

    print("dataset csv.")
    print("date_name: {}".format(args.date_name))
    print("img_dir: {}".format(args.img_dir))
    print("output_csv_path: {}".format(args.output_csv_path))

    # 生成 dataset csv
    dataset_csv(args)

    # 保存 crop data
    args.output_crop_data_img_dir = os.path.join(args.output_crop_data_dir, "Images")
    args.output_crop_data_nolabel_img_dir = os.path.join(args.output_crop_data_dir, "Images_nolabel")
    write_crop_data(args)

    # 保存 error data
    args.output_error_data_img_dir = os.path.join(args.output_error_data_dir, "Images")
    args.output_error_data_json_dir = os.path.join(args.output_error_data_dir, "Json")
    write_error_data(args)