import argparse
import cv2
import numpy as np
import os
import sys 
import shutil
from tqdm import tqdm

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *
from Image.Basic.script.json.platform_json_write import PlatformJsonWriter

# sys.path.insert(0, '/home/huanyuan/code/demo/Image/recognition2d/lpr')
sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/lpr')
from infer.lpr_seg import LPRSegCaffe, LPRSegPytorch


def pre_label(args):

    # mkdir 
    create_folder(args.output_dir)

    # init 
    if args.caffe_bool:
        city_seg = LPRSegCaffe(args.city_seg_caffe_prototxt, args.city_seg_caffe_model_path, args.city_seg_dict_name)
        color_seg = LPRSegCaffe(args.color_seg_caffe_prototxt, args.color_seg_caffe_model_path, args.color_seg_dict_name)
    elif args.pytorch_bool:
        city_seg = LPRSegPytorch(args.city_seg_pth_path, args.city_seg_dict_name)
        color_seg = LPRSegPytorch(args.color_seg_pth_path, args.color_seg_dict_name)

    # img list
    jpg_list = np.array(os.listdir(args.input_img_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list.sort()

    platform_json_writer = PlatformJsonWriter()

    for idx in tqdm(range(len(jpg_list))):

        img_name = jpg_list[idx]
        img_path = os.path.join(args.input_img_dir, img_name)
        tqdm.write(img_path)

        img = cv2.imread(img_path) 

        # info 
        image_width = img.shape[1]
        image_height = img.shape[0]

        # run
        city_preds_mask, city_seg_mask, city_seg_bbox, city_seg_info = city_seg.run(img)
        color_preds_mask, color_seg_mask, color_seg_bbox, color_seg_info = color_seg.run(img)

        seg_bbox = {}
        seg_bbox.update(city_seg_bbox)
        seg_bbox.update(color_seg_bbox)
        
        # 标签检测和标签转换
        rect_list = []
        for key in seg_bbox.keys():
            for idy in range(len(seg_bbox[key])):
                rect_list.append([seg_bbox[key][idy][0], seg_bbox[key][idy][1], seg_bbox[key][idy][0] + seg_bbox[key][idy][2], seg_bbox[key][idy][1] + seg_bbox[key][idy][3], key])

        # json
        output_json_path = os.path.join(args.output_dir, str(img_name).replace('.jpg', '.json'))
        platform_json_writer.write_json(image_width, image_height, img_name, output_json_path, frame_num=idx, rect_list=rect_list)

        output_img_path = os.path.join(args.output_dir, img_name)
        shutil.copy(img_path, output_img_path)

    out_meta_json_path = os.path.join(args.output_dir, 'meta.json')
    platform_json_writer.write_meta_json(out_meta_json_path, task_name="{}_{}".format(args.task_name, len(jpg_list)), label_list=args.label_list)


def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.caffe_bool = False
    args.pytorch_bool = True

    # # zd: seg_city_cartype_kind_num_zd_1019
    # args.city_seg_caffe_prototxt = ""
    # args.city_seg_caffe_model_path = ""
    # args.city_seg_pth_path = "/yuanhuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_1019/LaneNetNova.pth"
    # args.city_seg_dict_name = "dataset_zd_dict"

    # zd: seg_city_cartype_kind_num_zd_1220
    args.city_seg_caffe_prototxt = ""
    args.city_seg_caffe_model_path = ""
    args.city_seg_pth_path = "/yuanhuan/model/image/lpr/zd/seg_city_cartype_kind_num_zd_1220/LaneNetNova.pth"
    args.city_seg_dict_name = "dataset_zd_dict"

    # # zd: seg_color_zd_1104
    # args.color_seg_caffe_prototxt = ""
    # args.color_seg_caffe_model_path = ""
    # args.color_seg_pth_path = "/yuanhuan/model/image/lpr/zd/seg_color_zd_1104/LaneNetNova.pth"
    # args.color_seg_dict_name = "dataset_zd_dict_color"

    # zd: seg_color_zd_1220
    args.color_seg_caffe_prototxt = ""
    args.color_seg_caffe_model_path = ""
    args.color_seg_pth_path = "/yuanhuan/model/image/lpr/zd/seg_color_zd_1220/LaneNetNova.pth"
    args.color_seg_dict_name = "dataset_zd_dict_color"

    # normal
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0804_0809/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0810_0811/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0904_0905/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0901_0903/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0828_0831/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_1024_1029/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_1024_1029_1080p/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_1115_1116/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_1115_1116_1080p/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_0/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_1/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_2/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_3/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_4/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_5/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_6/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_7/"
    args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_8/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_color_green_0/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_color_yellow_0/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_city_RAK_0/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_city_SHARJAH_0/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_city_UMMALQAIWAIN_0/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_city_FUJAIRAH_0/"
    # args.input_dir = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_city_AJMAN_0/"

    # args.output_dir = "/yuanhuan/data/image/LicensePlate_ocr/training/seg_zd/data_crop/Json_color"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_0804_0809/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_0810_0811/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_0904_0905/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_0901_0903/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_0828_0831/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_1024_1029/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_1024_1029_1080p/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_1115_1116/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_1115_1116_1080p/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_0/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_1/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_2/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_3/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_4/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_5/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_6/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_7/"
    args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_8/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_color_green_0/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_color_yellow_0/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_city_RAK_0/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_city_SHARJAH_0/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_city_UMMALQAIWAIN_0/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_city_FUJAIRAH_0/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/seg_city_refine_city_AJMAN_0/"

    args.input_img_dir = os.path.join(args.input_dir, "Images")
    args.task_name = "anpr"

    args.label_list = [
                        "kind", "num", 
                        "UAE", "UAE_f", "Oman", "Oman_f", 
                        "AD", "ABUDHABI", "DUBAI", "AJMAN", "SHARJAH", "SHJ", "RAK", "UMMALQAIWAIN", "FUJAIRAH",
                        "AD_f", "ABUDHABI_f", "DUBAI_f", "AJMAN_f", "SHARJAH_f", "SHJ_f", "RAK_f", "UMMALQAIWAIN_f", "FUJAIRAH_f",
                        "TAXI", "POLICE", "PUBLIC", "TRP", "PROTOCOL", "PTR", "TRADE", "TRAILER", "CONSULATE", "DIPLOMAT",
                        "TAXI_f", "POLICE_f", "PUBLIC_f", "TRP_f", "PROTOCOL_f", "PTR_f", "TRADE_f", "TRAILER_f", "CONSULATE_f", "DIPLOMAT_f"
                        "YELLOW", "RED", "GREEN", "BULE", "ORANGE", "BROWN"
                        ]

    pre_label(args)

    # # test
    # # args.input_dir = "/yuanhuan/data/image/ZD_anpr/test_video/ZD_DUBAI/jpg文件/size/signal/small/"
    # # args.output_dir = "/yuanhuan/model/image/lpr/zd/size_signal_small/"
    # # args.input_dir = "/yuanhuan/data/image/ZD_anpr/test_video/ZD_DUBAI/jpg文件/size/signal/big/"
    # # args.output_dir = "/yuanhuan/model/image/lpr/zd/size_signal_big/"
    # # args.input_dir = "/yuanhuan/data/image/ZD_anpr/test_video/ZD_DUBAI/jpg文件/size/double/small/"
    # # args.output_dir = "/yuanhuan/model/image/lpr/zd/size_double_small/"
    # args.input_dir = "/yuanhuan/data/image/ZD_anpr/test_video/ZD_DUBAI/jpg文件/size/double/big/"
    # args.output_dir = "/yuanhuan/model/image/lpr/zd/size_double_big/"

    # args.input_img_dir = os.path.join(args.input_dir)
    # args.task_name = "anpr"

    # args.label_list = [
    #                     "kind", "num", 
    #                     "UAE", "UAE_f", "Oman", "Oman_f", 
    #                     "AD", "ABUDHABI", "DUBAI", "AJMAN", "SHARJAH", "SHJ", "RAK", "UMMALQAIWAIN", "FUJAIRAH",
    #                     "AD_f", "ABUDHABI_f", "DUBAI_f", "AJMAN_f", "SHARJAH_f", "SHJ_f", "RAK_f", "UMMALQAIWAIN_f", "FUJAIRAH_f",
    #                     "TAXI", "POLICE", "PUBLIC", "TRP", "PROTOCOL", "PTR", "TRADE", "TRAILER", "CONSULATE", "DIPLOMAT",
    #                     "TAXI_f", "POLICE_f", "PUBLIC_f", "TRP_f", "PROTOCOL_f", "PTR_f", "TRADE_f", "TRAILER_f", "CONSULATE_f", "DIPLOMAT_f"
    #                     "YELLOW", "RED", "GREEN", "BULE", "ORANGE", "BROWN"
    #                     ]

    # pre_label(args)


if __name__ == '__main__':
    main()