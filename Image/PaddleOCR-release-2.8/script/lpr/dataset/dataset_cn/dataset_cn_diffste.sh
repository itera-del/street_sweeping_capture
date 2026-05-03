#!/bin/bash

data_dir=/yuanhuan/data/image/RM_ANPR/original/cn_202406/china
data_csv_dir=/yuanhuan/data/image/RM_ANPR/original/cn_202406/china_csv/
data_region_csv_dir=/yuanhuan/data/image/RM_ANPR/original/cn_202406/region_csv/
data_crop_dir=/yuanhuan/data/image/RM_ANPR/original/cn_202406/china_crop/
error_data_dir=/yuanhuan/data/image/RM_ANPR/original/cn_202406/china_error_data/
analysis_dir=/yuanhuan/data/image/RM_ANPR/original/cn_202406/china_analysis/
analysis_test_data=/yuanhuan/data/image/RM_ANPR/original/cn_202406/china_analysis_test_data/

seg_name=seg_cn_202407
ocr_name=plate_cn_202407
training_data_dir=/yuanhuan/data/image/RM_ANPR/training/

data_diffste_dir=/yuanhuan/data/image/RM_ANPR/original/cn/DIFFSTE/
ocr_diffste_name=plate_cn_202407

# # #################################
# # # step 2：生成训练数据 seg
# # #################################

# # done
# date_name_list=(jilin liaoning neimenggu ningxia qinghai shandong shanghai shanxi shanxi_jin sichuan tianjin xianggangaomen xinjiang yunnan zhejiang yellow_license_plate 特殊车牌 ZG)
# date_name_list=(guangzhou_8mm_2024/20231214_20240123 guangzhou_8mm_2024/20231214_20240123_2)

# # todo
# date_name_list=()

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name

#     # seg
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_mask/gen_seg_mask.py --date_name=$date_name --seg_name=$seg_name --input_csv_dir=$data_csv_dir --output_dir=$training_data_dir
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_train_test_split/data_train_test_split_seg.py --date_name=$date_name --seg_name=$seg_name --input_dir=$training_data_dir

#     # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_mask/gen_seg_mask_augment.py --date_name=$date_name --seg_name=$seg_name --output_dir=$training_data_dir
#     # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_train_test_split/data_train_test_split_seg_augment.py --date_name=$date_name --seg_name=$seg_name --input_dir=$training_data_dir

# done

# # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_train_test_split/data_train_test_split_seg_merge.py --seg_name=$seg_name --input_dir=$training_data_dir


#################################
# step 3：生成训练数据 ocr
#################################

# done
# date_name_list=()

# todo
date_name_list=(diffste_all_city_clean_100w)

for date_name in ${date_name_list[@]}; do 
    echo $date_name

    # diffste ocr
    python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_mask/gen_ocr_img_diffste.py --date_name=$date_name --ocr_name=$ocr_diffste_name --input_dir=$data_diffste_dir --output_dir=$training_data_dir
    python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_train_test_split/data_train_test_split_ocr.py --date_name=$date_name --ocr_name=$ocr_diffste_name --input_dir=$training_data_dir

done

# ocr
python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_train_test_split/data_train_test_split_ocr_merge.py --ocr_name=$ocr_name --input_dir=$training_data_dir


#################################
# step 4：2 paddleocr label
#################################
# 2 paddleocr label
paddle_ocr_name=plate_cn_202407
image_set_name=ImageSetsOcrLabel
paddle_ocr_data_dir=/yuanhuan/model/image/lpr/paddle_dict/$paddle_ocr_name
python /yuanhuan/code/demo/Image/recognition2d/script/paddle/dataset/lpr_to_paddleocr_label.py --input_dir=$training_data_dir/$ocr_name --image_set_name=$image_set_name --output_dir=$paddle_ocr_data_dir
python /yuanhuan/code/demo/Image/recognition2d/script/paddle/dataset/label_dict.py --output_dir=$paddle_ocr_data_dir --output_name=cn_dict.txt --data_dict_name=script.lpr.dataset.dataset_cn.dataset_dict.dataset_cn_dict_normal