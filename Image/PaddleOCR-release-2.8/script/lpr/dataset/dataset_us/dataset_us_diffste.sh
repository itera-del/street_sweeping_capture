#!/bin/bash

data_dir=/yuanhuan/data/image/RM_ANPR/original/us/us/
data_csv_dir=/yuanhuan/data/image/RM_ANPR/original/us/us_csv/
data_crop_dir=/yuanhuan/data/image/RM_ANPR/original/us/us_crop/
error_data_dir=/yuanhuan/data/image/RM_ANPR/original/us/us_error_data/
analysis_dir=/yuanhuan/data/image/RM_ANPR/original/us/us_analysis/
data_diffste_dir=/yuanhuan/data/image/RM_ANPR/original/us/DIFFSTE/

ocr_name=plate_us_diffet_202406
ocr_diffste_name=plate_us_diffet_202406
training_data_dir=/yuanhuan/data/image/RM_ANPR/training/


#################################
# step 2：生成训练数据 ocr
#################################

# # done
# # date_name_list=(Texas_20230811 Texas_20230904_20230910 Alabama_2022)
# # date_name_list=(diffste_Texas_665_scale_padding)

# # todo
# date_name_list=(diffste_Al_US_network_7631_scale_padding)

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name

#     # diffste ocr
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_us/dataset_mask/gen_ocr_img_diffste.py --date_name=$date_name --ocr_name=$ocr_diffste_name --input_dir=$data_diffste_dir --output_dir=$training_data_dir
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_us/dataset_train_test_split/data_train_test_split_ocr.py --date_name=$date_name --ocr_name=$ocr_diffste_name --input_dir=$training_data_dir

# done

# # ocr & diffste ocr
# python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_us/dataset_train_test_split/data_train_test_split_ocr_merge.py --ocr_name=$ocr_name --input_dir=$training_data_dir


#################################
# step 3：2 paddleocr label
#################################
# 2 paddleocr label
paddle_ocr_name=$ocr_name
# paddle_ocr_name=$ocr_diffste_name
# image_set_name=ImageSetsOcrLabel
image_set_name=ImageSetsOcrLabelNoAug
paddle_ocr_data_dir=/yuanhuan/model/image/lpr/paddle_dict/plate_us_diffet_202406
python /yuanhuan/code/demo/Image/recognition2d/script/paddle/dataset/lpr_to_paddleocr_label.py --input_dir=$training_data_dir/$paddle_ocr_name --image_set_name=$image_set_name --output_dir=$paddle_ocr_data_dir
python /yuanhuan/code/demo/Image/recognition2d/script/paddle/dataset/label_dict.py --output_dir=$paddle_ocr_data_dir --output_name=us_dict.txt --data_dict_name=script.lpr.dataset.dataset_us.dataset_dict.dataset_us_dict_normal