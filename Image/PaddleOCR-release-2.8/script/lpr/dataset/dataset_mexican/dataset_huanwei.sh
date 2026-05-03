#!/bin/bash
data_closed_dir=/yuanhuan/data/dataset_closed_loop/frames/huanwei
data_dir=/yuanhuan/data/image/RM_HUANWEI/original/Argentina/Argentina
data_csv_dir=/yuanhuan/data/image/RM_HUANWEI/original/Argentina/Argentina_csv/
data_crop_dir=/yuanhuan/data/image/RM_HUANWEI/original/Argentina/Argentina_crop/
error_data_dir=/yuanhuan/data/image/RM_HUANWEI/original/Argentina/Argentina_error_data/
analysis_dir=/yuanhuan/data/image/RM_HUANWEI/original/Argentina/Argentina_analysis/
data_diffste_dir=

ocr_name=plate_huanwei_20251030
training_data_dir=/yuanhuan/data/image/RM_HUANWEI/training_ocr/


# 数据闭环系统（抽帧、去重、标注）
# bash /yuanhuan/code/demo/Image/dataset_closed_loop/dataset_project/data_closing_huanwei.sh

# #################################
# # step 0：数据拷贝：将需要标注的图片拷贝到对应的目录下
# #################################
# # done
# # date_name_list=(20250910_无编号有垃圾桶)    # group_{date_name}_frames_5f
# # date_name_list=(20250910_有编号有垃圾桶)    # group_{date_name}_frames_5f
# # date_name_list=(20250820_无编号有垃圾桶)    # group_{date_name}_frames_5f
# # date_name_list=(20250820_有编号有垃圾桶)    # group_{date_name}_frames_5f
# # date_name_list=(20250925_无编号有垃圾桶)    # group_{date_name}_frames_5f
# # date_name_list=(20250925_有编号有垃圾桶)    # group_{date_name}_frames_5f
# # date_name_list=(20251021_无编号有垃圾桶)    # group_{date_name}_frames_5f
# date_name_list=(20251021_有编号有垃圾桶)    # group_{date_name}_frames_5f

# source /opt/conda/bin/activate base
# conda init bash
# conda activate mmdet

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name
    
#     # group_{date_name}_frames_5f
#     # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_move/data_move.py --input_dir=$data_closed_dir/group_${date_name}_frames_5f --output_dir=$data_dir/Argentina_${date_name} --jpg_subdir=batch_1 --xml_subdir=xml_huanwei_det_v1.0_1
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_move/data_move.py --input_dir=$data_closed_dir/group_${date_name}_frames_5f --output_dir=$data_dir/Argentina_${date_name} --jpg_subdir=batch_1 --xml_subdir=xml_huanwei_det_v1.0_1_ocr
#     python /yuanhuan/code/demo/Image/Basic/script/xml/xml_to_platform_json_HUANWEI.py --input_dir=$data_dir/Argentina_${date_name}

# done

# #################################
# # step 1：处理标注数据
# #################################

# # done
# # date_name_list=(Argentina_20240910 Argentina_20240911 Argentina_20240912)
# # date_name_list=(Argentina_20250910_无编号有垃圾桶 Argentina_20250910_有编号有垃圾桶)
# # date_name_list=(Argentina_20250820_无编号有垃圾桶 Argentina_20250820_有编号有垃圾桶)
# # date_name_list=(Argentina_20250925_无编号有垃圾桶 Argentina_20250925_有编号有垃圾桶)
# # date_name_list=(Argentina_20251021_无编号有垃圾桶 Argentina_20251021_有编号有垃圾桶)

# # todo
# # date_name_list=()

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name

#     # normal
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_huanwei/dataset_check/dataset_csv.py --date_name=$date_name --input_dir=$data_dir --output_csv_dir=$data_csv_dir --output_crop_data_dir=$data_crop_dir --output_error_data_dir=$error_data_dir --bool_write_error_data
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_huanwei/dataset_analysis/analysis_dataset_label_num.py --date_name=$date_name --input_csv_dir=$data_csv_dir --output_analysis_dir=$analysis_dir

# done

# python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_huanwei/dataset_analysis/analysis_dataset_label_num_merge.py --input_dir=$analysis_dir


#################################
# step 2：生成训练数据 ocr
#################################

# # done
# date_name_list=(Argentina_20240910 Argentina_20240911 Argentina_20240912 \
#                 Argentina_20250910_有编号有垃圾桶 \
#                 Argentina_20250820_有编号有垃圾桶 \
#                 Argentina_20250925_有编号有垃圾桶 \
#                 Argentina_20251021_有编号有垃圾桶)

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name
    
#     # ocr
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_huanwei/dataset_mask/gen_ocr_img.py --date_name=$date_name --ocr_name=$ocr_name --input_csv_dir=$data_csv_dir --output_dir=$training_data_dir
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_huanwei/dataset_train_test_split/data_train_test_split_ocr.py --date_name=$date_name --ocr_name=$ocr_name --input_dir=$training_data_dir

#     # ocr augment
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_huanwei/dataset_mask/gen_ocr_img_augment.py --date_name=$date_name --ocr_name=$ocr_name --output_dir=$training_data_dir
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_huanwei/dataset_train_test_split/data_train_test_split_ocr_augment.py --date_name=$date_name --ocr_name=$ocr_name --input_dir=$training_data_dir

# done

# # ocr & diffste ocr
# python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_huanwei/dataset_train_test_split/data_train_test_split_ocr_merge.py --ocr_name=$ocr_name --input_dir=$training_data_dir


# #################################
# step 3：2 paddleocr label
#################################
# 2 paddleocr label
paddle_ocr_name=$ocr_name
# image_set_name=ImageSetsOcrLabel
# paddle_ocr_data_dir=/yuanhuan/model/image/lpr/paddle_dict/plate_huanwei_20251030
image_set_name=ImageSetsOcrLabelNoDiffsteNoAug
paddle_ocr_data_dir=/yuanhuan/model/image/lpr/paddle_dict/plate_huanwei_20251030_NoDiffsteNoAug
python /yuanhuan/code/demo/Image/recognition2d/script/paddle/dataset/lpr_to_paddleocr_label.py --input_dir=$training_data_dir/$paddle_ocr_name --image_set_name=$image_set_name --output_dir=$paddle_ocr_data_dir
python /yuanhuan/code/demo/Image/recognition2d/script/paddle/dataset/label_dict.py --output_dir=$paddle_ocr_data_dir --output_name=huanwei_dict.txt --data_dict_name=script.lpr.dataset.dataset_huanwei.dataset_dict.dataset_huanwei_dict_normal


# # #################################
# # step 4：检测器训练准备
# # #################################

# # done
# date_name_list=(Argentina_20240910 Argentina_20240911 Argentina_20240912 \
#                 Argentina_20250820_无编号有垃圾桶 Argentina_20250820_有编号有垃圾桶 \
#                 Argentina_20250910_无编号有垃圾桶 Argentina_20250910_有编号有垃圾桶 \
#                 Argentina_20250925_无编号有垃圾桶 Argentina_20250925_有编号有垃圾桶 \
#                 Argentina_20251021_无编号有垃圾桶 Argentina_20251021_有编号有垃圾桶)

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name
    
#     # # rm
#     # cd $data_dir/$date_name
#     # rm -rf ./Annotations
#     # rm -rf ./Annotations_test
#     # rm -rf ./JPEGImages_test
#     # rm -rf ./ImageSets
#     # rm -rf ./*.json

#     # platform_json_to_xml
#     python /yuanhuan/code/demo/Image/Basic/script/json/platform_json_to_xml.py --input_dir=$data_dir/$date_name --jpg_name=JPEGImages --json_name=Jsons --xml_name=Annotations

#     python /yuanhuan/code/demo/Image/detection2d/script/dataset/dataset_train_test_split/data_train_test_split.py --input_dir=$data_dir/$date_name --jpg_name=JPEGImages --xml_name=Annotations
#     bash /yuanhuan/code/demo/Image/detection2d/rm_ai_mmdet_3_3_yolox/run_manifest_from_txt_bash.sh $data_dir/$date_name
#     python /yuanhuan/code/demo/Image/detection2d/script/dataset/dataset_move/find_txt_data.py --input_dir=$data_dir/$date_name --input_jpg_name=JPEGImages --output_jpg_name=JPEGImages_test
# done

# 训练检测大模型
# python /yuanhuan/code/demo/Image/Basic/script/json/platform_json_to_coco/make_test_json_imageset_HuanWei.py