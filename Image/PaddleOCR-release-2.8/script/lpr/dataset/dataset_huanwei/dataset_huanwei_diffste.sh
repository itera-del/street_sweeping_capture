#!/bin/bash

data_dir=/yuanhuan/data/image/RM_HUANWEI/original/Argentina/Argentina
data_csv_dir=/yuanhuan/data/image/RM_HUANWEI/original/Argentina/Argentina_csv/
data_crop_dir=/yuanhuan/data/image/RM_HUANWEI/original/Argentina/Argentina_crop/
error_data_dir=/yuanhuan/data/image/RM_HUANWEI/original/Argentina/Argentina_error_data/
analysis_dir=/yuanhuan/data/image/RM_HUANWEI/original/Argentina/Argentina_analysis/
data_diffste_dir=/yuanhuan/data/image/RM_HUANWEI/original/Argentina/DIFFSTE/

ocr_name=plate_huanwei_20251030
ocr_diffste_name=plate_huanwei_20251030
training_data_dir=/yuanhuan/data/image/RM_HUANWEI/training_ocr/
training_diffste_data_dir=/yuanhuan/data/image/RM_HUANWEI/training_ocr/DIFFSTE

################################
# step 1：生成预标注标签 ocr（样本生成 char mask）
################################

# done
# date_name_list=(Argentina_20250910_有编号有垃圾桶 Argentina_20250820_有编号有垃圾桶 Argentina_20250925_有编号有垃圾桶)

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name
    
#     # 运行脚本 dataset_select/dataset_group.py，数据分组，按照车牌长度和字符分组 
#     python /yuanhuan/code/demo/Image/sd/script/diffste/dataset_plate/dataset_select/dataset_group.py --date_name=${date_name} --ocr_name=$ocr_name --input_dir=$training_data_dir
# done

# # done
# date_name_list=(DIFFSTE/original_20250820_20250925)

# img_name=JPEGImages
# xml_name=Annotations_Character_MMGroundingDINO
# xml_nms_name=Annotations_Character_MMGroundingDINO_NMS
# json_platform_name=Jsons_Prelabel_Platform

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name

#     # resize_padding
#     python /yuanhuan/code/demo/Image/sd/script/diffste/dataset_plate/dataset_pretreatment/dataset_resize_padding.py --input_dir=$data_dir/$date_name --jpg_name=$img_name --output_dir=$data_dir/${date_name}_scale_padding_256_128 --out_jpg_name=$img_name

#     # 检测大模型
#     # 镜像：pytorch-mmdet-mmseg-yuanhuan:py3-9_torch2-0_cuda11-7
#     export HF_ENDPOINT=https://hf-mirror.com
#     source /opt/conda/bin/activate base
#     conda init bash
#     conda activate mmdet
#     export http_proxy=http://192.168.151.254:7890 ; export https_proxy=http://192.168.151.254:7890

#     epoch=epoch_20
#     model_root=/yuanhuan/model/image/mm_grounding_dino/mm_grounding_dino_l_character
#     config=$model_root/grounding_dino_swin-l_finetune_character_rm.py
#     checkpoint=$model_root/$epoch.pth

#     image="$data_dir/${date_name}_scale_padding_256_128/$img_name/"
#     xml="$data_dir/${date_name}_scale_padding_256_128/$xml_name/"
#     xml_nms="$data_dir/${date_name}_scale_padding_256_128/$xml_nms_name/"

#     cd /yuanhuan/code/demo/Image/detection2d/ori_mmdetection/mmdetection/
#     python ./demo/image_demo_xml.py \
#         "$image" \
#         "$config" \
#         --weights "$checkpoint" \
#         --out-dir "$xml" \
#         --save-xml \
#         --batch-size 4 \
#         --texts 'char .'

#     python /yuanhuan/code/demo/Image/Basic/script/xml/xml_nms_dupes.py \
#         --jpg_dir "$image" \
#         --input_xml_dir "$xml" \
#         --output_xml_dir "$xml_nms"

#     # xml 2 platform json
#     python /yuanhuan/code/demo/Image/sd/script/diffste/dataset_plate/dataset_xml/xml_to_platform_json_LPR_char_bbox.py --input_dir=$data_dir/${date_name}_scale_padding_256_128 --jpg_name=$img_name --xml_name=$xml_nms_name --json_name=$json_platform_name

# done

# ################################
# # step 1.2：检查标注字符结构
# ################################

# # done
# date_name_list=(original_20250820_20250925_scale_padding_256_128)

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name
    
#     # 运行脚本 dataset_select/dataset_group.py，数据分组，按照车牌长度和字符分组 
#     python /yuanhuan/code/demo/Image/sd/script/diffste/dataset_plate/dataset_xml/platform_json_to_xml_char.py --input_dir=${data_diffste_dir}/${date_name} --jpg_name=JPEGImages --json_name=Jsons_manual --xml_name=Annotations
#     python /yuanhuan/code/demo/Image/sd/script/diffste/dataset_plate/dataset_check/dataset_check_annotation.py --input_dir=${data_diffste_dir}/${date_name} --jpg_name=JPEGImages --json_name=Jsons_manual --xml_name=Annotations
# done


# ################################
# # step 1.3：准备数据集
# ################################

# # done
# date_name_list=(original_20250820_20250925_scale_padding_256_128)

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name
    
#     python /yuanhuan/code/demo/Image/sd/script/diffste/dataset_plate/dataset_train_test_split/data_train_test_split.py --input_dir=${data_diffste_dir}/${date_name} --jpg_name=JPEGImages --xml_name=Annotations
#     python /yuanhuan/code/demo/Image/sd/script/diffste/dataset_plate/dataset_pretreatment/dataset_move.py --input_dir=${data_diffste_dir}/${date_name} --output_dir=$training_diffste_data_dir/${date_name} --jpg_name=JPEGImages --xml_name=Annotations
#     python /yuanhuan/code/demo/Image/sd/script/diffste/dataset_plate/dataset_pretreatment/dataset_mask.py --input_dir=$training_diffste_data_dir/${date_name}
# done

# ################################
# # step 1.4：样本生成结果验收
# ################################

# # done
# date_name_list=(diffste_huanwei_20250820_20250925_3w)

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name
    
#     python /yuanhuan/code/demo/Image/sd/script/diffste/dataset_plate/dataset_check/dataset_check.py --input_dir=$data_diffste_dir/${date_name} --output_dir=$data_diffste_dir/${date_name}_check
#     # 需要手动选择设置需要剔除的样本
#     # python /yuanhuan/code/demo/Image/sd/script/diffste/dataset_plate/dataset_check/dataset_remove.py --input_dir=$data_diffste_dir/${date_name}
# done


# ################################
# step 2：生成训练数据 ocr
# ################################

# # done
# date_name_list=(diffste_huanwei_test_40w diffste_huanwei_20240910_20240911_1018_60w diffste_huanwei_20240912_323_20w)
# # date_name_list=(diffste_huanwei_20250820_20250925_3w)

# # todo
# # date_name_list=()

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name

#     # diffste ocr
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_huanwei/dataset_mask/gen_ocr_img_diffste.py --date_name=$date_name --ocr_name=$ocr_diffste_name --input_dir=$data_diffste_dir --output_dir=$training_data_dir
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_huanwei/dataset_train_test_split/data_train_test_split_ocr.py --date_name=$date_name --ocr_name=$ocr_diffste_name --input_dir=$training_data_dir

# done

# ocr & diffste ocr
# python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_huanwei/dataset_train_test_split/data_train_test_split_ocr_merge.py --ocr_name=$ocr_name --input_dir=$training_data_dir --diffste