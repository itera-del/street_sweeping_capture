#!/opt/conda/bin bash

data_closed_dir=/yuanhuan/data/dataset_closed_loop/frames/capture
data_dir=/yuanhuan/data/image/RM_ANPR/original/zd/Shate/Shate/
data_csv_dir=/yuanhuan/data/image/RM_ANPR/original/zd/Shate/Shate_csv/
data_crop_dir=/yuanhuan/data/image/RM_ANPR/original/zd/Shate/Shate_crop/
data_crop_csv_dir=/yuanhuan/data/image/RM_ANPR/original/zd/Shate/Shate_crop_csv/
data_crop_scale_padding_dir=/yuanhuan/data/image/RM_ANPR/original/zd/Shate/Shate_crop_scale_padding
error_data_dir=/yuanhuan/data/image/RM_ANPR/original/zd/Shate/Shate_error_data/
error_crop_data_dir=/yuanhuan/data/image/RM_ANPR/original/zd/Shate/Shate_error_crop_data/
analysis_dir=/yuanhuan/data/image/RM_ANPR/original/zd/Shate/Shate_analysis/
analysis_crop_dir=/yuanhuan/data/image/RM_ANPR/original/zd/Shate/Shate_analysis_crop/

diffste_name=diffste_2094_shate
diffste_dir=/yuanhuan/data/image/RM_ANPR/original/zd/Shate/Shate_crop_scale_padding/DIFFSTE

seg_name=seg_shate_202508
ocr_name=plate_shate_mask_202508
training_data_dir=/yuanhuan/data/image/RM_ANPR/training/

# #################################
# # step 0：数据拷贝：将需要标注的图片拷贝到对应的目录下
# #################################
# # done
# date_name_list=(20241230 20241231 20250101 20250102 20250105 20250106 20250107 20250115)    # capture_shate_${date_name}_frames_5f

# source /opt/conda/bin/activate base
# conda init bash
# conda activate mmdet

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name
    
#     # capture_shate_${date_name}_frames_5f 20241230 20241231 20250101 20250102 20250105 20250106 20250107 20250115
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_move/data_move.py --input_dir=$data_closed_dir/capture_shate_${date_name}_frames_5f --output_dir=$data_dir/shate_${date_name}
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_move/data_flip.py --input_dir=$data_dir/shate_${date_name}
#     python /yuanhuan/code/demo/Image/Basic/script/xml/xml_to_platform_json_LPR.py --input_dir=$data_dir/shate_${date_name}

# done

# #################################
# # step 1：处理标注数据
# #################################

# # done
# date_name_list=(shate_20241230 shate_20241231 shate_20250101 shate_20250102 shate_20250105 shate_20250106 shate_20250107 shate_20250115)

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name

#     # new style
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_check/dataset_csv.py --date_name=$date_name --input_dir=$data_dir --output_csv_dir=$data_csv_dir --output_crop_data_dir=$data_crop_dir --output_error_data_dir=$error_data_dir --new_style --bool_write_error_data --bool_write_crop_data --bool_check_img --json_folder=Json_manual

#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_analysis/analysis_dataset_label_num.py --date_name=$date_name --input_csv_dir=$data_csv_dir --output_analysis_dir=$analysis_dir

# done

# python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_analysis/analysis_dataset_label_num_merge.py --input_dir=$analysis_dir

# #################################
# # step 2：生成预标注标签 seg
# #################################

# # done
# date_name_list=(shate_20241230 shate_20241231 shate_20250101 shate_20250102 shate_20250105 shate_20250106 shate_20250107 shate_20250115)

# img_name=Images
# json_name=Jsons_Prelabel
# json_platform_name=Jsons_Prelabel_Platform
# xml_name=Xmls_Prelabel
# json_color_name=Jsons_Prelabel_Color
# json_city_name=Jsons_Prelabel_City
# xml_color_name=Xmls_Prelabel_Color
# xml_city_name=Xmls_Prelabel_City

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name

#     # # seg 板端模型
#     # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_pre_label/dataset_pre_label_seg_mask.py --date_name=$date_name --input_crop_data_dir=$data_crop_dir --json_name=$json_name
#     # python /yuanhuan/code/demo/Image/Basic/script/json/platform_json_to_xml.py --input_dir=$data_crop_dir/$date_name --jpg_name=$img_name --json_name=$json_name --xml_name=$xml_name

#     # seg 分割大模型
#     # 镜像：pytorch-mmdet-mmseg-miniforge-yuanhuan:py3-9_torch2-0_cuda11-7
#     source /opt/conda/bin/activate base
#     conda init bash
#     conda activate mmdet
#     IMGS=$data_crop_dir/$date_name/$img_name
#     CONFIG=/yuanhuan/model/image/seg/mask2former_r50_8xb2-rm_anpr_zd_seg_color-512x1024/20250206/mask2former_r50_8xb2-90k_rm_anpr_zd_seg_color-512x1024.py
#     CHECKPOINT=/yuanhuan/model/image/seg/mask2former_r50_8xb2-rm_anpr_zd_seg_color-512x1024/20250206/iter_90000.pth
#     OUTPUT=$data_crop_dir/$date_name
#     cd /yuanhuan/code/demo/Image/segmentation2d/mmsegmentation
#     python /yuanhuan/code/demo/Image/segmentation2d/mmsegmentation/demo/image_dir_demo_w_json.py --imgs $IMGS --config $CONFIG --checkpoint $CHECKPOINT --outdir $OUTPUT --json_name=$json_color_name --bool_save_json --bool_json_seg_rect
#     python /yuanhuan/code/demo/Image/Basic/script/json/platform_json_to_xml.py --input_dir=$data_crop_dir/$date_name --jpg_name=$img_name --json_name=$json_color_name --xml_name=$xml_color_name

#     CONFIG=/yuanhuan/model/image/seg/mask2former_r50_8xb2-rm_anpr_zd_seg_city-512x1024/20250206/mask2former_r50_8xb2-90k_rm_anpr_zd_seg_city-512x1024.py
#     CHECKPOINT=/yuanhuan/model/image/seg/mask2former_r50_8xb2-rm_anpr_zd_seg_city-512x1024/20250206/iter_90000.pth
#     OUTPUT=$data_crop_dir/$date_name
#     cd /yuanhuan/code/demo/Image/segmentation2d/mmsegmentation
#     python /yuanhuan/code/demo/Image/segmentation2d/mmsegmentation/demo/image_dir_demo_w_json.py --imgs $IMGS --config $CONFIG --checkpoint $CHECKPOINT --outdir $OUTPUT --json_name=$json_city_name --bool_save_json --bool_json_seg_rect
#     python /yuanhuan/code/demo/Image/Basic/script/json/platform_json_to_xml.py --input_dir=$data_crop_dir/$date_name --jpg_name=$img_name --json_name=$json_city_name --xml_name=$xml_city_name

#     python /yuanhuan/code/demo/Image/Basic/script/xml/xml_merge.py --input_dir=$data_crop_dir/$date_name --jpg_name=$img_name --xml_name=$xml_color_name  --xml_2_name=$xml_city_name --xml_out_name=$xml_name

#     # xml 2 platform json
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_xml_json/xml_to_platform_json_LPR_zd_seg_mask.py --input_dir=$data_crop_dir/$date_name --jpg_name=$img_name --xml_name=$xml_name --json_name=$json_platform_name

# done


# #################################
# # step 3：生成训练数据 seg
# ################################

# # done
# date_name_list=(shate_20241230 shate_20241231 shate_20250101 shate_20250102 shate_20250105 shate_20250106 shate_20250107 shate_20250115)

# # todo
# # date_name_list=()

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name

#     # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd_shate/dataset_check/dataset_crop_csv.py --date_name=$date_name --input_csv_dir=$data_csv_dir --input_crop_data_dir=$data_crop_dir --output_csv_dir=$data_crop_csv_dir --output_error_crop_data_dir=$error_crop_data_dir --bool_write_error_data
#     # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd_shate/dataset_analysis/analysis_dataset_crop_label_num.py --date_name=$date_name --input_csv_dir=$data_crop_csv_dir --output_analysis_dir=$analysis_crop_dir --new_style
    
#     # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd_shate/dataset_mask/gen_seg_mask.py --date_name=$date_name --seg_name=$seg_name --input_csv_dir=$data_crop_csv_dir --output_dir=$training_data_dir --new_style
#     # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd_shate/dataset_train_test_split/data_train_test_split_seg.py --date_name=$date_name --seg_name=$seg_name --input_dir=$training_data_dir

#     # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd_shate/dataset_mask/gen_seg_mask_augment.py --date_name=$date_name --seg_name=$seg_name --output_dir=$training_data_dir --new_style
#     # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_train_test_split/data_train_test_split_seg_augment.py --date_name=$date_name --seg_name=$seg_name --input_dir=$training_data_dir
# done

# python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_train_test_split/data_train_test_split_seg_merge.py --seg_name=$seg_name --input_dir=$training_data_dir
# python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_analysis/analysis_dataset_crop_label_num_merge.py --input_dir=$analysis_crop_dir

# ################################
# # step 4：生成训练数据 ocr
# ################################

# # done
# # date_name_list=(shate_20241230 shate_20241231 shate_20250101 shate_20250102 shate_20250105 shate_20250106 shate_20250107 shate_20250115)
# # done ocr diffste
# date_name_list=(shate_20241230 shate_20241231 shate_20250101 shate_20250102 shate_20250105 shate_20250106 shate_20250107 shate_20250115)

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name
    
#     # ocr
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd_shate/dataset_mask/gen_ocr_img.py --date_name=$date_name --seg_name=$seg_name --ocr_name=$ocr_name --output_dir=$training_data_dir
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_train_test_split/data_train_test_split_ocr.py --date_name=$date_name --ocr_name=$ocr_name --input_dir=$training_data_dir
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd_shate/dataset_mask/gen_ocr_img_augment.py --date_name=$date_name --seg_name=$seg_name --ocr_name=$ocr_name --output_dir=$training_data_dir
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_train_test_split/data_train_test_split_ocr_augment.py --date_name=$date_name --ocr_name=$ocr_name --input_dir=$training_data_dir

#     # ocr diffste
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd_shate/dataset_mask/gen_ocr_img_diffste.py --date_name=$date_name --seg_name=$seg_name --ocr_name=$ocr_name --output_dir=$training_data_dir --diffste_name $diffste_name --diffste_dir $diffste_dir
# done

# # ocr diffste
# python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_train_test_split/data_train_test_split_ocr.py --date_name=$diffste_name --ocr_name=$ocr_name --input_dir=$training_data_dir

# python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_train_test_split/data_train_test_split_ocr_merge.py --ocr_name=$ocr_name --input_dir=$training_data_dir

# #################################
# # step 5：2 paddleocr label
# #################################
# # 2 paddleocr label
# paddle_ocr_name=plate_shate_mask_202509_Diffste_NoAug
# image_set_name=ImageSetsOcrLabelNoAug
# # paddle_ocr_name=plate_shate_mask_202509_Diffste
# # image_set_name=ImageSetsOcrLabel
# paddle_ocr_data_dir=/yuanhuan/model/image/lpr/paddle_dict/$paddle_ocr_name
# python /yuanhuan/code/demo/Image/recognition2d/script/paddle/dataset/lpr_to_paddleocr_label.py --input_dir=$training_data_dir/$ocr_name --image_set_name=$image_set_name --output_dir=$paddle_ocr_data_dir
# # python /yuanhuan/code/demo/Image/recognition2d/script/paddle/dataset/lpr_to_paddleocr_label.py --input_dir=$training_data_dir/$ocr_name --image_set_name=$image_set_name --output_dir=$paddle_ocr_data_dir --random_scale=0.2
# python /yuanhuan/code/demo/Image/recognition2d/script/paddle/dataset/label_dict.py --output_dir=$paddle_ocr_data_dir --output_name=zd_dict.txt --data_dict_name=script.lpr.dataset.dataset_zd.dataset_dict.dataset_zd_dict_normal

# ################################
# # step 6：生成预标注标签 ocr（样本生成 char mask）
# ################################

# # done
# date_name_list=(shate_20250101 shate_20250102 shate_20250105 shate_20250106 shate_20250107 shate_20250115)

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name
    
#     # 运行脚本 dataset_select/dataset_group.py，数据分组，按照车牌长度和字符分组 
#     python /yuanhuan/code/demo/Image/sd/script/diffste/dataset_plate/dataset_select/dataset_group.py --date_name=${date_name}/city_label --ocr_name=$seg_name --input_dir=$training_data_dir
# done

# # todo
# date_name_list=(DIFFSTE/original_2094_Shate)

# img_name=Images
# xml_name=Xmls_Prelabel
# xml_nms_name=Xmls_Prelabel_NMS
# json_platform_name=Jsons_Prelabel_Platform

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name

#     # resize_padding
#     python /yuanhuan/code/demo/Image/sd/script/diffste/dataset_plate/dataset_pretreatment/dataset_resize_padding.py --input_dir=$data_crop_dir/$date_name --jpg_name=$img_name --output_dir=$data_crop_scale_padding_dir/$date_name --out_jpg_name=$img_name

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

#     image="$data_crop_scale_padding_dir/$date_name/$img_name/"
#     xml="$data_crop_scale_padding_dir/$date_name/$xml_name/"
#     xml_nms="$data_crop_scale_padding_dir/$date_name/$xml_nms_name/"

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
#     python /yuanhuan/code/demo/Image/sd/script/diffste/dataset_plate/dataset_xml/xml_to_platform_json_LPR_char_bbox.py --input_dir=$data_crop_scale_padding_dir/$date_name --jpg_name=$img_name --xml_name=$xml_nms_name --json_name=$json_platform_name

# done

# #################################
# # step 7：获得量化图片
# #################################

# python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_move/quantization_data_ocr.py --input_dir=$training_data_dir/$ocr_name --split_dir=ImageSetsOcrLabelNoAugNoDiffste
python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_move/quantization_data_seg.py --input_dir=$training_data_dir/$seg_name --split_dir=ImageSetsLabelNoAug