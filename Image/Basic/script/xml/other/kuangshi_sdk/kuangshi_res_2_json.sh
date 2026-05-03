#!/usr/bin/env bash

#从旷世板端res到可训练的image和json
root_dir=/yuanhuan/data/image/RM_Capture/original/20240709_haidate/picture_7_12
jpg_dir=$root_dir/JPEGImages_dupes_0_95   #照片地址
sdk_res_dir=$root_dir/plate #旷世板子标出来的地址
sdk_xml_dir=$root_dir/Annotations_dupes_0_95_kuangshi_sdk #通过旷世res生成xml地址
gd_xml_dir=$root_dir/Annotations_dupes_0_95_Captue_MMGroundingDINO_NMS #通过大模型标注xml地址
output_xml_dir=$root_dir/Annotations_dupes_0_95_Captue_MMGroundingDINO_NMS_w_plate #修正后xml地址
output_jpg_dir=$root_dir/JPEGImages_dupes_0_95_w_plate #修正后xml地址
output_json_dir=$root_dir/Json_w_plate #通过生成的json地址
container_dir=/yuanhuan/data/image/RM_ANPR/original/cn_202406/china/202407_ChongqingjiangsuHunan

#旷世板端res转xml
python /yuanhuan/code/demo/Image/Basic/script/xml/other/kuangshi_sdk/kaungshi_sdk_res_2_xml.py \
        --input_dir $jpg_dir \
        --sdk_res_dir $sdk_res_dir \
        --output_xml_dir $sdk_xml_dir

#xml修正(grouding dino)
python /yuanhuan/code/demo/Image/Basic/script/xml/other/kuangshi_sdk/kuangshi_xml_change.py \
        --img_dir $jpg_dir \
        --xml_dir $sdk_xml_dir \
        --xml_2_dir $gd_xml_dir \
        --output_xml_dir $output_xml_dir 

python /yuanhuan/code/demo/Image/Basic/script/xml/other/kuangshi_sdk/find_data_xml.py \
        --input_dir $jpg_dir \
        --input_xml_dir $output_xml_dir \
        --output_jpg_folder $output_jpg_dir

#to json
python /yuanhuan/code/demo/Image/Basic/script/xml/xml_to_platform_json_LPR.py \
        --jpg_dir $output_jpg_dir \
        --xml_dir $output_xml_dir \
        --platform_json_dir $output_json_dir 

# copy
python /yuanhuan/code/demo/Image/Basic/script/xml/other/kuangshi_sdk/copy_data.py \
        --jpg_dir=$output_jpg_dir \
        --xml_dir=$gd_xml_dir \
        --json_dir=$output_json_dir \
        --output_dir=$container_dir