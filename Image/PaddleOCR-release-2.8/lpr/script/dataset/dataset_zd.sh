# #################################
# # step 1：处理 车牌检测框 标注数据
# #################################
# # done
# # date_name_list=(0804_0809 0810_0811 0828_0831 0901_0903 0904_0905 1024_1029 1024_1029_1080p 1115_1116 1115_1116_1080p)

# # todo
# date_name_list=(1115_1116_1080p)

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name
#     # normal
#     data_dir=/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE/check_$date_name
#     crop_data_dir=/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_$date_name

    
#     python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_seg_zd/dataset_group/dataset_group_2_crop.py --input_dir=$data_dir --output_dir=$crop_data_dir

# done

#################################
# step 2：处理 车牌属性 标注数据
#################################
# done
# date_name_list=(0828_0831 0901_0903 0904_0905 1024_1029 1024_1029_1080p 1115_1116 1115_1116_1080p 0 1 2 3 4 5 6 7 8 color_green_0 color_yellow_0 city_AJMAN_0 city_FUJAIRAH_0 city_RAK_0 city_SHARJAH_0 city_UMMALQAIWAIN_0)

# todo
date_name_list=( )
data_check_list=(0 1 2 3 4 5 6 7 8 color_green_0 color_yellow_0 city_AJMAN_0 city_FUJAIRAH_0 city_RAK_0 city_SHARJAH_0 city_UMMALQAIWAIN_0)
seg_name=seg_zd_202301
ocr_name=plate_zd_mask_202301
seg_dir=/yuanhuan/data/image/LicensePlate_ocr/training/$seg_name/
ocr_dir=/yuanhuan/data/image/LicensePlate_ocr/training/$ocr_name/

for date_name in ${date_name_list[@]}; do 
    echo $date_name
    # normal
    if [[ "${data_check_list[*]}"  =~ "${date_name}" ]];then
        data_dir=/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE/check
    else
        data_dir=/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE/check_$date_name
    fi
    
    crop_data_dir=/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/UAE_crop/check_crop_$date_name
    seg_data_dir=$seg_dir/data_crop_$date_name
    ocr_data_dir=$ocr_dir/data_crop_$date_name
    
    python /yuanhuan/code/demo/Image/Basic/script/json/platform_json_to_xml.py --input_dir=$crop_data_dir
    python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_seg_zd/dataset_group/dataset_group_check.py --input_dir=$crop_data_dir --output_dir=$seg_data_dir

    # segs
    python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_seg_zd/dataset_mask/gen_seg_mask.py --input_dir=$seg_data_dir
    python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_seg_zd/dataset_mask/gen_seg_mask_augment.py --input_dir=$seg_data_dir --ori_input_dir=$data_dir
    python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_seg_zd/dataset_train_test_split/data_train_test_split_seg.py --input_dir=$seg_data_dir
    python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_seg_zd/dataset_train_test_split/data_train_test_split_seg_copy.py --input_dir=$seg_data_dir
    python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_seg_zd/dataset_train_test_split/data_train_test_split_merge.py --input_dir=$seg_dir
    python /yuanhuan/code/demo/Image/recognition2d/lpr/script/analysis_dataset/dataset_seg_zd/analysis_dataset_seg_mask_label_num.py --input_dir=$seg_data_dir

    # ocr
    python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_ocr_zd/dataset_mask/gen_ocr_img.py --input_dir=$seg_data_dir --output_dir=$ocr_data_dir
    python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_ocr_zd/dataset_mask/gen_ocr_img_augment.py --input_dir=$seg_data_dir --output_dir=$ocr_data_dir
    python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_ocr_zd/dataset_train_test_split/data_train_test_split_ocr_copy.py --input_dir=$seg_data_dir --output_dir=$ocr_data_dir
    python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_ocr_zd/dataset_train_test_split/data_train_test_split_merge.py --input_dir=$ocr_dir
    python /yuanhuan/code/demo/Image/recognition2d/lpr/script/analysis_dataset/dataset_ocr_zd/analysis_dataset_ocr_label_num.py --input_dir=$ocr_data_dir
done

# done
# # ocr: check -> check_crop -> data_crop
# # ocr_data_dir=$ocr_dir/data_crop
# python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset_pre_label/dataset_ocr_zd/dataset_pre_img_ocr.py --output_dir=$ocr_data_dir
# python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_ocr_zd/dataset_train_test_split/data_train_test_split_ocr.py --input_dir=$ocr_data_dir
# python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_ocr_zd/dataset_train_test_split/data_train_test_split_merge.py --input_dir=$ocr_dir

# 2 paddleocr label
paddle_ocr_name=$ocr_name
paddle_ocr_data_dir=/yuanhuan/model/image/lpr/paddle_dict/$paddle_ocr_name
python /yuanhuan/code/demo/Image/recognition2d/script/paddle/dataset/lpr_to_paddleocr_label.py --input_dir=$ocr_dir --output_dir=$paddle_ocr_data_dir
paddle_ocr_data_dir=$ocr_dir/ImageSetsPaddle
python /yuanhuan/code/demo/Image/recognition2d/script/paddle/dataset/lpr_to_paddleocr_label.py --input_dir=$ocr_dir --output_dir=$paddle_ocr_data_dir