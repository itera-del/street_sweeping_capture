#!/opt/conda/bin bash
data_closed_dir=/yuanhuan/data/dataset_closed_loop/frames/capture/
data_dir=/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore/
data_csv_dir=/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_csv/
data_crop_dir=/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_crop/
data_crop_csv_dir=/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_crop_csv/
data_crop_scale_padding_dir=/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_crop_scale_padding
error_data_dir=/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_error_data/
error_crop_data_dir=/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_error_crop_data/
analysis_dir=/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_analysis/
analysis_crop_dir=/yuanhuan/data/image/RM_ParkingNumber/original/Singapore/Singapore_analysis_crop/

#################################
# step 1：处理标注数据
#################################

# done
date_name_list=(Singapore_202504_1080p Singapore_202504_D1)

for date_name in ${date_name_list[@]}; do 
    echo $date_name

    python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_singapore_parking_number/dataset_check/dataset_csv.py --date_name=$date_name --input_dir=$data_dir --output_csv_dir=$data_csv_dir --output_crop_data_dir=$data_crop_dir --output_error_data_dir=$error_data_dir --new_style --bool_write_error_data --bool_write_crop_data --bool_check_img --json_folder=Json_manual

done