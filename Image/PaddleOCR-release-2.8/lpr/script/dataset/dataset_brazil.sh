######################
# Brazil_motor
######################

# done
# date_name_list=(2022_0914 2022_0915 2022_0916 2022_0919 2022_0930 2022_1001 2022_1001_1 2022_1003 2022_1004 2022_1005 2022_1007 2022_1012 2022_1013 2022_1026 2023_0110)

# todo
date_name_list=(2023_0110)
ocr_name=plate_brazil

for date_name in ${date_name_list[@]}; do 
    echo $date_name
    # normal
    # 策略：构建两个 ocr 模型，分别识别摩托车车牌第一行信息、第二行信息
    data_dir=/yuanhuan/data/image/LicensePlate_ocr/original/Brazil/Brazil/Brazil_motor/$date_name
    crop_data_dir=/yuanhuan/data/image/LicensePlate_ocr/original/Brazil/Brazil_crop/Brazil_motor/crop_$date_name
    ocr_dir=/yuanhuan/data/image/LicensePlate_ocr/training/$ocr_name/
    ocr_data_dir=$ocr_dir/data_crop_$date_name

    python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_ocr_brazil/dataset_group/dataset_2_crop.py --input_dir=$data_dir --output_dir=$crop_data_dir
    python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_ocr_brazil/dataset_mask/gen_ocr_img.py --input_dir=$crop_data_dir --output_dir=$ocr_data_dir
    python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_ocr_brazil/dataset_mask/gen_ocr_img_augment.py --input_dir=$data_dir --output_dir=$ocr_data_dir
    python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_ocr_brazil/dataset_train_test_split/data_train_test_split.py --input_dir=$ocr_data_dir
    python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_ocr_brazil/dataset_train_test_split/data_train_test_split_copy.py --input_dir=$ocr_data_dir
    python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_ocr_brazil/dataset_train_test_split/data_train_test_split_merge.py --input_dir=$ocr_dir
    python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_ocr_brazil/dataset_label/data_label_split.py --input_dir=$ocr_dir

    # # 2_line
    # # 策略：手工将摩托车车牌切分为两行，用一个 ocr 模型识别两行内容，效果一般
    # data_dir=/yuanhuan/data/image/LicensePlate_ocr/original/Brazil/Brazil/Brazil_motor/$date_name
    # crop_data_dir=/yuanhuan/data/image/LicensePlate_ocr/original/Brazil/Brazil_crop/Brazil_motor/crop_$date_name
    # ocr_dir=/yuanhuan/data/image/LicensePlate_ocr/training/plate_brazil_2_line/
    # ocr_data_dir=$ocr_dir/data_crop_$date_name

    # python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_ocr_brazil/dataset_group/dataset_2_crop.py --input_dir=$data_dir --output_dir=$crop_data_dir
    # python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_ocr_brazil/dataset_mask/gen_ocr_img_2_line.py --input_dir=$crop_data_dir --output_dir=$ocr_data_dir
    # python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_ocr_brazil/dataset_mask/gen_ocr_img_2_line_augment.py --input_dir=$data_dir --output_dir=$ocr_data_dir
    # python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_ocr_brazil/dataset_train_test_split/data_train_test_split.py --input_dir=$ocr_data_dir
    # python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_ocr_brazil/dataset_train_test_split/data_train_test_split_copy.py --input_dir=$ocr_data_dir
    # python /yuanhuan/code/demo/Image/recognition2d/lpr/script/dataset/dataset_ocr_brazil/dataset_train_test_split/data_train_test_split_merge.py --input_dir=$ocr_dir
done