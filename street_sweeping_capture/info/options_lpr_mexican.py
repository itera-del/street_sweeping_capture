from easydict import EasyDict as edict

options = edict()


###########################################
# resolution
###########################################
# # 5M（按原始视频方向处理，输入算法尺寸为 2592x1920）
options.image_width = 2592
options.image_height = 1920

# 4M
# options.image_width = 2688
# options.image_height = 1520
# options.image_width = 2592
# options.image_height = 1520

# 2M
# options.image_width = 1920
# options.image_height = 1080
# options.image_width = 1080
# options.image_height = 1920

# # 720p
# options.image_width = 1280
# options.image_height = 720


###########################################
# gpu
###########################################
options.gpu_bool = True
# options.gpu_bool = False
# options.device = 'cpu'
options.device = 'cuda:0'


###########################################
# detector
###########################################
# lpr
options.ssd_bool = False
options.yolov6_bool = False
options.yolox_bool = True
# options.ssd_bool = True
# options.yolov6_bool = False
options.ssd_caffe_bool = True
options.ssd_openvino_bool = False

# # yolov6_rm_huanwei_8k
# options.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_rm_huanwei_8k/yolov6_rm_huanwei_deploy.py"
# options.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_rm_huanwei_8k/epoch_300_deploy.pth"
# options.yolov6_class_name = ["car", "license_plate"]
# options.yolov6_threshold_list = [0.4, 0.4, 0.4, 0.4, 0.4]

# yolox_large_rm_capture_6_class_512_288_20250915
options.yolox_bool = True
options.yolox_config = "/yuanhuan/code/demo/Image/detection2d/rm_ai_mmdet_3_3_yolox/my_configs/apc/yolox/Capture_all_yolox_s_512x288_car_bus_truck_cyclist_license_person_mhr.py"
options.yolox_checkpoint = "/yuanhuan/model/image/yolox/yolox_small_rm_capture_c28_6_class_512_288_20250913/epoch_300.pth"
options.yolox_class_name = ["car", "bus", "truck", "cyclist", "person", "license"]
# 夜间先放宽阈值，按模型类别顺序设置（最后一项为 plate/license）
options.yolox_threshold_list = [0.20, 0.20, 0.20, 0.20, 0.20, 0.05]


# 是否将 car\bus\truck 合并为一类输出
# options.object_attri_merge_bool = True
options.object_attri_merge_bool = False
options.object_attri_merge_name = 'car_bus_truck'
options.object_attri_name_list = [ 'car', 'bus', 'truck', 'cyclist' ]
options.license_plate_name = 'license'


###########################################
# options
###########################################
options.lpr_caffe_bool = False
options.lpr_pytorch_bool = False
options.lpr_onnx_bool = True

################
# mexican
################
options.mexican = edict()


# paddle_ocr_20240725_w_o_aug_w_o_diffste
options.lpr_paddle_bool = True
options.mexican.ocr_pth_path = ""
options.mexican.ocr_caffe_prototxt = ""
options.mexican.ocr_caffe_model_path = ""
options.mexican.ocr_onnx_model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_mexican_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_gray_64_256_20260415_092206/inference_500/onnx/model.onnx"
options.mexican.input_shape = (1, 64, 256)
options.mexican.ocr_labels_dict_path = "/defaultShare/mexican_lpr/Mexican_20260331_frames_5f_10060/paddle_dict_0414/mexican.txt"
options.mexican.ocr_prefix_beam_search_bool = False
options.mexican.padding_bool = False

# 车牌长宽阈值（夜间适当放宽下限）
options.plate_height = [12, 500]
options.plate_width = [12, 500]

options.lpr_ocr_width_expand_ratio = 0.00
options.lpr_ocr_column_threshold = 2.0


###########################################
# sort
###########################################
options.sort_type = "car"

options.max_age = 10
options.min_hits = 3 
options.iou_threshold = 0.1
options.sort_expand_ratio = 1.5
options.sort_class_name = ['license_plate']


###########################################
# cache
###########################################
# 缓存间隔
options.cache_interval = 2
# 缓存容器长度
options.cache_container_length = 8


###########################################
# state
###########################################
# 状态容器长度
options.bbox_state_container_length = 10       # 车辆框连续丢失上报，从容器中清除该车辆信息
options.lpr_ocr_state_container_length = 20    # 车牌状态长度阈值
options.lpr_city_state_container_length = 10   # 车牌状态长度阈值

# 更新车辆行驶状态
options.update_state_num_threshold = 5         # 车辆行驶状态计数最大值，用于记录车辆处于同一行驶状态的帧数
options.update_state_threshold = 1
options.update_state_stable_loc_alpha = float(0.6)   # 平滑车辆框参数


###########################################
# capture
###########################################
# 抓拍线
# options.capture_line_up_down_ratio = [0.03, 0.5, 0.9, 0.97]
# options.capture_line_left_right_ratio = [0.03, 0.25, 0.75, 0.97]
options.capture_line_up_down_ratio = [0.03, 0.45, 0.9, 0.97]
options.capture_line_left_right_ratio = [0.03, 0.25, 0.75, 0.97]

# 报警时间长短
options.capture_frame_num_threshold = 16
options.capture_clear_frame_num_threshold = 5 * 60 * 25       # 经过多少帧，抓拍容器清空

options.capture_info_frame_threshold = 2
options.capture_outtime_frame_threshold_01 = 10
options.capture_outtime_frame_threshold_02 = 150
options.capture_up_down_distance_boundary_threshold = 100
options.capture_left_right_distance_near_boundary_threshold = 200
options.capture_left_right_distance_far_boundary_threshold = 400

options.capture_lpr_score_threshold = 0.5
options.capture_lpr_num_frame_threshold = 2
options.capture_lpr_color_frame_threshold = 3


###########################################
# roi
###########################################

# 是否通过 roi 区域屏蔽部分检测结果
# options.roi_bool = False
options.roi_bool = True
options.roi_area = [0, 0, options.image_width, options.image_height]
# options.roi_area = [options.image_width * 0.25, 0, options.image_width * 0.75, options.image_height]

# 上下限阈值 & 左右限阈值
if options.roi_bool:
    options.ROI_Up_threshold = options.roi_area[1] + ( options.roi_area[3] - options.roi_area[1] ) * options.capture_line_up_down_ratio[0]
    options.ROI_Down_threshold = options.roi_area[1] + ( options.roi_area[3] - options.roi_area[1] ) * options.capture_line_up_down_ratio[3]
    options.Up_threshold = options.roi_area[1] + ( options.roi_area[3] - options.roi_area[1] ) * options.capture_line_up_down_ratio[1]
    options.Down_threshold = options.roi_area[1] + ( options.roi_area[3] - options.roi_area[1] ) * options.capture_line_up_down_ratio[2]
    options.ROI_Left_threshold = options.roi_area[0] + ( options.roi_area[2] - options.roi_area[0] ) * options.capture_line_left_right_ratio[0]
    options.ROI_Right_threshold = options.roi_area[0] + ( options.roi_area[2] - options.roi_area[0] ) * options.capture_line_left_right_ratio[3]
    options.Left_threshold = options.roi_area[0] + ( options.roi_area[2] - options.roi_area[0] ) * options.capture_line_left_right_ratio[1]
    options.Right_threshold = options.roi_area[0] + ( options.roi_area[2] - options.roi_area[0] ) * options.capture_line_left_right_ratio[2]
else:
    options.ROI_Up_threshold = options.image_height * options.capture_line_up_down_ratio[0]
    options.ROI_Down_threshold = options.image_height * options.capture_line_up_down_ratio[3]
    options.Up_threshold = options.image_height * options.capture_line_up_down_ratio[1]
    options.Down_threshold = options.image_height * options.capture_line_up_down_ratio[2]
    options.ROI_Left_threshold = options.image_width * options.capture_line_left_right_ratio[0]
    options.ROI_Right_threshold = options.image_width * options.capture_line_left_right_ratio[3]
    options.Left_threshold = options.image_width * options.capture_line_left_right_ratio[1]
    options.Right_threshold = options.image_width * options.capture_line_left_right_ratio[2]

###########################################
# strategy
###########################################
    
options.strategy = edict()
# options.strategy.only_one_plate = True      # 广州出租，只识别图像一张车牌
options.strategy.only_one_plate = False      # other