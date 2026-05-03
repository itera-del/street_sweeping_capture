from easydict import EasyDict as edict

options = edict()

###########################################
# resolution
###########################################
# # 5M
# options.image_width = 2592
# options.image_height = 1920

# other
options.image_width = 2592
options.image_height = 1520
# options.image_width = 2688
# options.image_height = 1520

# # 2M
# options.image_width = 1920
# options.image_height = 1080

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
# face
options.ssd_bool = False
options.yolov6_bool = True
options.ssd_caffe_bool = False
options.ssd_openvino_bool = False

# yolov6 face person
options.landmark_bool = False
options.landmark_degree_bool = False
options.landmark_degree_cls_bool = False
options.face_peron_bool = True
options.yolov6_config = "/mnt/huanyuan/model/image/yolov6/yolov6_zg_capture_face_20240720_45W_512_288/yolov6_zg_capture_face_512_288_deploy.py"
options.yolov6_checkpoint = "/mnt/huanyuan/model/image/yolov6/yolov6_zg_capture_face_20240720_45W_512_288/epoch_300_deploy.pth"
options.yolov6_class_name = ['face','side_face','face_occlusion','person','bicyclist','motorcyclist']
options.yolov6_threshold_list = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]

# 是否将 person\bicyclist\motorcyclist 合并为一类输出
options.object_attri_merge_bool = False
options.object_attri_merge_name = 'person_bicyclist_motorcyclist'
options.object_attri_name_list = [ 'person', 'bicyclist', 'motorcyclist' ]
options.face_name = 'face'
options.filter_face_by_person_bool = True

###########################################
# sort
###########################################
options.sort_type = "face"
# options.sort_type = "person"

options.max_age = 10
options.min_hits = 3 
options.iou_threshold = 0.1
options.sort_expand_ratio = 1.5
options.sort_class_name = ['face']


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
options.lpr_face_state_container_length = 10   # 人脸状态长度阈值

# 更新车辆行驶状态
options.update_state_num_threshold = 5         # 车辆行驶状态计数最大值，用于记录车辆处于同一行驶状态的帧数
options.update_state_threshold = 1
options.update_state_stable_loc_alpha = float(0.6)   # 平滑车辆框参数


###########################################
# capture
###########################################
# 抓拍线
options.capture_line_up_down_ratio = [0.03, 0.5, 0.9, 0.97]
options.capture_line_left_right_ratio = [0.03, 0.25, 0.75, 0.97]

# 报警时间长短
options.capture_frame_num_threshold = 16
options.capture_clear_frame_num_threshold = 60 * 25        # 经过多少帧，抓拍容器清空

options.capture_info_frame_threshold = 5
options.capture_outtime_frame_threshold_01 = 25
options.capture_outtime_frame_threshold_02 = 150
options.capture_up_down_distance_boundary_threshold = 100
options.capture_left_right_distance_near_boundary_threshold = 200
options.capture_left_right_distance_far_boundary_threshold = 400

options.capture_face_landmark_degree_threshold = 0.5
options.face_landmark_degree_frame_threshold  = 4

# 人脸抓拍尺寸
options.face_size_threshold = [100, 100]


###########################################
# roi
###########################################

# 是否通过 roi 区域屏蔽部分检测结果
options.roi_bool = False
# options.roi_bool = True
options.roi_area = [0, 0, options.image_width, options.image_height]

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