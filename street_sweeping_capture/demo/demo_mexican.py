import argparse
import cv2
import copy
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *
from Image.Demo.street_sweeping_capture.demo.RMAI_API_stabel_mexican import CaptureApi
from Image.Demo.street_sweeping_capture.utils.draw_tools import DrawApi 
from Image.Basic.script.xml.xml_write import write_xml


def inference_video(args):
    # mkdir 
    os.makedirs(args.output_video_dir, exist_ok=True)

    # capture api
    capture_api = CaptureApi(args.demo_type, args.country_type)

    # draw api
    draw_api = DrawApi(args.demo_type, args.country_type)

    # video init 
    video_list = np.array(os.listdir(args.video_dir))
    video_list = video_list[[video.endswith(args.suffix) for video in video_list]]
    video_list.sort()

    if not args.clear_state_per_video:
        
        frame_idx = 0

        # capture api 
        # 输入新视频，状态复位
        capture_api.clear()

    for idx in tqdm(range(len(video_list))):
        video_path = os.path.join(args.video_dir, video_list[idx])
        debug_log_f = None

        cap = cv2.VideoCapture(video_path) 
        print(int(cap.get(cv2.CAP_PROP_FPS)))              # 得到视频的帧率
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))           # 得到视频的宽
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))          # 得到视频的高
        print(cap.get(cv2.CAP_PROP_FRAME_COUNT))           # 得到视频的总帧

        output_video_path = os.path.join(args.output_video_dir, video_list[idx])
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        # 逐帧调试日志：记录检测框数量、跟踪数量、OCR结果，便于定位“无框/无识别”问题
        if args.write_debug_log_per_frame_bool:
            debug_log_path = os.path.join(
                args.output_video_dir,
                video_list[idx].replace(args.suffix, '_debug_log.txt')
            )
            debug_log_f = open(debug_log_path, "w", encoding="utf-8")
            debug_log_f.write("frame_idx\tcar_det\tplate_det\ttrack_num\tocr_num\traw_total\tpre_min_total\tpost_nms_total\traw_label_stats\tocr_detail\n")

        if args.clear_state_per_video:

            frame_idx = 0

            # capture api 
            # 输入新视频，状态复位
            capture_api.clear()

        # 是否保存视频结果
        if args.write_result_video_bool:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (capture_api.options.image_width, capture_api.options.image_height), True)

        while True:
            ret, img = cap.read()

            if not ret: # if the camera over return false
                break

            if frame_idx % args.frame_step == 0:
                # print(frame_idx)
                pass
            else:
                frame_idx += 1
                continue
            
            # if frame_idx == 39:
            #     print()

            # capture api（按原始方向处理，不做 90 度旋转）
            bboxes, tracker_bboxes, bbox_info_list, bbox_state_container, capture_line_points, capture_container, capture_res_container = capture_api.run(img, frame_idx)

            if args.write_debug_log_per_frame_bool and debug_log_f is not None:
                car_det_num = 0
                for car_name in capture_api.options.object_attri_name_list:
                    car_det_num += len(bboxes.get(car_name, []))
                plate_det_num = len(bboxes.get(capture_api.options.license_plate_name, []))
                track_num = len(tracker_bboxes)
                ocr_list = []
                for info in bbox_info_list:
                    plate_num = info.get('plate_info', {}).get('num', '')
                    plate_score = info.get('plate_info', {}).get('score', 0.0)
                    if plate_num:
                        ocr_list.append(f"{plate_num}:{plate_score:.3f}")

                det_debug = getattr(capture_api.detector, "last_debug_info", {}) if hasattr(capture_api, "detector") else {}
                raw_total = det_debug.get("raw_total", 0)
                pre_min_total = det_debug.get("pre_min_threshold_total", 0)
                post_nms_total = det_debug.get("post_nms_total", 0)
                raw_label_stats = det_debug.get("raw_label_stats", {})
                raw_label_stats_str = ",".join(
                    [f"{k}:{v.get('count', 0)}[{v.get('score_min', 0.0):.3f}-{v.get('score_max', 0.0):.3f}]" for k, v in raw_label_stats.items()]
                )

                debug_log_f.write(
                    f"{frame_idx}\t{car_det_num}\t{plate_det_num}\t{track_num}\t{len(ocr_list)}\t{raw_total}\t{pre_min_total}\t{post_nms_total}\t{raw_label_stats_str}\t{'|'.join(ocr_list)}\n"
                )

            if args.write_result_per_frame_bool or args.write_result_video_bool:
                # draw bbox
                # img = draw_api.draw_bbox_detect(copy.deepcopy(img), bboxes)
                # draw_img = draw_api.draw_bbox_tracker(copy.deepcopy(img), tracker_bboxes)
                draw_img = draw_api.draw_bbox_info(copy.deepcopy(img), bbox_info_list, capture_container=capture_container, capture_res_container=capture_res_container, mode='ltrb')
                # draw_img = draw_api.draw_bbox_info(draw_img, bbox_info_list, capture_container=capture_container, capture_res_container=capture_res_container, mode='ltrb')
                draw_img = draw_api.draw_bbox_state(draw_img, bbox_state_container)
                draw_img = draw_api.draw_capture_line(draw_img, capture_line_points, mode='ltrb')

            # 是否保存每一帧结果
            if args.write_result_per_frame_bool:
                output_img_path = os.path.join(args.output_video_dir, video_list[idx].replace(args.suffix, ''), video_list[idx].replace(args.suffix, '_{:0>5d}.jpg'.format(frame_idx)))
                os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
                cv2.imwrite(output_img_path, draw_img)

            # 是否保存视频结果
            if args.write_result_video_bool:
                output_video.write(draw_img)

            # 是否保存抓拍结果
            if args.write_capture_crop_bool:
                # crop capture result
                for _, capture_res_idy in capture_res_container.items():

                    if capture_res_idy['capture']['draw_bool']:
                        continue

                    track_id = capture_res_idy['track_id']
                    flage = capture_res_idy['capture']['flage']
                    capture_res_idy['capture']['draw_bool'] = True

                    for idz in range(len(capture_res_idy['capture']['img_bbox_info_list'])):
                        img_bbox_info = capture_res_idy['capture']['img_bbox_info_list'][idz]
                        img_crop = img_bbox_info['img']
                        bbox_info = img_bbox_info['bbox_info']

                        if args.demo_type == "lpr":
                            kind = capture_res_idy['plate_info']['kind']
                            num = capture_res_idy['plate_info']['num']
                            color = capture_res_idy['plate_info']['color']
                            country = capture_res_idy['plate_info']['country']
                            city = capture_res_idy['plate_info']['city']
                            car_type = capture_res_idy['plate_info']['car_type']
                            bbox_loc = [bbox_info['car_info']['roi'] if len(bbox_info['car_info']['roi']) else bbox_info['plate_info']['roi'] for bbox_info in bbox_info if bbox_info['track_id'] == track_id][0]

                            if args.country_type == "zd":
                                output_capture_path = os.path.join(args.output_video_dir, 'capture', video_list[idx].replace(args.suffix, ''), '{}_{}_{}_{}_{}_{}_{}_{}_{}.jpg'.format(frame_idx, kind, num, country, city, car_type, color, flage, idz))
                            elif args.country_type == "morocco":
                                output_capture_path = os.path.join(args.output_video_dir, 'capture', video_list[idx].replace(args.suffix, ''), '{}_{}_{}_{}_{}_{}.jpg'.format(frame_idx, kind, city, num, flage, idz))
                            else:
                                output_capture_path = os.path.join(args.output_video_dir, 'capture', video_list[idx].replace(args.suffix, ''), '{}_{}_{}_{}_{}.jpg'.format(frame_idx, num, color, flage, idz))
                        elif args.demo_type == "face":
                            landmark_degree = bbox_info[0]['face_info']['landmark_degree']
                            bbox_loc = [bbox_info['face_info']['roi'] for bbox_info in bbox_info if bbox_info['track_id'] == track_id][0]
                            output_capture_path = os.path.join(args.output_video_dir, 'capture', video_list[idx].replace(args.suffix, ''), '{}_{}_{}_{:.2f}.jpg'.format(frame_idx, flage, idz, landmark_degree))

                        # bbox_crop = img_crop[max( 0, int(bbox_loc[1]) ): min( int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(bbox_loc[3]) ), max( 0, int(bbox_loc[0]) ): min( int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(bbox_loc[2]) )]
                        img_h, img_w = img_crop.shape[:2]
                        x1 = max(0, int(bbox_loc[0]))
                        y1 = max(0, int(bbox_loc[1]))
                        x2 = min(img_w, int(bbox_loc[2]))
                        y2 = min(img_h, int(bbox_loc[3]))
                        bbox_crop = img_crop[y1:y2, x1:x2]
                        # 保存捕获结果
                        os.makedirs(os.path.dirname(output_capture_path), exist_ok=True)
                        if bbox_crop is not None and bbox_crop.size != 0:
                            cv2.imwrite(output_capture_path, bbox_crop)
            
            frame_idx += 1

            tqdm.write("{}: {}".format(video_path, str(frame_idx)))

        if debug_log_f is not None:
            debug_log_f.close()


def main():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    #######################################
    # license plate
    #######################################
    # # Chn, license plate demo
    # args.demo_type = "lpr"
    # args.country_type = "china"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/5M_230416/avi/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/5M_230416/test/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_230416/lpr_paddle_ocr/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/2M_8mm_16mm_211103/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/2M_8mm_16mm_211103/lpr_paddle_ocr/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_前方反馈_20230606/C27_16mm_2592_1920_avi/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_ZD_DUBAI_C27/5M_前方反馈_20230606/lpr_paddle_ocr/C27_16mm_2592_1920_avi/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_前方反馈_20230606/黑光C27_16mm_2592_1520_avi/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_ZD_DUBAI_C27/5M_前方反馈_20230606/lpr_paddle_ocr/黑光C27_2592_1520_avi/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_前方反馈_20230606/C40W_8mm_2688_1520_avi/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_ZD_DUBAI_C27/5M_前方反馈_20230606/lpr_paddle_ocr/C40W_8mm_2688_1520_avi/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_前方反馈_20230614/C27_16mm_2592_1920_avi/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_ZD_DUBAI_C27/5M_前方反馈_20230614/lpr_paddle_ocr/C27_16mm_2592_1920_avi/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_C37/2M_白天_16mm_2023_0330/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_C37/2M_白天_16mm_2023_0330/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/Distance_detection/Distance_detection_plate/avi/500W_new/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/Distance_detection/Distance_detection_plate/500W_new/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/camera_test/8mm_200W_c20D/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/camera_test/8mm_200W_c20D/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/M_20240327/adplus-车内前向-12mm-5MP/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/M_20240327/adplus-车内前向-12mm-5MP/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/M_20240327/电单车-爱芯B63-右前右后-8mm-4MP-25P/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/M_20240327/电单车-爱芯B63-右前右后-8mm-4MP-25P/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/M_20240327/警车车顶-白天-爱芯B63-前后-16mm-4MP/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/M_20240327/警车车顶-白天-爱芯B63-前后-16mm-4MP/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/M_20240327/警车车顶-白天-爱芯B63-前后-25mm-4MP/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/M_20240327/警车车顶-白天-爱芯B63-前后-25mm-4MP/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/M_20240327/警车车顶-白天-爱芯B63-右前右后左前左后-8mm-4MP/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/M_20240327/警车车顶-白天-爱芯B63-右前右后左前左后-8mm-4MP/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/M_20240327/警车车顶-夜间-爱芯B63-前后-16mm-4MP/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/M_20240327/警车车顶-夜间-爱芯B63-前后-16mm-4MP/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/M_20240327/警车车顶-夜间-爱芯B63-前后-25mm-4MP/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/M_20240327/警车车顶-夜间-爱芯B63-前后-25mm-4MP/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/M_20240327/警车车顶-夜间-爱芯B63-右前右后左前左后-8mm-4MP/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/M_20240327/警车车顶-夜间-爱芯B63-右前右后左前左后-8mm-4MP/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/5M_20240402_展会/avi/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_20240402_展会/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/5M_20240403_展会/avi_400W_cap_res_face/"
    # # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/5M_20240403_展会/avi_500W/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_20240403_展会_final_car/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/400w_240614_侧向镜头_检测尺寸过小/avi/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/400w_240614_侧向镜头_检测尺寸过小_caffe/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/400w_240712_侧向镜头_车牌漏检误检视频/avi_分类/大车牌漏报/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/400w_240712_侧向镜头_车牌漏检误检视频/avi_分类/ocr不稳漏/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/400w_240712_侧向镜头_车牌漏检误检视频/avi_分类/绿牌错误/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/400w_240712_侧向镜头_车牌漏检误检视频/avi_分类/难例字符错误/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/400w_240712_侧向镜头_车牌漏检误检视频/avi_分类/倾斜识别错误/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/400w_240712_侧向镜头_车牌漏检误检视频/倾斜识别错误/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/400w_240725_侧向镜头_客户测试视频/avi/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/400w_240725_侧向镜头_客户测试视频/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/4M5M_20240327_多镜头/demo/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZG_ZHJYZ_detection/4M5M_20240327_多镜头/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/ZG_ZHJYZ_detection/原始素材/ZG_TQ/5M_16mm_6M_白_0506/demo_cut/"
    # # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/ZG_ZHJYZ_detection/5M_16mm_6M_白_0506/"
    # # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/demo/CHANNEL_04_20200209140928_20200209140938/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/demo/CHANNEL_05_20200209140409_20200209140419/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/RM_SchBus_Police_Capture_Raw_Video/demo/"

    # # 8mm_广州出租
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_CHUZU/2M_4mm_test/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_CHUZU/2M_4mm_test/"
    # # args.video_dir = "/mnt/huanyuan/temp/8mm_广州出租/20240202_20240204/验证1小时抓拍/2m_4mm_白天_20240204"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_CHUZU/验证1小时抓拍/2m_4mm_白天_20240204_strict/"
    # # args.video_dir = "/mnt/huanyuan/temp/8mm_广州出租/20240202_20240204/验证1小时抓拍/2m_4mm_夜间_20240204"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_CHUZU/验证1小时抓拍/2m_4mm_夜间_20240204_strict/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_CHUZU/2M_20240702"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_CHUZU/2M_20240702_0/"
    # args.video_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/Taxi_1080p/2M_20250617/mp4"
    # args.output_video_dir = "/yuanhuan/data/temp_result/RM_SchBus_Police_Capture_Raw_Video/Taxi_1080p/2M_20250617"

    # # Brazil, license plate demo
    # args.demo_type = "lpr"
    # args.country_type = "brazil"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_BM_Brazil_C27/5M_白天_2022_1026/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_BM_Brazil_C27/5M_白天_2023_0110/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_BM_Brazil_C27/test/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/RoadCap_BM_Brazil_C27/avi_test/CH11/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/RoadCap_BM_Brazil_C27/avi_test/CH12/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_BM_Brazil_C27/5M_白天_2022_1026_yolov6_paddle/"

    # # America C37, license plate demo
    # args.demo_type = "lpr"
    # args.country_type = "us"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_BM_C37/2M_白天_16mm_2023_0811/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_BM_C37/2M_白天_16mm_2023_1108_1109/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_BM_C37/2M_白天_16mm_20240124_20240125/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_BM_C37/2M_白天_16mm_Alabama_2022/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_BM_C37/2M_白天_16mm_Alabama_2022_ALL_US/"

    # # Chile C40W, license plate demo
    # args.demo_type = "lpr"
    # args.country_type = "chile"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CHILE_C40W/2M_8mm_2023_1201_1205/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CHILE_C40W/2M_8mm_2023_1201_1205/"

    # # Europe C37, license plate demo
    # args.demo_type = "lpr"
    # args.country_type = "eu"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_EU_C37/2M_16mm_20230911_20230913/avi/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_EU_C37/2M_16mm_20230911_20230913/test/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_EU_C37/2M_16mm_20230911_20230913_new/"

    # zd, license plate demo
    # args.demo_type = "lpr"
    # args.country_type = "zd"
    # args.video_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_白天_侧向_0615/avi/"
    # args.output_video_dir = "/yuanhuan/data/temp_result/POLICE_ZD_DUBAI_C27/5M_白天_侧向_0615/"
    # args.video_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_白天_后向_0615/avi/"
    # args.output_video_dir = "/yuanhuan/data/temp_result/POLICE_ZD_DUBAI_C27/5M_白天_后向_0615/"
    # args.video_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_夜晚_侧向_0615/avi/"
    # args.output_video_dir = "/yuanhuan/data/temp_result/POLICE_ZD_DUBAI_C27/5M_夜晚_侧向_0615/"
    # args.video_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_夜晚_后向_0615/avi/"
    # args.output_video_dir = "/yuanhuan/data/temp_result/POLICE_ZD_DUBAI_C27/5M_夜晚_后向_0615/"
    # args.video_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_SHATE_C27/5M_白天_前向_20230208/avi"
    # args.video_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_SHATE_C27/5M_白天_前向_20230209/avi"
    # args.output_video_dir = "/yuanhuan/data/temp_result/POLICE_ZD_SHATE_C27/5M_白天_前向_20230208/"

    # # morocco, license plate demo
    # args.demo_type = "lpr"
    # args.country_type = "morocco"
    # args.video_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_MOROCCO_C27/2M5M_白天_0329/avi/avi_2592_1920"
    # # args.video_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_MOROCCO_C27/2M5M_白天_0329/avi/avi_2592_1520"
    # args.output_video_dir = "/yuanhuan/data/temp_result/POLICE_MOROCCO_C27/2M5M_白天_0329_new_detect/2592_1520_a_z"

    # ######################################
    # # face
    # ######################################
    # # zg, face demo
    # args.demo_type = "face"
    # args.country_type = "none"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/5M_230416/avi/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_230416/face_avi/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_230416/face_avi_landmark/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_230416/face_avi_landmark_center_offset/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_230416/face_avi_landmark_sigmoid_center_offset_qa_0605/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_230416/face_avi_landmark_degree/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_230416/face_avi_landmark_degree_cls/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_230416/face_avi_landmark_sigmoid_center_offset_yolox/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/5M_230811_人脸/avi/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/5M_230811_人脸/face_avi_landmark_degree_cls/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_前方反馈_20230606/C27_16mm_2592_1920_avi/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_ZD_DUBAI_C27/5M_前方反馈_20230606/yolov6_landmark_degree_cls/C27_16mm_2592_1920_avi/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_前方反馈_20230606/黑光C27_16mm_2592_1520_avi/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_ZD_DUBAI_C27/5M_前方反馈_20230606/yolov6_landmark_degree_cls/黑光C27_2592_1520_avi/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_前方反馈_20230606/C40W_8mm_2688_1520_avi/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_ZD_DUBAI_C27/5M_前方反馈_20230606/yolov6_landmark_degree_cls/C40W_8mm_2688_1520_avi/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_前方反馈_20230614/C27_16mm_2592_1920_avi/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_ZD_DUBAI_C27/5M_前方反馈_20230614/yolov6_landmark_degree_cls/C27_16mm_2592_1920_avi/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/5M_前方反馈_人脸抓拍_20230731/avi/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_ZD_DUBAI_C27/5M_前方反馈_人脸抓拍_20230731/yolov6_landmark_degree_cls/avi/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/M_20240327/adplus-车内前向-12mm-5MP/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/M_20240327_face_person_sort_face/adplus-车内前向-12mm-5MP/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/M_20240327/电单车-爱芯B63-右前右后-8mm-4MP-25P/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/M_20240327_face_person_sort_face/电单车-爱芯B63-右前右后-8mm-4MP-25P/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/M_20240327/警车车顶-白天-爱芯B63-前后-16mm-4MP/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/M_20240327_face_person_sort_face/警车车顶-白天-爱芯B63-前后-16mm-4MP/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/M_20240327/警车车顶-白天-爱芯B63-前后-25mm-4MP/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/M_20240327_face_person_sort_face/警车车顶-白天-爱芯B63-前后-25mm-4MP/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/M_20240327/警车车顶-白天-爱芯B63-右前右后左前左后-8mm-4MP/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/M_20240327_face_person_sort_face/警车车顶-白天-爱芯B63-右前右后左前左后-8mm-4MP/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/M_20240327/警车车顶-夜间-爱芯B63-前后-16mm-4MP/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/M_20240327_face_person_sort_face/警车车顶-夜间-爱芯B63-前后-16mm-4MP/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/M_20240327/警车车顶-夜间-爱芯B63-前后-25mm-4MP/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/M_20240327_face_person_sort_face/警车车顶-夜间-爱芯B63-前后-25mm-4MP/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/M_20240327/警车车顶-夜间-爱芯B63-右前右后左前左后-8mm-4MP/"
    # # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/POLICE_CN_ZG_HCZP/M_20240327_face_person_sort_face/警车车顶-夜间-爱芯B63-右前右后左前左后-8mm-4MP/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/5M_20240403_展会/avi_400W_cap_res/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/5M_20240403_展会/avi_500W/"
    # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/400w_240712_侧向镜头_人脸漏检误检视频/big_face_20240712/avi/"
    # # args.video_dir = "/mnt/huanyuan2/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/400w_240712_侧向镜头_人脸漏检误检视频/wheel_face_20240712/avi/"
    # args.output_video_dir = "/mnt/huanyuan/temp/pc_demo/RM_SchBus_Police_Capture_Raw_Video/POLICE_CN_ZG_HCZP/400w_240712_侧向镜头_人脸漏检误检视频/big_face_20240712/"


    #######################################
    # Huan Wei
    #######################################
    # Chn, license plate demo
    # args.demo_type = "lpr"
    # args.country_type = "huanwei"
    # args.video_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/HuanWei_Argentina/test/test_20240912_all/side/"
    # args.video_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/HuanWei_Argentina/test/test_20240912_all/front/"
    # args.video_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/HuanWei_Argentina/test/test_20250920/side/"
    # args.video_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/HuanWei_Argentina/test/test_20250920/front/"
    # args.output_video_dir = "/yuanhuan/data/temp_result/HuanWei_Argentina/test_20240912_all/side_ROTATE/"

    #######################################
    # Mexican
    #######################################
    # Chn, license plate demo
    args.demo_type = "lpr"
    args.country_type = "mexican"
    # args.video_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/HuanWei_Argentina/test/test_20240912_all/side/"
    # args.video_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/HuanWei_Argentina/test/test_20240912_all/front/"
    # args.video_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/HuanWei_Argentina/test/test_20250920/side/"
    args.video_dir = "/yuanhuan/data/temp_result/test/test_20260420/"
    args.output_video_dir = "/yuanhuan/data/temp_result/test/test_20260420/front"

    # args.suffix = '.avi'
    args.suffix = '.mp4'
    
    # 每多少帧执行一次抓拍算法
    # args.frame_step = 4
    args.frame_step = 1
    # 每个视频清空状态
    args.clear_state_per_video = True
    # 是否保存视频结果
    args.write_result_video_bool = True
    # 是否保存每一帧结果
    # args.write_result_per_frame_bool = True
    args.write_result_per_frame_bool = False
    # 是否保存抓拍结果
    args.write_capture_crop_bool = True
    # 是否记录逐帧调试日志（检测/OCR）
    args.write_debug_log_per_frame_bool = True

    inference_video(args)

    # # for 循环
    # # # zd, license plate demo
    # args.demo_type = "lpr"
    # args.country_type = "zd"

    # video_list = ["5M_白天_侧向_0615", "5M_白天_后向_0615", "5M_夜晚_侧向_0615", "5M_夜晚_后向_0615"]
    # for video_name in video_list:
    #     args.video_dir = "/yuanhuan/data/image/RM_SchBus_Police_Capture_Raw_Video/POLICE_ZD_DUBAI_C27/{}/avi/".format(video_name)
    #     args.output_video_dir = "/yuanhuan/data/temp_result/POLICE_ZD_DUBAI_C27/{}/".format(video_name)

    #     args.suffix = '.avi'
    #     # args.suffix = '.mp4'
        
    #     # 每多少帧执行一次抓拍算法
    #     # args.frame_step = 4
    #     args.frame_step = 1
    #     # 每个视频清空状态
    #     args.clear_state_per_video = True
    #     # 是否保存视频结果
    #     args.write_result_video_bool = True
    #     # 是否保存每一帧结果
    #     args.write_result_per_frame_bool = True
    #     # args.write_result_per_frame_bool = False
    #     # 是否保存抓拍结果
    #     args.write_capture_crop_bool = True

    #     inference_video(args)


if __name__ == '__main__':
    main()