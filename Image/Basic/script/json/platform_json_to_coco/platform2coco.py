from argparse import ArgumentParser
import os.path as osp
import numpy as np
import cv2

import sys 
sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.script.json.platform_json_to_coco.utils import get_files, safe_make_dirs, get_img_label_pair, cv_fill_area
from Basic.script.json.platform_json_to_coco.prase_target import PlatformTargetParser
from Basic.script.json.platform_json_to_coco.cocojsonwriter import JsonWriter


def addAnnItem(class_name, bbox=None, segmentation=None):
    assert bbox is not None or segmentation is not None
    annotation_item = dict()
    if bbox is None:
        segmentation_mixup = np.concatenate(segmentation, axis=0)
        xywh = np.array(cv2.boundingRect(segmentation_mixup))
        json_segmentation = [seg.reshape(-1).tolist() for seg in segmentation]
        annotation_item['segmentation'] = json_segmentation
    elif segmentation is None:
        xywh = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
    annotation_item['area'] = int(xywh[2] * xywh[3])
    annotation_item['bbox'] = xywh.astype(int).tolist()
    annotation_item['class_name'] = class_name
    return annotation_item

class pars():
    def __init__(self, ):
        self.mode = 'test'
        self.input_path = '/yuanhuan/data/image/RM_Capture/original_dupes_0_1_0_95'
        self.seg_class_names = [ ]
        self.det_class_names = ['car', 'bus', 'truck', 'bicycle', 'motorcycle', 'license', ]
        # self.det_class_names = ['bus', 'car', 'truck', 'person', 'bicyclist', 'motorcyclist', ]


if __name__ == '__main__':
    
    date = [
        'original_dupes_0_1_0_95', 
    ]
    
    for d in date:
    
        args = pars()
        args.input_path = '{}/{}'.format(args.input_path, d)
        
        for m in ['train', 'test']:
            args.mode = m
            
            input_path = osp.join(args.input_path, args.mode)
            # build input
            images = get_files(input_path, target='.jpg')
            jsons = get_files(input_path, target='.json')

            out_pairs = get_img_label_pair(images, jsons)
            print(len(out_pairs))

            target_parser = PlatformTargetParser(args.seg_class_names, args.det_class_names)
            # build output 
            annotations = osp.join(args.input_path, "annotations")
            safe_make_dirs(annotations)
            if args.seg_class_names:
                seg_path = osp.join(args.input_path, "stuffthingmaps", args.mode)
                safe_make_dirs(seg_path)
            # 如果json file没有删掉，请修改   mode='w'
            writer = JsonWriter(osp.join(annotations, "instances_" + args.mode + ".json"), args.det_class_names, mode='w')

            # 
            for i, (img_name, json_name) in enumerate(out_pairs):
                json_t = target_parser.parse(json_name)

                # mapping to coco
                image_info = {}
                image_info["file_name"] = osp.basename(img_name)
                image_info["height"] = json_t["height"]
                image_info["width"] = json_t["width"]

                annotations_info = []
                for instacne in json_t["annotations"]:
                    if instacne['type'] == 'rectangle':
                        annotations_info.append(addAnnItem(instacne["class_name"], np.array(instacne["points"]).reshape(-1)))
                writer.insert_image(image_info, annotations_info)

                # stuff thing create 
                if args.seg_class_names:
                    mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
                    for index, target in enumerate(args.seg_class_names):
                        line_points = []
                        for instacne in json_t["annotations"]:
                            if instacne['type'] == 'polygon' and instacne["class_name"] == target:
                                line_points.append(instacne["points"])
                        mask = cv_fill_area(mask, line_points, color=index + 1)
                    # tmp = osp.join(seg_path, osp.splitext(image_info["file_name"])[0] + ".png")
                    cv2.imwrite(osp.join(seg_path, osp.splitext(image_info["file_name"])[0] + ".png"), mask)
            writer.write_json()
