from src.model_new import MODEL_NEW
import argparse
import os
import numpy as np
from typing import Union
from pathlib import Path
import cv2

def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def getlist(dir, extension):
    """get a list of all files based on the specified file suffix name, like .jpg/.pts etc.
    :param dir:
        root diectory path.
    :param extension:
        specified file suffix name, like .jpg/.pts etc.
    :return:
        files list.
    """
    files_list = []
    files_name = []
    if not isinstance(dir, list):
        dir = [dir]
    if isinstance(extension, str):
        extension = [extension]

    for dir_item in dir:
        print(dir_item)
        for root, dirs, files in os.walk(dir_item, topdown=False):
            for name in dirs:
                # print(os.path.join(root, name))
                pass
            for name in files:
                filename, ext = os.path.splitext(name)
                if ext in extension:
                    files_list.append(os.path.join(root, name))
                    files_name.append(name)
    return files_list, files_name

def group_batch(images_path, group_len=900000000, ext=['.jpg', '.png', '.jpeg']):
    assert os.path.exists(images_path)
    imgs_list, imgs_names = getlist(images_path, ext)
    
    total_len = len(imgs_list)
    part_num = total_len//group_len
    if total_len % group_len == 0:
        indices = [group_len] * (total_len//group_len)
    else:
        indices = [group_len] * (total_len//group_len + 1)
        indices[-1] = total_len - part_num * group_len
        
    print(f"=> Fastdup, total load {total_len} images, split {indices} part.")
    grouped_images = []
    idx = 0
    for jdx in indices:
        grouped_images.append(imgs_list[idx: idx + jdx])
        idx = idx + jdx

    return grouped_images


def parse_args():
    parser = argparse.ArgumentParser(description='Run Auto_Dinov2_Processing')
    parser.add_argument('--task_id', type=int, default=1, help='task style id')
    parser.add_argument('--images_path', type=str, default="/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate/JPEGImages", help='root path to images')
    parser.add_argument('--results_path', type=str, default="/yuanhuan/data/image/RM_ADAS_AllInOne/allinone_w_licenseplate/res_cc_thres_0_90/", help='root folder to save results')
    parser.add_argument('-q', '--query_image', type=str, help='path to a query image file')
    parser.add_argument('-s', '--cc_threshold', type=float, default=0.95, help='image similar for duplicate')
    parser.add_argument('-d', '--is_dinov2', type=bool, default=False, help='whether use dinov2 to get feature vectors')
    parser.add_argument('-n', '--num', type=int, default=20, help='save n image in processing method')
    parser.add_argument('-k', '--topk', type=int, default=10,
                        help='The number of nearest neighbors to search for image')

    parser.add_argument('-c', '--crop_coord', nargs='+', type=int, default=[],
                        help='Crop coordinates in the format: x1 y1 x2 y2')
    parser.add_argument('-t', '--crop_temp_path', type=str, default=None, help='root folder to save crop temp results')
    args = parser.parse_args()
    return args


class Run():
    def __init__(self,args, images_path: Union[str, Path, list] = None, results_path: Union[str, Path] = None):
        '''
        :param imagepath: root path of checking images
        :param results_path:save path of results
        :param id_dinov2: whether use dinov2 to get feature vectors
        :param num: data_processing classmethod to save num images
        '''
        # self.images_path = images_path
        # self.results_path = args.results_path
        self.top_k = args.topk
        self.num = args.num
        self.cc_threshold = args.cc_threshold
        self.query_path = args.query_image
        self.is_crop_image = args.crop_coord and args.crop_temp_path
        self.model = MODEL_NEW(images_path, results_path, is_dinov2=args.is_dinov2, is_crop_image=self.is_crop_image)
        # if len(os.listdir(os.path.join(self.results_path, 'work_dir'))) == 0:
        self.model.create_model(self.cc_threshold)
        print('###******create model sucess********###')

    # get duplicate.html、components.html、outlier.html、cluster.png
    # def forward_vis(self):
    #     self.model.vis_processing()

    def forward_processing(self):
        self.model.vis_processing()
        self.model.data_processing(self.num)

    def forward_retrieval(self):
        self.model.retrieval(self.top_k, self.query_path)

    # delete duplicate images in root images path
    def forward_delete_duplicate(self):
        self.model.delete_duplicate_images()

if __name__ == '__main__':
    args = parse_args()
    if args.crop_coord and args.crop_temp_path:
        print(f"=> process crop temp image save to {args.crop_temp_path}.")
        safe_mkdir(args.crop_temp_path)
        x1, y1, x2, y2 = args.crop_coord
        for i in args.images_path:
            img = cv2.imread(i)
            crop_image = img[y1:y2, x1:x2, :]
            cv2.imwrite(os.path.join(args.crop_temp_path, os.path.basename(i)), crop_image)
        print('=> Crop temp image save done.')
        args.images_path = args.crop_temp_path
    grouped_images = group_batch(args.crop_temp_path) if args.crop_coord and args.crop_temp_path else group_batch(args.images_path)
    group_idx = 1
    print("len(grouped_images):{}".format(len(grouped_images)))
    imgs = grouped_images[0]
    print(f"=> process {group_idx} group images.")
    group_name = f"batch_{group_idx}"
    # results_path = os.path.join(args.results_path, group_name)
    results_path = args.results_path
    safe_mkdir(results_path)
    model = Run(args, imgs, results_path)        
    if args.task_id == 0:
        model.forward_processing()
    elif args.task_id == 1:
        model.forward_delete_duplicate()
    else:
        model.forward_retrieval()        
    group_idx += 1
    print('=> Done.')