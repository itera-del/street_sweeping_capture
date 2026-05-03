import argparse
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

from sklearn.model_selection import train_test_split

# sys.path.insert(0, '/home/huanyuan/code/demo')
sys.path.insert(0, '/yuanhuan/code/demo')
from Image.Basic.utils.folder_tools import *


def split(args):
    
    # mkdir
    create_folder(os.path.dirname(args.to_trainval_file))

    if os.path.exists(args.input_csv_path):
        # pd
        data_pd = pd.read_csv(args.input_csv_path)

        with open(args.to_trainval_file, "w") as f:

            for _, row in tqdm(data_pd.iterrows(), total=len(data_pd)):
                if row['split_type'] == "train" or row['split_type'] == "val":
                    if os.path.exists(row["roi_img_path"]) == False:
                        print("not exists: {}".format(row["roi_img_path"]))
                        continue
                    f.write('{}'.format(row["roi_img_path"]))
                    f.write("\n")

        with open(args.to_train_file, "w") as f:

            for _, row in tqdm(data_pd.iterrows(), total=len(data_pd)):
                if row['split_type'] == "train":
                    if os.path.exists(row["roi_img_path"]) == False:
                        print("not exists: {}".format(row["roi_img_path"]))
                        continue
                    f.write('{}'.format(row["roi_img_path"]))
                    f.write("\n")

        with open(args.to_val_file, "w") as f:

            for _, row in tqdm(data_pd.iterrows(), total=len(data_pd)):
                if row['split_type'] == "val":
                    if os.path.exists(row["roi_img_path"]) == False:
                        print("not exists: {}".format(row["roi_img_path"]))
                        continue
                    f.write('{}'.format(row["roi_img_path"]))
                    f.write("\n")

        with open(args.to_test_file, "w") as f:

            for _, row in tqdm(data_pd.iterrows(), total=len(data_pd)):
                if row['split_type'] == "test":
                    if os.path.exists(row["roi_img_path"]) == False:
                        print("not exists: {}".format(row["roi_img_path"]))
                        continue
                    f.write('{}'.format(row["roi_img_path"]))
                    f.write("\n")
    else:
        jpg_dir = os.path.join(args.input_dir, "Images")
        jpg_list = np.array(os.listdir(jpg_dir))
        jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]

        trainval_list, test_list = train_test_split(jpg_list, test_size=args.test_size, random_state=0)
        train_list, val_list = train_test_split(trainval_list, test_size=args.val_size, random_state=0)
        print("length: trainval: {}, train: {}, val: {}, test: {}, all: {}".format(len(trainval_list), len(train_list), len(val_list), len(test_list), (len(train_list) + len(val_list) + len(test_list))))

        with open(args.to_trainval_file, "w") as f:
            for jpg in trainval_list:
                f.write(os.path.join(jpg_dir, jpg))
                f.write("\n")

        with open(args.to_train_file, "w") as f:
            for jpg in train_list:
                f.write(os.path.join(jpg_dir, jpg))
                f.write("\n")

        with open(args.to_val_file, "w") as f:
            for jpg in val_list:
                f.write(os.path.join(jpg_dir, jpg))
                f.write("\n")

        with open(args.to_test_file, "w") as f:
            for jpg in test_list:
                f.write(os.path.join(jpg_dir, jpg))
                f.write("\n")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="morocco_20250327") 
    parser.add_argument('--ocr_name', type=str, default="plate_Morocco_mask_202504") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/RM_ANPR/training/")  
    args = parser.parse_args()

    args.input_dir = os.path.join(args.input_dir, args.ocr_name, args.date_name)

    print("data train test split.")
    print("date_name: {}".format(args.date_name))
    print("ocr_name: {}".format(args.ocr_name))
    print("input_dir: {}".format(args.input_dir))

    date_name = args.date_name
    # *_seg_res.csv 2 *.csv
    date_name = date_name.replace("_seg_res", "")
    print(date_name)

    args.input_csv_path = os.path.join(args.input_dir, '{}.csv'.format(date_name))
    args.to_trainval_file = os.path.join(args.input_dir, "ImageSets/Main/trainval.txt")
    args.to_train_file = os.path.join(args.input_dir, "ImageSets/Main/train.txt")
    args.to_val_file = os.path.join(args.input_dir, "ImageSets/Main/val.txt")
    args.to_test_file = os.path.join(args.input_dir, "ImageSets/Main/test.txt")

    args.test_size = 0.1
    args.val_size = 0.1

    split(args)